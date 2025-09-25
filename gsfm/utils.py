import contextlib
import importlib
import inspect
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime

import lightning as L
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import yaml
from huggingface_hub import HfApi, PyTorchModelHubMixin, hf_hub_download
from tqdm.auto import tqdm

def if_not_exists(local, invalidate=False):
  def decorator(fn):
    if not pathlib.Path(local).exists() or invalidate:
      pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
      fn(local)
  return decorator

def wget(url, local):
  pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
  with tqdm(desc=f"Downloading", unit="B", unit_scale=True) as pbar:
    urllib.request.urlretrieve(url, local, reporthook=lambda b, bsize, tsize: pbar.update(b*bsize-pbar.n))
  return local

def cached_wget(url, local, invalidate=False):
  @if_not_exists(local, invalidate=invalidate)
  def _(local): wget(url, local)
  return local

def read_gmt(local):
  with pathlib.Path(local).open('r') as fr:
    for line in fr:
      line_split = line.rstrip('\r\n').split('\t')
      if len(line_split) < 3: continue
      term, desc, *genes = line_split
      yield term, desc, list(filter(None, genes))

def read_gmt_background(f):
  all_genes = set()
  for _, _, genes in read_gmt(f):
    all_genes.update(genes)
  return all_genes

def chunked(L, cs):
  C = []
  for item in L:
    C.append(item)
    if len(C) >= cs:
      yield C
      C = []
  if C: yield C

def norm(x: pd.Series, eps=1e-6):
  if isinstance(x, pd.Series):
    mu, std = st.norm.fit(x.values)
    return pd.Series(st.norm.cdf((x.values-mu)/(std+eps)), index=x.index)
  else:
    mu, std = st.norm.fit(x)
    return st.norm.cdf((x-mu)/(std+eps))

def ES(y_true, y_score):
  ''' The peak of a random walk, best possible value is 1
  worst value is 0
  '''
  i = np.argsort(y_score)
  n_pos = y_true.sum()
  n_neg = y_true.shape[0]-n_pos
  sorted_y_true = y_true[i][::-1]
  walk = np.zeros_like(y_score)
  if n_pos: walk = sorted_y_true/n_pos
  if n_neg: walk -= (1-sorted_y_true)/n_neg
  walk = np.cumsum(walk)
  return np.max(walk)

def partition(L, y_ratio):
  p = int(len(L) * (1-y_ratio))
  return L[:p], L[p:]

def multi_hot(indices: list[int], num_classes: int):
  # return torch.nn.functional.one_hot(indices, num_classes=num_classes).max(0).values
  x = torch.zeros(num_classes)
  x[indices] = 1
  return x

def multihot_tensor(indices: torch.Tensor, num_classes: int, dtype=torch.int64, device=None):
  *bs, _ = indices.shape
  return torch.zeros((*bs, num_classes,), device=device, dtype=dtype).scatter(1, indices, 1)

def shuffle_tensor(input: torch.Tensor):
  *_, dim = input.shape
  random_indices = torch.multinomial(torch.ones_like(input, dtype=torch.float), num_samples=dim, replacement=False)
  return torch.gather(input, -1, random_indices)

def partition_padded_tensor(indices: torch.Tensor, partition: float = 0.5, padding_idx = 0, device=None):
  ''' Equivalent to partition of a list but works with multiple padded tensors
  '''
  n_non_padded_indices = (indices != padding_idx).sum(1)
  shuffled_indices = shuffle_tensor(indices)
  n_non_padded_indices_to_left = torch.cumsum(shuffled_indices != padding_idx, 1)
  mask = (n_non_padded_indices_to_left < (n_non_padded_indices * (1-partition))[:, None]).bool()
  x = torch.where(mask, shuffled_indices, padding_idx)
  y = torch.where(~mask, shuffled_indices, padding_idx)
  return x, y

def mp_run(fn, *args, **kwargs):
  proc = mp.get_context('spawn').Process(target=fn, args=args, kwargs=kwargs)
  proc.start()
  proc.join()
  assert proc.exitcode == 0

class MLP(torch.nn.Module):
  def __init__(self, *dims, activation=torch.nn.ReLU, dropout=0.2):
    super().__init__()
    activation = activation()
    dropout = torch.nn.Dropout(dropout)
    self.layers = torch.nn.ModuleList([
      layer
      for a, b in zip(dims, dims[1:])
      for layer in (
        torch.nn.Linear(a, b),
        activation,
        dropout,
      )
    ][:-2]) # the last layer doesn't need activation/dropout
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class Vocab:
  def __init__(self, vocab, default_index=0):
    self.vocab = vocab
    self.default_index = default_index
    self.lookup = {token: i for i, token in enumerate(vocab)}

  def __call__(self, sentence):
    return [self.lookup.get(token, self.default_index) for token in sentence]

  @staticmethod
  def build_vocab_from_iterator(it, min_freq=1, specials=[], special_first=True):
    vocab = []
    if special_first:
      vocab += specials
    tokens = Counter()
    for sentence in it:
      tokens.update(sentence)
    for token, freq in tokens.most_common():
      if freq < min_freq: continue
      vocab.append(token)
    if not special_first:
      vocab += specials
    return Vocab(vocab)

  def set_default_index(self, default_index):
    self.default_index = default_index

  def __len__(self):
    return len(self.vocab)

  def __reduce__(self):
    return (Vocab, (self.vocab,))

  def save_txt(self, filename):
    with open(filename, 'w') as fw:
      for token in self.vocab:
        print(token, file=fw)

  @staticmethod
  def from_txt(filename):
    with open(filename, 'r') as fr:
      return Vocab([line for line in map(str.rstrip, fr) if line])

  def save(self, filename):
    with open(filename, 'wb') as fw:
      pickle.dump(self, fw)

  @staticmethod
  def from_file(filename):
    with open(filename, 'rb') as fr:
      return pickle.load(fr)

  @staticmethod
  def from_pretrained(repo_id: str, path_in_repo='vocab.txt'):
    vocab_txt = hf_hub_download(
      repo_id=repo_id,
      filename=path_in_repo,
    )
    return Vocab.from_txt(vocab_txt)

  def push_to_hub(self, repo_id: str, path_in_repo='vocab.txt'):
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = pathlib.Path(tmpdir)
      self.save_txt(tmpdir/'vocab.txt')
      return api.upload_file(path_or_fileobj=tmpdir/'vocab.txt', repo_id=repo_id, path_in_repo=path_in_repo)

class LightningModuleEx(
  L.LightningModule,
  PyTorchModelHubMixin,
  tags=["gene", "gene set", "bioinformatics"],
):
  '''
  We bake some useful information about where the model source is and when
  '''
  def __init__(self, name=None, file=None, creation_timestamp=None):
    super().__init__()
    now = datetime.now()
    self.save_hyperparameters(dict(
      name=f"{self.__class__.__name__}_{now.strftime(r'%Y%m%d')}" if name is None else name,
      file=inspect.getabsfile(self.__class__) if file is None else file,
      creation_timestamp=now.isoformat() if creation_timestamp is None else creation_timestamp,
    ))

@contextlib.contextmanager
def from_directory(dir):
  orig_dir = pathlib.Path.cwd()
  os.chdir(dir)
  sys.path.insert(0, dir)
  orig_modules = {*sys.modules.keys()}
  try:
    yield
  finally:
    for key in ({*sys.modules.keys()} - orig_modules):
      if dir in (getattr(sys.modules[key], '__file__', None) or ''):
        del sys.modules[key]
    sys.path.remove(dir)
    os.chdir(orig_dir)

def import_class_path(class_path: str):
  mod, _, cls = class_path.rpartition('.')
  return getattr(importlib.import_module(mod), cls)

def model_config_from_ckpt(ckpt: str):
  ckpt_path = pathlib.Path(ckpt)
  with (ckpt_path.parent.parent / 'config.yaml').open('r') as fr:
    return yaml.safe_load(fr)

def model_tokenizer_from_ckpt(ckpt: str, config: dict = None, map_location=torch.device('cuda')):
  config = model_config_from_ckpt(ckpt) if config is None else config
  model = import_class_path(config['model']['class_path']).load_from_checkpoint(ckpt, map_location=map_location)
  model.eval()
  data = import_class_path(config['data']['class_path'])()
  data.setup('validate')
  return model, data.vocab

def model_tokenizer_from_ckpt_from_directory(ckpt: str, config: dict = None, map_location=torch.device('cuda')):
  with from_directory(str(pathlib.Path(ckpt).parent.parent.parent.parent)):
    return model_tokenizer_from_ckpt(ckpt, config, map_location)

def get_ncbi_lookup():
  human_geneinfo = pd.read_csv('data/Homo_sapiens.gene_info.gz', sep='\t')
  mouse_geneinfo = pd.read_csv('data/Mus_musculus.gene_info.gz', sep='\t')
  geneinfo = pd.concat([human_geneinfo, mouse_geneinfo]).drop_duplicates(['GeneID'])

  def maybe_split(record):
    ''' NCBI Stores Nulls as '-' and lists '|' delimited
    '''
    if record in {'', '-'}:
      return set()
    return set(record.split('|'))
  #
  def supplement_dbXref_prefix_omitted(ids):
    ''' NCBI Stores external IDS with Foreign:ID while most datasets just use the ID
    '''
    for id in ids:
      # add original id
      yield id
      # also add id *without* prefix
      if ':' in id:
        yield id.split(':', maxsplit=1)[1]
  #
  geneinfo['All_synonyms'] = [
    set.union(
      maybe_split(row['Symbol']),
      maybe_split(row['Symbol_from_nomenclature_authority']),
      maybe_split(str(row['GeneID'])),
      maybe_split(row['Synonyms']),
      maybe_split(row['Other_designations']),
      maybe_split(row['LocusTag']),
      set(supplement_dbXref_prefix_omitted(maybe_split(row['dbXrefs']))),
    )
    for _, row in geneinfo.iterrows()
  ]

  synonyms, symbols = zip(*{
    (synonym, row['Symbol'])
    for _, row in geneinfo.iterrows()
    for synonym in row['All_synonyms']
  })
  ncbi_lookup = pd.Series(symbols, index=synonyms)
  index_values = ncbi_lookup.index.value_counts()
  ambiguous = index_values[index_values > 1].index
  ncbi_lookup_disambiguated = ncbi_lookup[(
    (ncbi_lookup.index == ncbi_lookup) | (~ncbi_lookup.index.isin(ambiguous))
  )]
  return ncbi_lookup_disambiguated

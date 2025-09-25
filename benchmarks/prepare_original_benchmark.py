#%%
import pandas as pd
import numpy as np
import functools
import pathlib
import torch
import random
import concurrent.futures
from tqdm.auto import tqdm
from utils import read_gmt, multi_hot, partition, chunked, ES, model_config_from_ckpt, model_tokenizer_from_ckpt, read_gmt_background, if_not_exists, norm, mp_run
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from sklearn.dummy import DummyClassifier

#%%
batch_size = 128

class DummyVocab:
  def __init__(self, vocab, special=['<unk>', '<pad>']):
    self.vocab = [*special, *vocab]
    self.lookup = {t: i for i, t in enumerate(self.vocab)}
  def __len__(self):
    return len(self.vocab)
  def __call__(self, tokens):
    return [self.lookup.get(token, self.lookup['<unk>']) for token in tokens]

def enumerate_x_ys(*, library, vocab):
  for term, _, genes in read_gmt(f'data/{library}.gmt'):
    if len(genes) < 5: continue
    for i in range(5):
      np.random.shuffle(genes)
      x, y = partition(genes, 0.5)
      x_true = multi_hot(torch.tensor(vocab(x)), num_classes=len(vocab))
      y_true = multi_hot(torch.tensor(vocab(y)), num_classes=len(vocab)).numpy()
      if x_true.sum() < 5 or y_true.sum() < 5: continue
      yield term, torch.tensor(vocab(x)), x_true, y_true

def model_benchmark_from_ckpt(ckpt, library):
  config = model_config_from_ckpt(ckpt)
  model_base = pathlib.Path(ckpt).parent.parent.parent.parent.name
  model_version = pathlib.Path(ckpt).parent.parent.name
  @if_not_exists(f"data/classic-benchmark/2025-09-08/{library}/model/{model_base}-{model_version}/scores.tsv")
  def _(file):
    model, vocab = model_tokenizer_from_ckpt(ckpt, config=config, map_location=torch.device('cuda'))
    scores = []
    library_background = read_gmt_background(f'data/{library}.gmt')
    background = multi_hot(torch.tensor(vocab(library_background)), num_classes=len(vocab)).numpy().astype(bool)
    for rows in chunked(enumerate_x_ys(library=library, vocab=vocab), batch_size):
      batch_term, batch_x, batch_x_true, batch_y_true = zip(*rows)
      batch_y_score = torch.sigmoid(model(torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(model.device))).detach().cpu().numpy()
      for i, (term, x_true, y_true) in enumerate(zip(batch_term, batch_x_true, batch_y_true)):
        y_score = batch_y_score[i, :]
        # ignore genes not in the background
        y_true = y_true[background & ~(x_true.bool().numpy())]
        y_score = y_score[background & ~(x_true.bool().numpy())]
        t = 0.5
        scores.append(dict(
          name=library,
          model_base=pathlib.Path(ckpt).parent.parent.parent.parent.name,
          model_name=f"{config['model']['class_path']}({pathlib.Path(ckpt).parent.parent.name})",
          term=term,
          threshold=t,
          roc_auc=roc_auc_score(y_true, y_score),
          ap=average_precision_score(y_true, y_score),
          recall=recall_score(y_true, y_score>=t),
          f1=f1_score(y_true, y_score>=t),
          es=ES(y_true, y_score),
          n=y_true.shape[0],
          n_true=y_true.sum(),
          n_pred=(y_score>=t).sum(),
          n_correct=(y_score[y_true.astype(bool)]>=t).sum(),
        ))
    pd.DataFrame(scores).to_csv(file, sep='\t', index=None)

def dummy_benchmark(strategy, library):
  @if_not_exists(f"data/classic-benchmark/2025-09-08/{library}/dummy/{strategy}/scores.tsv")
  def _(file):
    scores = []
    library_background = read_gmt_background(f'data/{library}.gmt')
    vocab = DummyVocab(library_background)
    for i, (term, x, x_true, y_true) in enumerate(enumerate_x_ys(library=library, vocab=vocab)):
      clf = DummyClassifier(strategy=strategy)
      clf.fit(x_true.numpy(), x_true.numpy())
      y_score = clf.predict_proba(y_true)[:, 1]
      # ignore genes not in the background
      y_true = y_true[~(x_true.bool().numpy())]
      y_score = y_score[~(x_true.bool().numpy())]
      t = 0.5
      scores.append(dict(
        name=library,
        data_name=None,
        model_name=f"DummyClassifier({strategy=})",
        term=term,
        threshold=t,
        roc_auc=roc_auc_score(y_true, y_score),
        ap=average_precision_score(y_true, y_score),
        recall=recall_score(y_true, y_score>=t),
        f1=f1_score(y_true, y_score>=t),
        es=ES(y_true, y_score),
        n=y_true.shape[0],
        n_true=y_true.sum(),
        n_pred=(y_score>=t).sum(),
        n_correct=(y_score[y_true.astype(bool)]>=t).sum(),
      ))
    pd.DataFrame(scores).to_csv(file, sep='\t', index=None)


# @functools.cache
def gene_gene_sim_factory(name):
  gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'r')
  gene_gene = gene_gene_similarities[name].copy()
  gene_gene_similarities.close()
  np.fill_diagonal(gene_gene.values, float('nan'))
  return gene_gene

@functools.cache
def get_sims():
  gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'r')
  gene_gene_similarities_keys = list(gene_gene_similarities.keys())
  gene_gene_similarities.close()
  return gene_gene_similarities_keys

def sim_benchmark(name, libraries):
  gene_gene = gene_gene_sim_factory(name)
  for library in libraries:
    @if_not_exists(f"data/classic-benchmark/2025-09-08/{library}/sim/{name}/scores.tsv")
    def _(file):
      scores = []
      vocab = DummyVocab(gene_gene.index.tolist())
      vocab_arr = np.array(vocab.vocab)
      library_background = read_gmt_background(f'data/{library}.gmt')
      background = multi_hot(torch.tensor(vocab(library_background)), num_classes=len(vocab)).numpy().astype(bool)
      background[0] = 0
      background[1] = 0
      for i, (term, x, x_true, y_true) in enumerate(enumerate_x_ys(library=library, vocab=vocab)):
        x_true_mask = x_true.bool().numpy()
        x_true_mask[0] = False
        x_true_genes = vocab_arr[x_true_mask]
        y_score = norm(
          gene_gene.loc[x_true_genes, :]
            # we get the mean correlation
            .mean(axis=0, skipna=True)
        ).loc[vocab_arr[background & ~(x_true_mask)]]
        # ignore genes not in the background
        y_true = y_true[background & (~x_true_mask)]
        # get middle threshold
        t = 0.5
        scores.append(dict(
          name=library,
          data_name=None,
          model_name=f"Similarity({name=})",
          term=term,
          threshold=t,
          roc_auc=roc_auc_score(y_true, y_score),
          ap=average_precision_score(y_true, y_score),
          recall=recall_score(y_true, y_score>=t),
          f1=f1_score(y_true, y_score>=t),
          es=ES(y_true, y_score),
          n=y_true.shape[0],
          n_true=y_true.sum(),
          n_pred=(y_score>=t).sum(),
          n_correct=(y_score[y_true.astype(bool)]>=t).sum(),
        ))
      pd.DataFrame(scores).to_csv(file, sep='\t', index=None)

#%%
if __name__ == '__main__':
  libraries = [
    'GO', 'GO_MF_2025', 'GO_BP_2025', 'GO_CC_2025',
    'GWAS_Catalog_2025', 'MGI_2024', 'KEGG_2021', 'ChEA_2022',
    'Wiki_Pathways_2024',
  ]
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    tasks = []
    for library in libraries:
      for ckpt in pathlib.Path('../2025-08-29/lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt'):
        tasks.append((mp_run, model_benchmark_from_ckpt, ckpt, library))

      for ckpt in pathlib.Path('../2025-09-02/lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt'):
        tasks.append((mp_run, model_benchmark_from_ckpt, ckpt, library))

      for ckpt in pathlib.Path('lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt'):
        tasks.append((mp_run, model_benchmark_from_ckpt, ckpt, library))

      for strategy in ['most_frequent', 'stratified', 'uniform']:
        tasks.append((mp_run, dummy_benchmark, strategy, library))
    #
    for sim in get_sims():
      tasks.append((mp_run, sim_benchmark, sim.lstrip('/'), libraries))
    #
    random.shuffle(tasks)
    for fut in tqdm(concurrent.futures.as_completed((pool.submit(*task) for task in tasks)), total=len(tasks)):
      try:
        fut.exception()
      except KeyboardInterrupt:
        raise
      except:
        import traceback
        traceback.print_exc()
        continue

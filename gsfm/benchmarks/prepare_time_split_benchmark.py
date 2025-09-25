#%%
import pandas as pd
from utils import read_gmt, cached_wget, if_not_exists, mp_run, norm
import numpy as np
import functools

def get_libraries():
  GO_BP_2025 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2025', 'data/GO_BP_2025.gmt'))}
  GO_BP_2023 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023', 'data/GO_BP_2023.gmt'))}
  GO_BP = {}
  for term in (GO_BP_2025.keys() & GO_BP_2023.keys()):
    new = list(set(GO_BP_2025[term]) - set(GO_BP_2023[term]))
    if not new: continue
    GO_BP[term] = dict(
      train=set(GO_BP_2023[term]),
      test=new,
    )

  GO_CC_2025 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Cellular_Component_2025', 'data/GO_CC_2025.gmt'))}
  GO_CC_2023 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Cellular_Component_2023', 'data/GO_CC_2023.gmt'))}
  GO_CC = {}
  for term in (GO_CC_2025.keys() & GO_CC_2023.keys()):
    new = list(set(GO_CC_2025[term]) - set(GO_CC_2023[term]))
    if not new: continue
    GO_CC[term] = dict(
      train=set(GO_CC_2023[term]),
      test=new,
    )

  GO_MF_2025 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Molecular_Function_2025', 'data/GO_MF_2025.gmt'))}
  GO_MF_2023 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Molecular_Function_2023', 'data/GO_MF_2023.gmt'))}
  GO_MF = {}
  for term in (GO_MF_2025.keys() & GO_MF_2023.keys()):
    new = list(set(GO_MF_2025[term]) - set(GO_MF_2023[term]))
    if not new: continue
    GO_MF[term] = dict(
      train=set(GO_MF_2023[term]),
      test=new,
    )

  Wiki_Pathways_2023 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=WikiPathway_2023_Human', 'data/Wiki_Pathways_2023.gmt'))}
  Wiki_Pathways_2024 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=WikiPathways_2024_Human', 'data/Wiki_Pathways_2024.gmt'))}
  Wiki_Pathways = {}
  for term in (Wiki_Pathways_2024.keys() & Wiki_Pathways_2023.keys()):
    new = list(set(Wiki_Pathways_2024[term]) - set(Wiki_Pathways_2023[term]))
    if not new: continue
    Wiki_Pathways[term] = dict(
      train=set(Wiki_Pathways_2023[term]),
      test=new,
    )

  ChEA_2016 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ChEA_2016', 'data/ChEA_2016.gmt'))}
  ChEA_2022 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ChEA_2022', 'data/ChEA_2022.gmt'))}
  ChEA = {}
  for term in (ChEA_2022.keys() & ChEA_2016.keys()):
    new = list(set(ChEA_2022[term]) - set(ChEA_2016[term]))
    old = list(set(ChEA_2016[term]) - set(ChEA_2022[term]))
    if not new: continue
    train = set(ChEA_2016[term])
    test = set(ChEA_2022[term])
    ChEA[term] = dict(
      train=set(ChEA_2016[term]),
      test=new,
    )

  GWAS_Catalog_2019 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GWAS_Catalog_2019', 'data/GWAS_Catalog_2019.gmt'))}
  GWAS_Catalog_2023 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GWAS_Catalog_2023', 'data/GWAS_Catalog_2023.gmt'))}
  GWAS_Catalog = {}
  for term in (GWAS_Catalog_2023.keys() & GWAS_Catalog_2019.keys()):
    new = list(set(GWAS_Catalog_2023[term]) - set(GWAS_Catalog_2019[term]))
    if not new: continue
    GWAS_Catalog[term] = dict(
      train=set(GWAS_Catalog_2019[term]),
      test=new,
    )

  KEGG_2016 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2016', 'data/KEGG_2016.gmt'))}
  KEGG_2021 = {term: gene_set for term, desc, gene_set in read_gmt(cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human', 'data/KEGG_2021.gmt'))}
  KEGG_2016_keys = {term.partition('Homo sapiens')[0].strip(): term for term in KEGG_2016}
  KEGG = {}
  for term in (KEGG_2021.keys() & KEGG_2016_keys.keys()):
    new = list(set(KEGG_2021[term]) - set(KEGG_2016[KEGG_2016_keys[term]]))
    if not new: continue
    KEGG[term] = dict(
      train=set(KEGG_2016[KEGG_2016_keys[term]]),
      test=new,
    )

  libraries = [
    dict(library=GO_BP, name='GO_BP'),
    dict(library=GO_CC, name='GO_CC'),
    dict(library=GO_MF, name='GO_MF'),
    dict(library=Wiki_Pathways, name='Wiki_Pathways'),
    dict(library=GWAS_Catalog, name='GWAS_Catalog'),
    dict(library=KEGG, name='KEGG'),
  ]
  return libraries

#%%
import pathlib
import torch
from tqdm.auto import tqdm
from utils import read_gmt, multi_hot, chunked, ES, model_config_from_ckpt, model_tokenizer_from_ckpt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, average_precision_score, f1_score
from sklearn.dummy import DummyClassifier

#%%
batch_size = 16

class DummyVocab:
  def __init__(self, vocab, special=['<unk>', '<pad>']):
    self.vocab = [*special, *vocab]
    self.lookup = {t: i for i, t in enumerate(self.vocab)}
  def __len__(self):
    return len(self.vocab)
  def __call__(self, tokens):
    return [self.lookup.get(token, self.lookup['<unk>']) for token in tokens]

def enumerate_x_ys(*, library, vocab):
  for term, train_test in library.items():
    x_true = multi_hot(torch.tensor(vocab(train_test['train'])), num_classes=len(vocab))
    y_true = multi_hot(torch.tensor(vocab(train_test['test'])), num_classes=len(vocab)).numpy()
    if x_true.sum() < 5 or y_true.sum() < 5: continue
    yield term, torch.tensor(vocab(train_test['train'])), x_true, y_true

def model_benchmark_from_ckpt(ckpt, library):
  config = model_config_from_ckpt(ckpt)
  @if_not_exists(f"data/benchmark/2025-08-26/{library['name']}/model/{pathlib.Path(ckpt).parent.parent.name}/scores.tsv")
  def _(file):
    model, vocab = model_tokenizer_from_ckpt(ckpt, config=config, map_location=torch.device('cuda'))
    scores = []
    background = multi_hot(torch.tensor(vocab(library['background'])), num_classes=len(vocab)).numpy().astype(bool)
    for rows in tqdm(chunked(enumerate_x_ys(library=library['library'], vocab=vocab), batch_size)):
      batch_term, batch_x, batch_x_true, batch_y_true = zip(*rows)
      batch_y_score = torch.sigmoid(model(torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(model.device))).detach().cpu().numpy()
      for i, (term, x_true, y_true) in tqdm(enumerate(zip(batch_term, batch_x_true, batch_y_true))):
        y_score = batch_y_score[i, :]
        # ignore genes not in the background
        y_true = y_true[background & ~(x_true.bool().numpy())]
        y_score = y_score[background & ~(x_true.bool().numpy())]
        t = 0.5
        scores.append(dict(
          name=library['name'],
          model_name=f"{config['model']['class_path']}({pathlib.Path(ckpt).parent.parent.name})",
          data_name=config['data']['class_path'],
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
          **{f"model_{k}":v for k,v in config['model']['init_args'].items()}
        ))
    pd.DataFrame(scores).to_csv(file, sep='\t', index=None)

def dummy_benchmark(strategy, library):
  @if_not_exists(f"data/benchmark/2025-08-26/{library['name']}/dummy/{strategy}/scores.tsv")
  def _(file):
    scores = []
    for i, (term, x, x_true, y_true) in tqdm(enumerate(enumerate_x_ys(library=library['library'], vocab=DummyVocab(library['background'])))):
      clf = DummyClassifier(strategy=strategy)
      clf.fit(x_true.numpy(), x_true.numpy())
      y_score = clf.predict_proba(y_true)[:, 1]
      # ignore genes not in the background
      y_true = y_true[~(x_true.bool().numpy())]
      y_score = y_score[~(x_true.bool().numpy())]
      t = 0.5
      scores.append(dict(
        name=library['name'],
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


@functools.cache
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

def sim_benchmark(name, library):
  @if_not_exists(f"data/benchmark/2025-08-26/{library['name']}/sim/{name}/scores.tsv")
  def _(file):
    scores = []
    gene_gene = gene_gene_sim_factory(name)
    vocab = DummyVocab(gene_gene.index.tolist())
    vocab_arr = np.array(vocab.vocab)
    background = multi_hot(torch.tensor(vocab(library['background'])), num_classes=len(vocab)).numpy().astype(bool)
    background[0] = 0
    background[1] = 0
    for i, (term, x, x_true, y_true) in tqdm(enumerate(enumerate_x_ys(library=library['library'], vocab=vocab))):
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
      t = 0.5
      scores.append(dict(
        name=library['name'],
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
  libraries = get_libraries()
  for library in tqdm(libraries):
    library['background'] = list({gene for _term, train_test in library['library'].items() for split, genes in train_test.items() for gene in genes })

    for ckpt in tqdm(pathlib.Path('../2025-05-27/lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt')):
      mp_run(model_benchmark_from_ckpt, ckpt, library)

    for ckpt in tqdm(pathlib.Path('lightning_logs').glob('*/checkpoints/epoch=49-*.ckpt')):
      mp_run(model_benchmark_from_ckpt, ckpt, library)

    # for strategy in ['most_frequent', 'stratified', 'uniform']:
    #   mp_run(dummy_benchmark, strategy, library)
    
    # for sim in get_sims():
    #   mp_run(sim_benchmark, sim.lstrip('/'), library)

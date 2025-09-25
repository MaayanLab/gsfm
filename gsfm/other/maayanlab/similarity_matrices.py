# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
from supervenn import supervenn
from gsfm import utils

#%%
gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'a')

# %%
if 'archs4' not in gene_gene_similarities:
  gene_gene_similarities['archs4'] = pd.read_pickle(utils.cached_wget('https://s3.amazonaws.com/mssm-data/human_correlation_v2.4.pkl', 'data/inputs/archs4-2-4.pkl'))

#%%
def read_geneshot(f):
  import h5py
  f = h5py.File(f, 'r')
  columns = list(map(bytes.decode, f['meta/colid']))
  return pd.DataFrame(f['data/matrix'][:], columns=columns, index=columns)

#%%
if 'geneshot_coexpression' not in gene_gene_similarities:
  gene_gene_similarities['geneshot_coexpression'] = read_geneshot(utils.cached_wget('https://mssm-geneshot.s3.amazonaws.com/coexpression.h5', 'data/inputs/geneshot_coexpression.h5'))

#%%
if 'geneshot_generif' not in gene_gene_similarities:
  gene_gene_similarities['geneshot_generif'] = read_geneshot(utils.cached_wget('https://mssm-geneshot.s3.amazonaws.com/generif.h5', 'data/inputs/geneshot_generif.h5'))

#%%
if 'geneshot_tagger' not in gene_gene_similarities:
  gene_gene_similarities['geneshot_tagger'] = read_geneshot(utils.cached_wget('https://mssm-geneshot.s3.amazonaws.com/tagger.h5', 'data/inputs/geneshot_tagger.h5'))

#%%
if 'geneshot_enrichr' not in gene_gene_similarities:
  gene_gene_similarities['geneshot_enrichr'] = read_geneshot(utils.cached_wget('https://mssm-geneshot.s3.amazonaws.com/enrichr.h5', 'data/inputs/geneshot_enrichr.h5'))

# %%
if 'gtex' not in gene_gene_similarities:
  gene_gene_similarities['gtex'] = pd.read_csv(utils.cached_wget('https://maayanlab.cloud/static/hdfs/harmonizome/data/gtextissue23/gene_similarity_matrix_cosine.txt.gz', 'data/inputs/gtex.txt.gz'), sep='\t', index_col=0)

# %%
if 'hpa' not in gene_gene_similarities:
  hpa = pd.read_csv(utils.cached_wget('https://maayanlab.cloud/static/hdfs/harmonizome/data/hpatissuesprotein/gene_similarity_matrix_cosine.txt.gz', 'data/inputs/hpa.txt.gz'), sep='\t', index_col=[0, 1, 2], header=[0, 1, 2])
  gene_gene_similarities['hpa'] = pd.DataFrame(
    hpa.values,
    index=hpa.index.get_level_values(0),
    columns=hpa.columns.get_level_values(0),
  )

#%%
if 'hpamrna' not in gene_gene_similarities:
  hpamrna = pd.read_csv(utils.cached_wget('https://maayanlab.cloud/static/hdfs/harmonizome/data/hpatissuesmrna/gene_similarity_matrix_cosine.txt.gz', 'data/inputs/hpamrna.txt.gz'), sep='\t', index_col=[0, 1, 2], header=[0, 1, 2])
  gene_gene_similarities['hpamrna'] = pd.DataFrame(
    hpamrna.values,
    index=hpamrna.index.get_level_values(0),
    columns=hpamrna.columns.get_level_values(0),
  )

# %%
if 'enrichr' not in gene_gene_similarities:
  enrichr = pd.read_csv(utils.cached_wget('https://s3.amazonaws.com/mssm-data/list_off_co.tsv', 'data/inputs/enrichr.tsv'), sep='\t', index_col=0)
  gene_gene_similarities['enrichr'] = enrichr / np.diag(enrichr.values)

#%%
if 'enrichr_2024' not in gene_gene_similarities:
  enrichr_2024 = pd.HDFStore('data/enrichr-user-lists-fixed-co-occurrence.h5', 'r')['data']
  enrichr_2024 /= np.diag(enrichr_2024.values)
  gene_gene_similarities['enrichr_2024'] = enrichr_2024

# %%
@utils.if_not_exists('data/inputs/generif_gene_gene.tsv.gz')
def _(filename):
  generif = pd.read_csv(utils.cached_wget('https://s3.amazonaws.com/mssm-data/generif.tsv', 'data/inputs/generif.tsv'), sep='\t', header=None)

  # convert to sparse array
  generif_genes = pd.Series(generif[0].unique()).reset_index().rename({'index': 'gene_index'}, axis=1)
  generif_papers = pd.Series(generif[1].unique()).reset_index().rename({'index': 'paper_index'}, axis=1)
  coo = pd.merge(
    left=pd.merge(left=generif, left_on=0, right=generif_genes, right_on=0),
    left_on=1,
    right=generif_papers,
    right_on=0,
  )[['gene_index', 'paper_index']]
  coo['value'] = 1
  generif_sparse = sp.coo_array(
    (coo['value'].values, coo[['gene_index', 'paper_index']].values.T),
    shape=(generif_genes.shape[0], generif_papers.shape[0])
  )

  # get cosine similarity
  generif_gene_gene = pd.DataFrame(
    sklearn.metrics.pairwise.cosine_similarity(generif_sparse),
    index=generif_genes[0],
    columns=generif_genes[0],
  )
  generif_gene_gene.to_csv(filename, sep='\t', compression='gzip')

if 'generif' not in gene_gene_similarities:
  gene_gene_similarities['generif'] = pd.read_csv('data/inputs/generif_gene_gene.tsv.gz', sep='\t', compression='gzip', index_col=0)

#%%
if 'rummageo' not in gene_gene_similarities:
  rummageo = pd.read_feather(utils.cached_wget('https://s3.amazonaws.com/maayanlab-public/rummageo/random50k_co-occurrence_coding.f', 'data/inputs/rummageo-co-occurrence.f'))
  rummageo.index = rummageo.columns
  rummageo /= np.diag(rummageo.values)
  gene_gene_similarities['rummageo'] = rummageo

#%%
if 'rummagene' not in gene_gene_similarities:
  import gzip
  with gzip.open(utils.cached_wget('https://s3.amazonaws.com/maayanlab-public/rummagene/random50k_co-occurrence.f.gz', 'data/inputs/rummagene-co-occurrence.f.gz')) as fr:
    rummagene = pd.read_feather(fr)
  rummagene.index = rummagene.columns
  rummagene /= np.diag(rummagene.values)
  mask = ~pd.isna(rummagene).any()
  rummagene = rummagene.loc[mask, mask]
  gene_gene_similarities['rummagene'] = rummagene

#%%
sets, set_annotations = zip(*[
  (set(gene_gene_similarities[k].index), k)
  for k in gene_gene_similarities
])
common_genes = list(set.intersection(*sets))
all_genes = list(set.union(*sets))

#%%
@utils.if_not_exists('data/rummage-50k-50k-co-occurrence.f.gz')
def _(filename):
  import gzip
  import numpy as np
  rummagene = [
    clean_genes
    for _, _, genes in utils.read_gmt('data/train.gmt')
    for clean_genes in (list(set(filter(None, genes))),)
    if len(clean_genes) > 5 and len(clean_genes) < 1000
  ]
  rummagene_i = np.random.choice(len(rummagene), 50000, replace=False)
  rummagene = [rummagene[i] for i in rummagene_i]
  rummageo = [
    clean_genes
    for _, _, genes in utils.read_gmt('data/human-geo-auto.gmt')
    for clean_genes in (list(set(filter(None, genes))),)
    if len(clean_genes) > 5 and len(clean_genes) < 1000
  ]
  rummageo_i = np.random.choice(len(rummageo), 50000, replace=False)
  rummageo = [rummageo[i] for i in rummageo_i]
  rummage = rummagene + rummageo
  rummage_co = utils.co_occurrence(lambda: rummage)
  with gzip.open(filename, 'wb') as fw:
    rummage_co.to_feather(fw)

#%%
if 'rummage' not in gene_gene_similarities:
  import gzip
  with gzip.open('data/rummage-50k-50k-co-occurrence.f.gz', 'rb') as fr:
    rummage = pd.read_feather(fr)
  rummage.index = rummage.columns
  rummage /= np.diag(rummage.values)
  rummage.drop('', inplace=True)
  rummage.drop('', inplace=True, axis=1)
  gene_gene_similarities['rummage'] = rummage

#%%
@utils.if_not_exists('figures/fig1.pdf')
def _(filename):
  plt.figure(figsize=(12, 6))
  supervenn(sets, set_annotations, min_width_for_annotation=1000)
  plt.xlabel('GENES')
  plt.tight_layout()
  plt.savefig(filename)
  plt.show()

#%%
gene_gene_similarities.close()

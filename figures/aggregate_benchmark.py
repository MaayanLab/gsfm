#%%
import yaml
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import functools

#%%
@functools.cache
def get_model_config(model_base, model_version):
  model_path = pathlib.Path(f"~/Programs/work/experiments/genesetformer/{model_base}/lightning_logs/{model_version}/config.yaml").expanduser()
  if not model_path.exists():
    return None
  with model_path.open('r') as fr:
    model_config = yaml.safe_load(fr)
  return model_config

#%%
df = pd.concat([
  pd.read_csv(f, sep='\t')
  for f in pathlib.Path('data/classic-benchmark/2025-09-08').rglob('scores.tsv')
])

def rename_model(row):
  import re
  if m := re.match(r"^Similarity\(name='(.+?)'\)$", row['model_name']):
    name = m.group(1)
    if name == 'node2vec_a':
      name = 'Node2Vec Context (Rummagene > mean + 3 std)'
    elif name == 'node2vec_b':
      name = 'Node2Vec Node (Rummagene > mean + 3 std)'
    elif name == 'node2vec_c':
      name = 'Node2Vec (Rummagene > 0.5)'
    elif name == 'deepNF':
      name = 'deepNF (Rummagene > mean + 3 std)'
    elif name == 'deepNF_c':
      name = 'deepNF (Rummagene > 0.5)'
    elif name == "geneptsim":
      name = 'GenePT'
    elif name == 'gsfm_emb':
      name = 'GSFM middle layer'
    elif name == 'gsfm_enc':
      name = 'GSFM last layer'
    elif name == 'rummagene':
      name = 'Rummagene'
    elif name == 'rummageo':
      name = 'RummaGEO'
    elif name == 'rummage':
      name = 'RummaGEO/Gene'
    elif name == 'gtex':
      name = 'GTEx'
    elif name == 'enrichr_2024':
      name = 'Enrichr User List Co-Occurrence 2024'
    elif name == 'enrichr':
      name = 'Enrichr User List Co-Occurrence 2018'
    elif name == 'generif':
      name = 'GeneRIF 2019'
    elif name == 'geneshot_generif':
      name = 'GeneRIF 2021'
    elif name == 'geneformer':
      name = 'Geneformer'
    elif name == 'geneshot_coexpression':
      name = 'ARCHS4 Co-Expression 2021'
    elif name == 'geneshot_tagger':
      name = 'Tagger 2021'
    elif name == 'scgpt':
      name = 'scGPT'
    elif name == 'archs4':
      name = 'ARCHS4 Co-Expression 2024'
    elif name == 'hpa':
      # name = 'HPA'
      return None
    elif name == 'hpamrna':
      name = 'HPA'
    elif name == 'geneshot_enrichr':
      name = 'Enrichr User List Co-Occurrence 2021'
    return f"{name} Gene-Gene Similarity"
  elif m := re.match(r"^DummyClassifier\(strategy='(.+?)'\)$", row['model_name']):
    name = m.group(1)
    if name == 'most_frequent':
      return 'Most Frequent'
    elif name == 'stratified':
      return 'Stratified'
    elif name == 'uniform':
      return 'Uniform'
  elif m := re.match(r"^((\w+?)\.GeneSetMaskedEmbeddingAutoEncoder)\((.+?)\)$", row['model_name']):
    model_base = '2025-09-08' if pd.isna(row['model_base']) else row['model_base']
    _model_type = m.group(2)
    model_version = m.group(3)
    model_config = get_model_config(model_base, model_version)
    assert model_config is not None
    if model_config['data']['class_path'] == 'rummagene_setnoise.RummageneDataModule':
      return f"GSFM Rummagene noise={model_config['data']['init_args']['noise_size']}"
    if model_config['data']['class_path'] == 'rummagene_no_mito.RummageneDataModule':
      return f"GSFM Rummagene no-mito"
    if model_config['data']['class_path'] == 'rummagene_setsize.RummageneDataModule':
      return f"GSFM Rummagene max_setsize={model_config['data']['init_args']['max_size']}"
    if model_config['data']['class_path'] == 'rummagene_datasize.RummageneDataModule':
      return f"GSFM Rummagene subset={model_config['data']['init_args']['data_size']}"
    if model_config['data']['class_path'] == 'rummagene_seq2multihot.RummageneDataModule':
      if model_config['data']['init_args'].get('before_year'):
        return f"GSFM Rummagene year<{model_config['data']['init_args']['before_year']}"
      else:
        return f"GSFM Rummagene"
    if model_config['data']['class_path'] == 'rummage_human_seq2multihot.RummageDataModule':
      return f"GSFM Human RummaGEO/Gene"
    if model_config['data']['class_path'] == 'rummage_seq2multihot.RummageDataModule':
      return f"GSFM RummaGEO/Gene"
    if model_config['data']['class_path'] == 'rummageo_human_seq2multihot.RummaGEODataModule':
      return f"GSFM Human RummaGEO"
    if model_config['data']['class_path'] == 'rummageo_seq2multihot.RummaGEODataModule':
      if model_config['data']['init_args'].get('before_year'):
        return f"GSFM RummaGEO year<{model_config['data']['init_args']['before_year']}"
      else:
        return f"GSFM RummaGEO"
    raise NotImplementedError(model_config['data']['class_path'])
  raise NotImplementedError(row['model_name'])

df['model_name'] = df.apply(rename_model, axis=1)
df.dropna(subset=['model_name'], inplace=True)

#%%
df_ = df.groupby(['model_name', 'name', 'term'])['roc_auc'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['roc_auc'].median().sort_values(ascending=False).index
fig, ax= plt.subplots(1, 1, figsize=(6, 12))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['roc_auc'].median().unstack()[sorted_model_name].T,
  fmt="0.2f",
  annot=True,
  vmin=0.5,
  vmax=1.0,
)
plt.title('Median AUROC')
plt.savefig('all-results-auroc.pdf', bbox_inches='tight')

#%%
df_ = df.groupby(['model_name', 'name', 'term'])['es'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['es'].median().sort_values(ascending=False).index
fig, ax= plt.subplots(1, 1, figsize=(6, 12))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['es'].median().unstack()[sorted_model_name].T,
  fmt="0.2f",
  annot=True,
  vmin=0.5,
  vmax=1.0,
)
plt.title('Median Enrichment Score')
plt.savefig('all-results-es.pdf', bbox_inches='tight')

#%%
df_ = df.groupby(['model_name', 'name', 'term'])['ap'].mean().reset_index()
sorted_model_name = df_.groupby(['model_name'])['ap'].mean().sort_values(ascending=False).index
fig, ax= plt.subplots(1, 1, figsize=(6, 12))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['ap'].mean().unstack()[sorted_model_name].T,
  fmt="0.2f",
  annot=True,
  # vmin=0.5,
  # vmax=1.0,
)
plt.title('Average Precision')
plt.savefig('all-results-ap.pdf', bbox_inches='tight')

# %%

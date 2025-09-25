#%%
import yaml
import pathlib
import functools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df = pd.concat([
  pd.read_csv(f, sep='\t')
  for f in pathlib.Path('data/benchmark/2025-08-29').rglob('scores.tsv')
])

@functools.cache
def get_model_config(model_base, model_version):
  model_path = pathlib.Path(f"~/Programs/work/experiments/genesetformer/{model_base}/lightning_logs/{model_version}/config.yaml").expanduser()
  if not model_path.exists():
    return None
  with model_path.open('r') as fr:
    model_config = yaml.safe_load(fr)
  return model_config

def rename_model(row):
  import re
  if m := re.match(r"^((\w+?)\.GeneSetMaskedEmbeddingAutoEncoder)\((.+?)\)$", row['model_name']):
    model_base = '2025-08-29'
    _model_type = m.group(2)
    model_version = m.group(3)
    model_config = get_model_config(model_base, model_version)
    if not model_config: return None
    if model_config['data']['class_path'] == 'rummagene_seq2multihot.RummageneDataModule':
      if model_config['data']['init_args'].get('before_year'):
        return f"GSFM Rummagene year<{model_config['data']['init_args']['before_year']}"
      else:
        return f"GSFM Rummagene year<=2025"
    if model_config['data']['class_path'] == 'rummage_seq2multihot.RummageDataModule':
      return f"GSFM RummaGEO/Gene"
    if model_config['data']['class_path'] == 'rummageo_seq2multihot.RummaGEODataModule':
      return None
      if model_config['data']['init_args'].get('before_year'):
        return f"GSFM RummaGEO year<{model_config['data']['init_args']['before_year']}"
      else:
        return f"GSFM RummaGEO"
    raise NotImplementedError(model_config['data']['class_path'])
  return None

df['model_name'] = df.apply(rename_model, axis=1)
df.dropna(subset=['model_name'], inplace=True)

#%%
df_ = df[df['name'].str.startswith('GO_20')].groupby(['model_name', 'name', 'term'])['roc_auc'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['roc_auc'].median().sort_values(ascending=True).index
fig, ax= plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['roc_auc'].median().unstack()[sorted_model_name],
  fmt="0.2g",
  annot=True,
  vmin=0.5,
  vmax=1.0,
)
plt.title('Median AUROC Recovering Added Genes')
plt.savefig('revision-fig-s10c.pdf', bbox_inches='tight')

#%%
df_ = df[~df['name'].str.startswith('GO_20')].groupby(['model_name', 'name', 'term'])['roc_auc'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['roc_auc'].median().sort_values(ascending=True).index
fig, ax= plt.subplots(1, 1, figsize=(12, 8))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['roc_auc'].median().unstack()[sorted_model_name],
  fmt="0.2g",
  annot=True,
  vmin=0.5,
  vmax=1.0,
)
plt.title('Median AUROC Recovering Added Genes')
plt.savefig('revision-fig-s10a.pdf', bbox_inches='tight')

#%%
import re
def cmp(row):
  benchmark = int(re.match(r'^.+?(\d+)_to_\d+$', row['name']).group(1))
  training_data = int(re.match(r'^.+?(\d+)$', row['model_name']).group(1))
  return benchmark >= training_data
df_ = df[df['name'].str.startswith('GO_20')]
df_ = df_[df_.apply(cmp,axis=1)]
df_ = df_.groupby(['model_name', 'name', 'term'])['roc_auc'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['roc_auc'].median().index
fig, ax= plt.subplots(1, 1, figsize=(12, 8))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['roc_auc'].median().unstack()[sorted_model_name],
  fmt="0.2g",
  annot=True,
  vmin=0.5,
  vmax=1.0,
)
plt.title('Median AUROC Recovering Added Genes')
plt.savefig('revision-fig-s10b.pdf', bbox_inches='tight')

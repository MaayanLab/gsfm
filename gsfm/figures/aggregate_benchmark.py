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
  model_path = pathlib.Path(f"genesetformer/{model_base}/lightning_logs/{model_version}/config.yaml").expanduser()
  if not model_path.exists():
    return None
  with model_path.open('r') as fr:
    model_config = yaml.safe_load(fr)
  return model_config

#%%
df = pd.concat([
  pd.read_csv(f, sep='\t')
  for f in pathlib.Path('genesetformer/data/classic-benchmark/2025-09-08').rglob('scores.tsv')
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
plt.savefig('data/all-results-auroc.pdf', bbox_inches='tight')

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
plt.savefig('data/all-results-es.pdf', bbox_inches='tight')

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
plt.savefig('data/all-results-ap.pdf', bbox_inches='tight')

# %%
import scipy
import starbars
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['savefig.transparent'] = True

def make_plot_for(df, names):
  df_ = df[df['name'].isin(list(names))]
  pvals = []
  fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(4, 2, figsize=(18,36), dpi=600, constrained_layout=True)
  for (ax_label, ax), (library, d) in zip([('A', ax11), ('B', ax12), ('C', ax21), ('D', ax22), ('C', ax31), ('D', ax32), ('E', ax41), ('F', ax42)], df_.groupby('library')):
    d['name'] = d['name'].replace(names)
    y_order = list(d.groupby('name')['roc_auc'].median().sort_values(ascending=False).index)
    annotations = []

    for left, right in zip(y_order, y_order[1:]):
      x, y = d[d['name']==left], d[d['name']==right]
      left_roc_auc_mean = x.groupby(['term'])['roc_auc'].median().mean()
      left_roc_auc_std = x.groupby(['term'])['roc_auc'].median().std()
      right_roc_auc_mean = y.groupby(['term'])['roc_auc'].median().mean()
      right_roc_auc_std = y.groupby(['term'])['roc_auc'].median().std()
      x, y = x.groupby(['term'])['roc_auc'].median().align(y.groupby(['term'])['roc_auc'].median(), join='inner')
      # _, p_value = scipy.stats.ttest_rel(x, y)
      _, p_value = scipy.stats.ttest_ind(x, y, equal_var=False)
      if p_value <= 0.01:
        annotations.append((left, right, p_value))
        pvals.append(dict(
          library=library,
          left=left,
          left_roc_auc_mean=left_roc_auc_mean,
          left_roc_auc_std=left_roc_auc_std,
          right_roc_auc_mean=right_roc_auc_mean,
          right_roc_auc_std=right_roc_auc_std,
          right=right,
          p_value=p_value,
        ))
    sns.boxenplot(d, x='roc_auc', y='name', order=y_order, ax=ax, native_scale=True)
    ax.set_title(library.replace('_', ' '), fontsize=18)
    ax.set_xlabel('Median AUROC', fontsize=16)
    ax.set_ylabel('')
    ax.set_xlim((0.0, 1.0))
    starbars.draw_annotation(annotations, ax=ax, mode='horizontal', h_gap=0, bar_gap=0.1, fontsize=10)
    ax.set_xlim((0.0, 1.2))
    for ticklabel, tick in zip(ax.xaxis.get_ticklabels(), ax.xaxis.get_major_ticks()):
      text = ticklabel.get_text()
      try:
        if text and float(text) > 1:
          tick.set_visible(False)
      except ValueError: pass
    leftmost, rightmost = ax.get_xlim()
    ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
    bottommost, topmost = ax.get_ylim()
    ax.axhline(bottommost, 0, 0.875, color='0')
    ax.axhline(topmost, 0, 0.875, color='0')
    ax.axvline(leftmost, color='0')
    ax.axvline(1.05, color='0')
    ax.text(-0.1, 1.1, ax_label, transform=ax.transAxes, size=20, weight='bold')
  fig.tight_layout()
  return fig, pvals


#%%
print('\n'.join(df['model_name'].unique()))

#%%
print('\n'.join(df['name'].unique()))

#%%
names = [
  'KEGG_2021',
  'Wiki_Pathways_2024',
  'ChEA_2022',
  'GO_BP_2025',
  'GO_MF_2025',
  'MGI_2024',
  'GO_CC_2025',
  'GWAS_Catalog_2025',
]
model_names = {
'Uniform':'Random',
'GSFM Rummagene year<2015':'GSFM Rummagene year<2015',
'GSFM Rummagene noise=4.0':'GSFM Rummagene noise=4.0',
'GSFM Rummagene max_setsize=50':'GSFM Rummagene max_setsize=50',
'GSFM Rummagene':'GSFM Rummagene',
'GSFM Rummagene no-mito':'GSFM Rummagene no-mito',
'GSFM RummaGEO/Gene':'GSFM RummaGEO/Gene',
'GSFM RummaGEO':'GSFM RummaGEO',
'GenePT Gene-Gene Similarity':'GenePT Sim',
'Rummagene Gene-Gene Similarity':'Rummagene Sim',
'ARCHS4 Co-Expression 2021 Gene-Gene Similarity':'ARCHS4 Co-Expression 2021',
'Geneformer Gene-Gene Similarity':'Geneformer Sim',
'Tagger 2021 Gene-Gene Similarity':'Tagger 2021 Sim',
'Node2Vec (Rummagene > 0.5) Gene-Gene Similarity':'Node2Vec Sim',
'RummaGEO/Gene Gene-Gene Similarity':'RummaGEO/Gene Sim',
'RummaGEO Gene-Gene Similarity':'RummaGEO Sim',
'scGPT Gene-Gene Similarity':'scGPT Sim',
'deepNF (Rummagene > 0.5) Gene-Gene Similarity':'deepNF Sim',
'GeneRIF 2021 Gene-Gene Similarity':'GeneRIF 2021 Sim',
'Enrichr User List Co-Occurrence 2024 Gene-Gene Similarity':'Enrichr User List Co-Occurrence 2024 Sim',
}
d = df[df['name'].isin(names) & df['model_name'].isin(model_names.keys())].copy()
d['model_name'] = d['model_name'].replace(model_names)

#%%
fig, pvals = make_plot_for(df[df['name'].isin(names)].rename({
  'name': 'library',
  'model_name': 'name',
}, axis=1), model_names)

#%%
pd.DataFrame(pvals).sort_values('left_roc_auc_mean', ascending=False)

#%%
fig.savefig('data/main-benchmark.pdf')

#%%
pd.DataFrame(pvals).to_csv('data/main-benchmark-pvals.tsv', sep='\t')


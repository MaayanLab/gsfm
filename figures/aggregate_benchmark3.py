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

#%%
df_ = df[df['name'].str.startswith('GO_20')]
df_ = df_[df_.apply(cmp,axis=1)]
df_ = df_.groupby(['model_name', 'name', 'term'])['es'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['es'].median().index
fig, ax= plt.subplots(1, 1, figsize=(12, 8))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['es'].median().unstack()[sorted_model_name],
  fmt="0.2g",
  annot=True,
  vmin=0.0,
  vmax=1.0,
)

#%%
df_ = df.groupby(['model_name', 'name', 'term'])['es'].median().reset_index()
sorted_model_name = df_.groupby(['model_name'])['es'].median().sort_values(ascending=False).index
fig, ax= plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
  df_.groupby(['name', 'model_name'])['es'].median().unstack()[sorted_model_name],
  fmt="0.2g",
  annot=True,
  vmin=0.0,
  vmax=1.0,
)



#%%
import yaml
import importlib
import pathlib
import pathlib
import seaborn as sns
import pandas as pd
import starbars
import scipy.stats
import matplotlib, matplotlib.pyplot as plt

def from_directory(dir):
  def decorator(fn):
    import os, sys, functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      orig_dir = pathlib.Path.cwd()
      os.chdir(dir)
      sys.path.insert(0, dir)
      orig_modules = {*sys.modules.keys()}
      ret = fn(*args, **kwargs)
      for key in ({*sys.modules.keys()} - orig_modules):
        if dir in (getattr(sys.modules[key], '__file__', None) or ''):
          del sys.modules[key]
      sys.path.remove(dir)
      os.chdir(orig_dir)
      return ret
    return wrapper
  return decorator

def coalesce_glob(*globs):
  for glob in globs:
    if any(True for _ in pathlib.Path('/').glob(glob[1:])):
      return glob

models = {}
params = {}

for d in ['2025-05-07', '2025-05-09', '2025-05-27']:
  for f in pathlib.Path('~/Programs/work/experiments/genesetformer/').expanduser().glob(f'{d}/lightning_logs/*/config.yaml'):
    config = yaml.safe_load(f.read_bytes())
    ctx = {}
    *model_mods, ctx['model_cls_name'] = config['model']['class_path'].split('.')
    ctx['model_cls_mod'] = '.'.join(model_mods)
    ctx['model_init_args'] = config['model'].get('init_args', {})
    *data_mods, ctx['data_cls_name'] = config['data']['class_path'].split('.')
    ctx['data_cls_mod'] = '.'.join(data_mods)
    ctx['data_init_args'] = config['data'].get('init_args', {})
    for ckpt in f.parent.glob('checkpoints/*.ckpt'):
      epochs = int(ckpt.name.partition('=')[-1].partition('-')[0])+1
      @from_directory(str(f.parent.parent.parent))
      def model_factory(ctx=ctx, ckpt=ckpt):
        model_cls = getattr(importlib.import_module(ctx['model_cls_mod']), ctx['model_cls_name'])
        model = model_cls.load_from_checkpoint(ckpt, map_location='cpu', hparams=str(f.parent/'hparams.yaml'))
        model.eval()
        data_cls = getattr(importlib.import_module(ctx['data_cls_mod']), ctx['data_cls_name'])
        data = data_cls(**ctx['data_init_args'])
        data.setup('predict')
        return model, data
      model_name = f"{f.parent.parent.parent.name}_{f.parent.name}_{ctx['model_cls_mod']}_{ctx['model_cls_name']}_{ckpt.name}"
      models[model_name] = model_factory
      params[model_name] = dict(
        yaml.safe_load((f.parent/'hparams.yaml').read_bytes()),
        model_cls=f"{ctx['model_cls_mod']}.{ctx['model_cls_name']}",
        data_cls=ctx['data_cls_name'],
        benchmarks=f"/home/u8sand/Programs/work/experiments/genesetformer/{d}/data/{d}/gene-function-prediction--*--{f.parent.name}e{epochs}.tsv",
        epochs=epochs,
      )


#%%
def incl_name(df, name):
  df['name'] = name
  return df

df = pd.concat([
  pd.read_csv(f, sep='\t')
  for f in [
    *pathlib.Path('data').glob('2024-11-04/*random.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*archs4.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*rummagene.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*rummageo.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*geneptsim.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*geneshot_generif.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*enrichr.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*gtex.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*geneshot_tagger.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*random.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*hpa*.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*rummage.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*gsfm_*.tsv'),
    *pathlib.Path('data').glob('2025-05-06/*prismexp*.tsv'),
  ]
] + [
  incl_name(pd.read_csv(f, sep='\t'), model_name)
  for model_name, f in [
    (model_name, f)
    for model_name, P in params.items()
    if P.get('benchmarks')
    for f in pathlib.Path('/').glob(P['benchmarks'][1:])
  ]
], axis=0)
df.loc[pd.isna(df['split']), 'split'] = 0.5

#%%
df_params = pd.DataFrame(params).T.drop(['_instantiator', 'name', 'benchmarks', 'file', 'vocab_size'], axis=1)
df_ = pd.merge(left=df, left_on='name', right=df_params, right_index=True, how='left')
df_
# %%
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['savefig.transparent'] = True

def make_plot_for(df, names):
  df_ = df[df['name'].isin(list(names))]
  fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(18,12), dpi=600, constrained_layout=True)
  for (ax_label, ax), (library, d) in zip([('A', ax11), ('B', ax12), ('C', ax21), ('D', ax22)], df_.groupby('library')):
    d['name'] = d['name'].replace(names)
    y_order = list(d.groupby('name')['roc_auc'].median().sort_values(ascending=False).index)
    annotations = []

    for left, right in zip(y_order, y_order[1:]):
      x, y = d[d['name']==left], d[d['name']==right]
      x, y = x.groupby(['term'])['roc_auc'].median().align(y.groupby(['term'])['roc_auc'].median(), join='inner')
      # _, p_value = scipy.stats.ttest_rel(x, y)
      _, p_value = scipy.stats.ttest_ind(x, y, equal_var=False)
      if p_value <= 0.01: annotations.append((left, right, p_value))
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
  return fig


#%%
fig=make_plot_for(df, {
  'rummage': 'RummaGEO/Gene',
  'rummagene': 'Rummagene',
  'enrichr': 'Enrichr',
  'archs4': 'ARCHS4',
  'rummageo': 'RummaGEO',
  'geneptsim': 'GenePT',
  'geneshot_tagger': 'Tagger',
  'geneshot_generif': 'GeneRIF',
  'random': 'Random',
  'gtex': 'GTEx',
  # 'hpa': 'HPA',
})
# fig.savefig('figs/1.svg', bbox_inches='tight')

#%%
df_[(df_['epochs']==50) & (df_['model_cls']=='J.GeneSetMaskedEmbeddingAutoEncoder') & (df_['depth']==2) & (df_['d_model']==256)].groupby([
  'data_cls',
  'model_cls',
  'd_model',
  'depth',
  'epochs'
])['name'].agg(set).reset_index().values


#%%
import matplotlib
matplotlib.rc('axes', titlesize=15, labelsize=15)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
#%%
fig=make_plot_for(df, {
  # '2025-05-07_version_55_F_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GSEPAE[rummagene, e=20, d=1024]',
  # '2025-05-07_version_62_F_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=29320.ckpt': 'GSEPAE[rummageo, e=20, d=1024]',
  # '2025-05-07_version_52_F_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=93360.ckpt': 'GSEPAE[rummageo/gene, e=20, d=1024]',
  # '2025-05-07_version_49_A_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GSmEAE[rummagene, e20, d=128]',
  # '2025-05-07_version_47_B_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GESAE[rummagene, e20, d=128]',
  # '2025-05-07_version_63_F_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GSEPAE[rummagene, e20, d=128]',
  # '2025-05-07_version_48_F_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GSEPAE[rummagene, e20, d=128]',
  # '2025-05-07_version_73_J2_GeneSetMaskedEmbeddingAutoEncoder_epoch=19-step=16020.ckpt': 'GSAE[rummagene, e20, d=128]',
  # '2025-05-08_version_6_K_GeneSetMaskedEmbeddingAutoEncoder_epoch=99-step=10100.ckpt': 'GSAE[rummagene, e=100, d=1024, depth=4]',
  # '2025-05-07_version_84_K_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=233400.ckpt': 'GSFMold RummaGEO/Gene', #'GSAE[rummageo/gene, e=50, d=1024, depth=2]',
  # '2025-05-07_version_83_K_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=160150.ckpt': 'GSFMold Rummagene', #'GSAE[rummagene, e=50, d=1024, depth=2]',
  # '2025-05-07_version_85_K_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=73300.ckpt': 'GSFMold RummaGEO', #'GSAE[rummageo, e=50, d=1024, depth=2]',
  # '2025-05-09_version_20_J_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=40050.ckpt': 'GSFM Rummagene',
  '2025-05-09_version_47_J_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=18350.ckpt': 'GSFM RummaGEO',
  # '2025-05-09_version_46_J_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=58350.ckpt': 'GSFM RummaGEO/Gene',
  '2025-05-27_version_0_J_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=86950.ckpt': 'GSFM Rummagene',
  '2025-05-27_version_2_J_GeneSetMaskedEmbeddingAutoEncoder_epoch=49-step=105250.ckpt': 'GSFM RummaGEO/Gene',
  '2025-01-21-prismexp-archs4-gobp.y': 'PrismEXP ARCHS4/GO BP',
  '2025-01-22-prismexp-archs4-rummagene.y': 'PrismEXP ARCHS4/Rummagene',
  '2025-01-22-prismexp-rummagene-rummagene.y': 'PrismEXP Rummagene/Rummagene',
  'rummage': 'RummaGEO/Gene',
  'rummagene': 'Rummagene',
  'rummageo': 'RummaGEO',
  'enrichr': 'Enrichr',
  'archs4': 'ARCHS4',
  'geneptsim': 'GenePT',
  'random': 'Random',
})
# fig.savefig('figs/3.svg', bbox_inches='tight')
plt.show()

#%%
df_.loc[pd.isna(df_['dropout']), 'dropout'] = 0.2
df_.loc[df_['model_cls'].str.startswith('L') & pd.isna(df_['partition']), 'partition'] = 0.0
df_.loc[pd.isna(df_['partition']), 'partition'] = 0.8
df_.loc[pd.isna(df_['weighted_loss']), 'weighted_loss'] = 'none'
df_.loc[pd.isna(df_['latent_dim']), 'latent_dim'] = 0

#%%
d_ = df_.groupby(list(set(df_.columns)-{'roc_auc', 'es', 'f1', 'i', 'split','_class_path','creation_timestamp'}))['roc_auc'].median().reset_index()
d_

d_['type'] = d_['model_cls'].apply(lambda m: {
  'L': 'VAE',
  'J': 'DAE',
  'F': 'EPDAE',
  'A': 'mEDAE',
  'B': 'EDAE',
}.get(m.partition('.')[0]))

d_['dataset'] = d_['data_cls'].apply(lambda m: {
  'RummageoDataModule': 'RummaGEO',
  'RummageDataModule': 'RummaGEO/Gene',
  'RummageneDataModule': 'Rummagene',
}.get(m))

d_ = d_[(d_['epochs']<=100)&(d_['name'].str.startswith('2025-05-09'))]

#%%
# d_[(d_['type'] == 'GSEPAE')& (d_['partition']==0) & (d_['dropout']==0.2) & (d_['weighted_loss']=='none')].apply(set)

#%%
def select(attrs):
  attrs = {k:v for k,v in dict(attr_defaults, **attrs).items() if v is not None}
  d = d_
  for k, v in attrs.items():
    if v is not None:
      d = d[(d[k] == v)]
  return d

#%%
col_order = ['ChEA_2022', 'GO_Biological_Process_2023', 'GWAS_Catalog_2023', 'KEGG_2021_Human']
attr_defaults = {'dropout': 0.2, 'partition': 0.8, 'depth': 2, 'd_model': 256, 'dataset': 'Rummagene', 'type': 'DAE', 'weighted_loss': 'none'}
def clean(s):
  if s == 'data_cls=RummageneDataModule': return 'data=Rummagene'
  return s

params = {'legend.title_fontsize': 15,'legend.fontsize': 14}
matplotlib.rcParams.update(params)

def make_line_plot(attrs, hue, labels=['A','B','C','D'], legend_pos_override=1.1):
  attrs = {k:v for k,v in dict({k:v for k,v in attr_defaults.items() if k != hue}, **attrs).items() if v is not None}
  d = d_
  for k, v in attrs.items():
    if v is not None:
      d = d[(d[k] == v)]
  g = (
  sns.FacetGrid(
    d, col='library', hue=hue, col_order=col_order, col_wrap=2, sharey=False, height=3.54, aspect=1., legend_out=True,
    subplot_kws=dict(adjustable='box'),
    )
    .map(sns.lineplot, 'epochs', 'roc_auc')
  )
  g.set_titles("{col_name}")
  for ax_label, ax in zip(labels, g.axes):
    ax.set_title(ax.get_title().replace('_', ' '))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.text(-0.1, 1.15, ax_label, transform=ax.transAxes, size=20, weight='bold')

  g.axes[0].set_ylabel('Median AUROC')
  g.axes[2].set_ylabel('Median AUROC')
  g.axes[2].set_xlabel('Epochs')
  g.axes[3].set_xlabel('Epochs')
  # g.fig.suptitle('\n'.join(sorted(clean(f"{k}: {v}") for k,v in attrs.items())), y=-0.01)
  plt.tight_layout()
  g.fig.text(1.1, 1.5, '\n'.join(sorted(clean(f"{k}: {v}") for k,v in attrs.items() if k!=hue)), transform=ax.transAxes, size=14)
  g.add_legend(loc="upper center", bbox_to_anchor=[legend_pos_override,0.55])
  # plt.tight_layout()
  return g

#%%
d_.groupby(['dataset', 'dropout', 'partition', 'depth', 'd_model', 'type', 'weighted_loss', 'epochs'])['name'].nunique().reset_index().groupby('name').agg(set)

# %%
make_line_plot(dict(partition=0.0, depth=2, dropout=0.2, d_model=256, weighted_loss='none', type='DAE'), 'type')#.savefig('figs/2g.svg', bbox_inches='tight')


#%%
d_ = d_[(d_['epochs']<=50)]

#%%
make_line_plot(dict(), 'partition', 1.)#.savefig('figs/2a.svg', bbox_inches='tight')

#%%
select(dict())
#%%
d__ = select(dict(partition=None))
sns.heatmap(pd.concat({
  k: pd.get_dummies(d___['partition'].astype(str)).corrwith(d___['roc_auc'])
  for k, d___ in d__[d__['epochs']==50].groupby('library')
}).unstack())
# %%
make_line_plot(dict(partition=0), 'dropout', 1.05)#.savefig('figs/2b.svg', bbox_inches='tight')

#%%
d__ = select(dict(partition=0, dropout=None))
sns.heatmap(pd.concat({
  k: pd.get_dummies(d___['dropout'].astype(str)).corrwith(d___['roc_auc'])
  for k, d___ in d__[d__['epochs']==50].groupby('library')
}).unstack())
#%%
make_line_plot(dict(partition=0, dropout=0.2), 'depth', 1.05)#.savefig('figs/2c.svg', bbox_inches='tight')
#%%
make_line_plot(dict(partition=0, depth=2, dropout=0.2), 'd_model', 1.)#.savefig('figs/2d.svg', bbox_inches='tight')

#%%
make_line_plot(dict(partition=0, depth=2, dropout=0.2, d_model=256), 'weighted_loss', 0.95)#.savefig('figs/2e.svg', bbox_inches='tight')

# %%
make_line_plot(dict(partition=0, depth=2, dropout=0.2, d_model=256, weighted_loss='none'), 'type', 1.)#.savefig('figs/2f.svg', bbox_inches='tight')

#%%
make_line_plot(dict(partition=0.0, depth=2, dropout=0.2, d_model=256, weighted_loss='none', type='DAE'), 'dataset', 0.9)#.savefig('figs/2h.svg', bbox_inches='tight')

# %%
d__ = d_[(d_['epochs']==50)]

#%%
pd.concat({
  k: pd.get_dummies(d__[['partition', 'dropout', 'depth', 'd_model', 'weighted_loss', 'type', 'dataset']].astype(str)).corrwith(d__['roc_auc'])
  for k, d__ in d_[(d_['epochs']==50)].groupby('library')
}).unstack(0)

#%%
d__z
#%%

#%%
d_['epochs'].value_counts()
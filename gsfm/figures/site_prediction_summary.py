#%%
import sys; sys.path.insert(0, '..')
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib, matplotlib.pyplot as plt

# %%
dfs = []
for f in pathlib.Path('../data/site-predictions/').rglob('eval.tsv'):
  df = pd.read_csv(f, sep='\t')
  df = df.groupby(list(set(df.columns)-{'i', 'roc_auc'}))['roc_auc'].median().reset_index()
  df['library'] = df['library'].apply(lambda lib: lib.partition('/')[2].partition('.')[0].replace('_', ' ').replace('ASCTpB', 'ASCT+B'))
  df['library_avg'] = df['roc_auc'].mean()
  dfs.append(df)

df = pd.concat(dfs)

# %%
matplotlib.rc('axes', titlesize=15, labelsize=15)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

#%%
for name, records in df.groupby('name'):
  fig = plt.figure(figsize=(3.54, 7.08), dpi=600)
  sns.boxenplot(records.sort_values('library_avg'), x='roc_auc', y='library')
  plt.vlines([0.5], ymin=0, ymax=17, linestyle='dashed', color='black')
  plt.ylabel('')
  plt.xlabel('Median AUROC')
  plt.title(name)
  # fig.savefig('figs/4.svg', bbox_inches='tight')
  plt.show()

#%%
import numpy as np
import pathlib
import torch
import seaborn as sns
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from gsfm import Vocab, GSFM
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import collections
import re
from tqdm.auto import tqdm

#%%
def read_gmt(f):
  with pathlib.Path(f).open('r') as fr:
    for line in fr:
      line_split = list(filter(None, line.rstrip('\r\n').split('\t')))
      if len(line_split) < 3: continue
      term, desc, *genes = line_split
      yield term, desc, genes

#%%
def classic_tfid_umap(gmt):
  keys, values = zip(*gmt.items())
  vectorized = TfidfVectorizer(analyzer=lambda gs: gs).fit_transform(values)
  return pd.DataFrame(
    UMAP(random_state=42).fit_transform(vectorized),
    columns=['UMAP-1', 'UMAP-2'],
    index=keys,
  )

#%%
def classic_mh_umap(gmt):
  keys, values = zip(*gmt.items())
  vectorized = pd.DataFrame([{g: 1 for g in gs} for gs in values], index=keys).fillna(0)
  return pd.DataFrame(
    UMAP(random_state=42).fit_transform(vectorized),
    columns=['UMAP-1', 'UMAP-2'],
    index=keys,
  )

#%%
def classic_mh_pca(gmt):
  keys, values = zip(*gmt.items())
  vectorized = pd.DataFrame([{g: 1 for g in gs} for gs in values], index=keys).fillna(0)
  pca = PCA()
  view = pca.fit_transform(vectorized)[:, [0, 1]]
  return pd.DataFrame(
    view,
    columns=[f"PC-1 {pca.explained_variance_ratio_[0]*100:0.2f}%", f"PC-2 {pca.explained_variance_ratio_[1]*100:0.2f}%"],
    index=keys,
  )
#%%
def classic_tfid_pca(gmt):
  keys, values = zip(*gmt.items())
  vectorized = TfidfVectorizer(analyzer=lambda gs: gs).fit_transform(values)
  pca = PCA()
  view = pca.fit_transform(vectorized)[:, [0, 1]]
  return pd.DataFrame(
    view,
    columns=[f"PC-1 {pca.explained_variance_ratio_[0]*100:0.2f}%", f"PC-2 {pca.explained_variance_ratio_[1]*100:0.2f}%"],
    index=keys,
  )
#%%
# load gsfm vocabulary and model weights
vocab = Vocab.from_pretrained('maayanlab/gsfm')
gsfm = GSFM.from_pretrained('maayanlab/gsfm')
gsfm.eval()

#%%
def gsfm_umap(gmt):
  keys, values = zip(*gmt.items())
  token_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(vocab(geneset)) for geneset in values], padding_value=1, batch_first=True)
  encoding = gsfm.encode(token_ids).detach().cpu().numpy()
  return pd.DataFrame(
    UMAP(random_state=42).fit_transform(encoding),
    columns=['UMAP-1', 'UMAP-2'],
    index=keys,
  )

#%%
def gsfm_pca(gmt):
  keys, values = zip(*gmt.items())
  token_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(vocab(geneset)) for geneset in values], padding_value=1, batch_first=True)
  encoding = gsfm.encode(token_ids).detach().cpu().numpy()
  pca = PCA()
  view = pca.fit_transform(encoding)
  return pd.DataFrame(
    view,
    columns=[
      f"PC-{n} {r*100:0.2f}%"
      for n, r in enumerate(pca.explained_variance_ratio_, start=1)
    ],
    index=keys,
  )

#%%
keys, _, values = zip(*read_gmt('GTEx_Tissues_2023.gmt'))
GMT = dict(zip(keys, values))


# %%
hue = {key: m.group(5) for key in keys for m in (re.match(r'(.+?)( - (.+?))? (Male|Female) (.+?) (Up|Down)', key),) }

#%%
def scatter(pca, hue):
  x, y, *_ = pca.columns
  ax = sns.scatterplot(
    pca,
    x=x,
    y=y,
    s=10,
    legend=True,
    hue=hue,
  )
  sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
  plt.show()

#%%
scatter(classic_mh_pca(GMT), hue)
scatter(classic_tfid_pca(GMT), hue)
scatter(classic_tfid_umap(GMT), hue)
scatter(gsfm_umap(GMT), hue)
#%%
scatter(gsfm_pca(GMT), hue)

#%%
gsfm_pca(GMT).corrwith(pd.get_dummies(pd.Series(hue))['20-29']).sort_values()


#%%
scatter(gsfm_pca(GMT)[['PC-49 0.24%', 'PC-59 0.18%']], hue)


#%%
gsfm_pca(GMT)

#%%
keys, _, values = zip(*read_gmt('Tabula_Sapiens.gmt'))
GMT = dict(zip(keys, values))

#%%
keys
#%%
hue = pd.Series({key: m.group(1) for key in keys for m in (re.match(r'(.+?)-(.+)', key),) })
hue.name = 'hue'

#%%
ax = sns.scatterplot(
  classic_mh_umap(GMT).merge(hue, left_index=True, right_index=True),
  x='UMAP-1',
  y='UMAP-2',
  # s=10,
  legend=True,
  hue=hue,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

#%%
ax = sns.scatterplot(
  gsfm_umap(GMT).merge(hue, left_index=True, right_index=True),
  x='UMAP-1',
  y='UMAP-2',
  # s=10,
  hue=hue,
  legend=True,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

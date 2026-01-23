#%%
import pathlib
import torch
import seaborn as sns
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from gsfm import Vocab, GSFM
from sklearn.feature_extraction.text import TfidfVectorizer
import glasbey
import sklearn.cluster
import re

#%%
def read_gmt(f):
  with pathlib.Path(f).open('r') as fr:
    for line in fr:
      line_split = list(filter(None, line.rstrip('\r\n').split('\t')))
      if len(line_split) < 3: continue
      term, desc, *genes = line_split
      yield term, desc, genes

#%%
def multihot(gmt):
  keys, values = zip(*gmt.items())
  return pd.DataFrame([{g: 1 for g in gs} for gs in values], index=keys).fillna(0)

def idf(gmt):
  keys, values = zip(*gmt.items())
  return TfidfVectorizer(analyzer=lambda gs: gs).fit_transform(values)

def umap(vectorized):
  return pd.DataFrame(
    UMAP(random_state=42).fit_transform(vectorized),
    columns=['UMAP-1', 'UMAP-2'],
    index=keys,
  )

def pca(vectorized):
  pca = PCA()
  view = pca.fit_transform(vectorized)[:, [0, 1]]
  return pd.DataFrame(
    view,
    columns=[f"PC-1 {pca.explained_variance_ratio_[0]*100:0.2f}%", f"PC-2 {pca.explained_variance_ratio_[1]*100:0.2f}%"],
    index=keys,
  )

def hdbscan(vectorized):
  hdbscan = sklearn.cluster.HDBSCAN()
  return pd.Series(hdbscan.fit_predict(vectorized), index=keys).astype(str)

#%%
# load gsfm vocabulary and model weights
vocab = Vocab.from_pretrained('maayanlab/gsfm-rummagene')
gsfm = GSFM.from_pretrained('maayanlab/gsfm-rummagene')
gsfm.eval()

#%%
def gsfm_encode(gmt):
  keys, values = zip(*gmt.items())
  token_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(vocab(geneset)) for geneset in values], padding_value=1, batch_first=True)
  return gsfm.encode(token_ids).detach().cpu().numpy()

#%%
keys, _, values = zip(*read_gmt('data/GTEx_Tissues_V8_2023.gmt'))
GMT = dict(zip(keys, values))
hue = pd.Series({key: m.group(1) for key in keys for m in (re.match(r'(.+?)( - (.+?))? (Male|Female) (.+?) (Up|Down)', key),) }, name='Organ')

#%%
palette = dict(zip(hue.unique(), glasbey.create_palette(hue.nunique())))

#%%
def scatter(title, pca, hue=None, palette=None, ax=None):
  x, y, *_ = pca.columns
  sax = sns.scatterplot(
    pca,
    x=x,
    y=y,
    s=hue.apply(lambda hue: 20 if hue else 1) if hue is not None and hue.nunique() == 2 else 10,
    legend=True,
    hue=hue.astype(str) if hue is not None else None,
    palette=palette,
    ax=ax,
  )
  plt.legend(markerscale=2)
  sax.set_title(title)
  if hue is not None:
    sns.move_legend(sax, "upper left", bbox_to_anchor=(1, 1))
  plt.show()

#%%
scatter('classic_mh_pca', pca(multihot(GMT)), hue, palette)
scatter('classic_tfid_umap', umap(idf(GMT)), hue, palette)
scatter('gsfm_umap', umap(gsfm_encode(GMT)), hue, palette)
scatter('PCA(IDF(GTEx))', pca(idf(GMT)), hue, palette)
scatter('PCA(GSFM.Encode(GTEx))', pca(gsfm_encode(GMT)), hue, palette)

#%%
X_pca = pca(idf(GMT))
X_gsfm_pca = pca(gsfm_encode(GMT))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
x, y, *_ = X_pca.columns
g = sns.scatterplot(
  X_pca,
  x=x,
  y=y,
  s=10,
  legend=False,
  hue=hue,
  palette=palette,
  ax=ax1,
)
ax1.set_title('PCA(IDF(GTEx_Tissues_V8_2023))')
x, y, *_ = X_gsfm_pca.columns
g = sns.scatterplot(
  X_gsfm_pca,
  x=x,
  y=y,
  s=10,
  legend=True,
  hue=hue,
  palette=palette,
  ax=ax2,
)
ax2.set_title('PCA(GSFM(GTEx_Tissues_V8_2023))')
fig.legend(
  markerscale=2,
  loc="upper center",
  bbox_to_anchor=(.5, 0),
  ncol=4,
  fancybox=True,
)
g.legend_.remove()
plt.savefig('/home/u8sand/Downloads/pca.pdf', bbox_inches='tight')
plt.show()

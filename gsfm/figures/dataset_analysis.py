#%%
import sklearn.decomposition
import sklearn.feature_extraction.text
from tqdm import tqdm
from gsfm import utils

#%%
rummagene_train = [
  clean_genes
  for _, year, genes in utils.read_gmt(f'data/2025-08-29-rummagene.gmt')
  for clean_genes in (list(set(map(str.upper, filter(None, genes)))),)
  if len(clean_genes) > 10 and len(clean_genes) < 1000
]
len(rummagene_train)

#%%
rummageo_train = [
  clean_genes
  for f in [f'data/2025-08-29-rummageo-human.gmt', f'data/2025-08-29-rummageo-mouse.gmt']
  for _, year, genes in utils.read_gmt(f)
  for clean_genes in (list(set(map(str.upper, filter(None, genes)))),)
  if len(clean_genes) > 10 and len(clean_genes) < 1000
]

#%%
unique_rummagene_train = set(map(frozenset, rummagene_train))

# %%
unique_rummageo_train = set(map(frozenset, rummageo_train))
len(unique_rummageo_train)

# %%
unique_rummage_train = unique_rummageo_train | unique_rummagene_train
assert len(rummagene_train) + len(rummageo_train) == len(unique_rummage_train)
# I.e. verified no true duplicated gene sets in training data

#%%
# let's look for *near duplicates*

# %%
import scipy.special

def jaccard(gs1, gs2):
  i = len(set(gs1).intersection(gs2))
  return i/(len(gs1)+len(gs2)-i)

rummage_train = rummagene_train + rummageo_train

#%%
import numpy as np
N = 100_000_000
I = np.random.choice(np.arange(len(rummage_train)), size=N)
J = np.random.choice(np.arange(len(rummage_train)), size=N)
D = pd.DataFrame([dict(
  i=i,
  j=j,
  s=jaccard(rummage_train[i], rummage_train[j])
) for i, j in tqdm(zip(I, J),total=N) if i!=j])
D.drop_duplicates(['i','j'], inplace=True)
D['i_is_rummagene']=D['i']<len(rummagene_train)
D['j_is_rummagene']=D['j']<len(rummagene_train)
D['kind'] = D.apply(lambda row: {
  (True, True): 'Rummagene',
  (True, False): 'Cross Database',
  (False, True): 'Cross Database',
  (False, False): 'RummaGEO',
}[(row['i']<len(rummagene_train), row['j']<len(rummagene_train))], axis=1)

#%%
D[D['s']>0.9].count()

#%%
D.to_csv('data/D.tsv',sep='\t',index=None)

#%%
import pandas as pd

D = pd.read_csv('data/D.tsv', sep='\t')
D['kind'].replace({'cross-db':'Cross DB', 'rummagene': 'Rummagene', 'rummageo': 'RummaGEO'}, inplace=True)
D
#%%
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.histplot(
  D,
  x='s',
  hue='kind',
  bins=50,
  multiple='stack',
)
ax.get_legend().set_title('')
ax.set_yscale('log')
plt.ylabel('Gene Set Pairs')
plt.xlabel('Jaccard Similarity')
plt.tight_layout()
plt.savefig('data/jaccard-similarity-across-100mln-gene-set-pairs.pdf')
# plt.show()

#%%
# let's look for bias
from collections import Counter

rummagene_gene_counts = Counter()
for gs in tqdm(rummagene_train):
  rummagene_gene_counts.update(gs)

#%%
rummageo_gene_counts = Counter()
for gs in tqdm(rummageo_train):
  rummageo_gene_counts.update(gs)

#%%
with open('data/gene_counts.pickle', 'wb') as fw:
  pickle.dump(dict(
    rummagene_gene_counts=rummagene_gene_counts,
    rummageo_gene_counts=rummageo_gene_counts,
  ), fw)

#%%
import pickle
with open('data/gene_counts.pickle', 'rb') as fr:
  data = pickle.load(fr)
  rummagene_gene_counts = data['rummagene_gene_counts']
  rummageo_gene_counts = data['rummageo_gene_counts']

# %%
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2)
pd.Series(rummagene_gene_counts).hist(log='y',ax=ax1, bins=20)
pd.Series(rummageo_gene_counts).hist(log='y',ax=ax2, bins=20)
ax1.set_title(f'Rummagene\nGS={len(rummagene_train):,}, G={len(rummagene_gene_counts):,}')
ax1.set_ylabel('Genes')
ax1.set_xlabel('Sets')
ax2.set_title(f'RummaGEO\nGS={len(rummageo_train):,}, G={len(rummageo_gene_counts):,}')
ax2.set_xlabel('Sets')
fig.savefig('data/rummage-gene-counts.pdf')

#%%
rummageo_gene_counts_aligned = pd.Series(rummageo_gene_counts)[pd.Series(rummageo_gene_counts).index.isin(pd.Series(rummagene_gene_counts).index)]

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2)
pd.Series(rummagene_gene_counts).hist(log='y',ax=ax1, bins=20)
rummageo_gene_counts_aligned.hist(log='y',ax=ax2, bins=20)
ax1.set_title(f'Rummagene\nGS={len(rummagene_train):,}, G={len(rummagene_gene_counts):,}')
ax1.set_ylabel('Genes')
ax1.set_xlabel('Sets')
ax2.set_title(f'RummaGEO\nGS={len(rummageo_train):,}, G={len(rummageo_gene_counts_aligned):,}')
ax2.set_xlabel('Sets')


#%%
sum(map(len, rummagene_train)) / len(rummagene_train)

# %%
sum(map(len, rummageo_train)) / len(rummageo_train)

# %%
sum(map(len, rummagene_train+rummageo_train)) / len(rummagene_train+rummageo_train)

#%%
rummagene_genes = {g for gs in rummagene_train for g in gs}
rummageo_genes = {g for gs in rummageo_train for g in gs}

# %%
len(rummagene_train+rummageo_train)
# %%
len(rummageo_genes)

#%%
svd = sklearn.decomposition.PCA(4)
idf = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=lambda gs: gs)
X_pca = svd.fit_transform(
  idf.fit_transform(rummagene_train+rummageo_train)
)

#%%
svd.explained_variance_ratio_
# %%
import matplotlib.pyplot as plt

plt.title('PCA(IDF(Rummagene+RummaGEO))')
plt.scatter(X_pca[:len(rummagene_train), 0], X_pca[:len(rummagene_train), 1], label='RummaGene', s=1, alpha=0.1, rasterized=True)
plt.scatter(X_pca[len(rummagene_train):, 0], X_pca[len(rummagene_train):, 1], label='RummaGEO', s=1, alpha=0.1, rasterized=True)
plt.xlabel(f"PC-1 ({svd.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC-2 ({svd.explained_variance_ratio_[1]*100:.2f}%)")
legend = plt.legend()
for lh in legend.legend_handles:
  lh.set_alpha([1])
  lh.set_sizes([12])
plt.tight_layout()
plt.savefig('data/pca-idf-rummage-pc12.pdf', dpi=300)
# plt.show()

# %%
import matplotlib.pyplot as plt

plt.title('PCA(IDF(Rummagene+RummaGEO))')
plt.scatter(X_pca[:len(rummagene_train), 2], X_pca[:len(rummagene_train), 3], label='RummaGene', s=1, alpha=0.1, rasterized=True)
plt.scatter(X_pca[len(rummagene_train):, 2], X_pca[len(rummagene_train):, 3], label='RummaGEO', s=1, alpha=0.1, rasterized=True)
plt.xlabel(f"PC-3 ({svd.explained_variance_ratio_[2]*100:.2f}%)")
plt.ylabel(f"PC-4 ({svd.explained_variance_ratio_[3]*100:.2f}%)")
legend = plt.legend()
for lh in legend.legend_handles:
  lh.set_alpha([1])
  lh.set_sizes([12])
plt.tight_layout()
plt.savefig('data/pca-idf-rummage-pc34.pdf', dpi=300)
plt.show()

#%%

#%%
from umap import UMAP
umap = UMAP()
idf = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=lambda gs: gs)
X_umap = umap.fit_transform(
  idf.fit_transform(rummagene_train+rummageo_train)
)
# %%

import matplotlib.pyplot as plt

plt.title('UMAP(IDF(Rummagene+RummaGEO))')
plt.scatter(X_umap[:len(rummagene_train), 0], X_umap[:len(rummagene_train), 1], label='RummaGene', s=1, alpha=0.1, rasterized=True)
plt.scatter(X_umap[len(rummagene_train):, 0], X_umap[len(rummagene_train):, 1], label='RummaGEO', s=1, alpha=0.1, rasterized=True)
plt.xlabel(f"UMAP-1")
plt.ylabel(f"UMAP-2")
legend = plt.legend()
for lh in legend.legend_handles:
  lh.set_alpha([1])
  lh.set_sizes([12])
plt.tight_layout()
plt.savefig('data/umap-idf-rummage.pdf', dpi=300)
plt.show()

# %%

import matplotlib.pyplot as plt

plt.title('UMAP(IDF(Rummagene+RummaGEO))')
plt.scatter(X_umap[:len(rummagene_train), 0], X_umap[:len(rummagene_train), 1], label='RummaGene', s=1, alpha=0.1, rasterized=True)
plt.scatter(X_umap[len(rummagene_train):, 0], X_umap[len(rummagene_train):, 1], label='RummaGEO', s=1, alpha=0.1, rasterized=True)
plt.xlabel(f"UMAP-1")
plt.ylabel(f"UMAP-2")
plt.xlim((X_umap[:, 0].mean()-2*X_umap[:, 0].std(), X_umap[:, 0].mean()+2*X_umap[:, 0].std(),))
plt.ylim((X_umap[:, 1].mean()-2*X_umap[:, 1].std(), X_umap[:, 1].mean()+2*X_umap[:, 1].std(),))
legend = plt.legend()
for lh in legend.legend_handles:
  lh.set_alpha([1])
  lh.set_sizes([12])
plt.tight_layout()
plt.savefig('data/umap-idf-rummage-zoom.pdf', dpi=300)
plt.show()

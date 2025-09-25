import pandas as pd
from tqdm.auto import tqdm
from gsfm import utils

#%%
# load GO
gene2go = pd.read_csv('data/gene2go.tsv', sep='\t')
# curl -LO https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
human_geneinfo = pd.read_csv('data/Homo_sapiens.gene_info.gz', sep='\t')
# curl -LO https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz
mouse_geneinfo = pd.read_csv('data/Mus_musculus.gene_info.gz', sep='\t')
geneinfo = pd.concat([human_geneinfo, mouse_geneinfo]).drop_duplicates(['GeneID'])
gene2go = gene2go.merge(geneinfo[['GeneID', 'Symbol']], on='GeneID')
gene2go = gene2go.dropna(subset=['Symbol'])

#%%
# load GO mitocondrial gene sets gene sets
GO_Mito = {}
mito_gene2go = gene2go[gene2go['GO_term'].str.contains('mito')]
for GO, records in mito_gene2go.groupby('GO_ID'):
  symbols = list(map(str.upper, records['Symbol'].unique()))
  GO_Mito[GO] = symbols

#%%
# load rummagene gene sets
rummagene_train = [
  (term, year, clean_genes)
  for term, year, genes in utils.read_gmt(f'data/2025-08-29-rummagene.gmt')
  for clean_genes in (list(set(map(str.upper, filter(None, genes)))),)
  if len(clean_genes) > 10 and len(clean_genes) < 1000
]

#%%
# find rummagene sets with high overlap with GO mitocondrial gene sets
matches = set()
for i, (_, _, rummagene_gs) in enumerate(tqdm(rummagene_train)):
  for term, mito_gs in GO_Mito.items():
    overlap_ratio = len(set(rummagene_gs) & set(mito_gs)) / len(set(mito_gs))
    if overlap_ratio >= 0.8:
      matches.add(i)
      break

#%%
len(matches)
# 165161

#%%
with open('data/2025-09-02-rummagene-no-mito.gmt', 'w') as fw:
  for i, (term, year, rummagene_gs) in enumerate(tqdm(rummagene_train)):
    if i in matches: continue
    print(term, year, *rummagene_gs, sep='\t', file=fw)

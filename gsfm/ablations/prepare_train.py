#%%
import pandas as pd
from tqdm.auto import tqdm
from gsfm import utils

#%%[markdown]
# # Rummagene

#%%
rummagene_PMCs = pd.Series({
  term: term.partition('-')[0]
  for term, _, _ in utils.read_gmt('data/latest.gmt')
}).to_frame('PMCID')

#%%
import pandas as pd
PMCs = pd.read_csv('data/PMC-ids.csv.gz', converters={'PMCID': str})

#%%
PMCs = PMCs.merge(rummagene_PMCs, how='right', on='PMCID')

PMCs['Year'].value_counts().sort_index()

#%%
train_PMCIds = set(PMCs[PMCs['Year'] <= 2016]['PMCID'].unique())

#%%
PMC_year_lookup = PMCs.groupby('PMCID')['Year'].min().to_dict()
PMC_year_lookup

#%%
#%%
gene_set_hashes = set()
with open('data/2025-08-29-rummagene.gmt', 'w') as fw:
  for term, desc, gene_set in tqdm(read_gmt('data/latest.gmt')):
    if frozenset(gene_set) in gene_set_hashes: continue # no gene sets
    pmcid = term.partition('-')[0]
    year = PMC_year_lookup.get(pmcid)
    if pd.isna(year): continue
    print(term, str(int(year)), *gene_set, sep='\t', file=fw)
    gene_set_hashes.add(frozenset(gene_set))

#%%[markdown]
# # RummaGEO Human

#%%
human_rummageo_GSEs = pd.Series({
  term: gse
  for term, _, _ in read_gmt('data/human-geo-auto.gmt')
  for gses in (term.partition('-')[0],)
  for gse in gses.split(',')
}).to_frame('GSE')

#%%
import h5py
f = h5py.File('/home/u8sand/Programs/work/Downloads/human_gene_v2.latest.h5', 'r')

human_GSEs = pd.DataFrame({
  'submission_date': f['meta/samples/submission_date'].asstr()[:],
  'series_id': f['meta/samples/series_id'].asstr()[:],
})
human_GSEs['series_id'] = human_GSEs['series_id'].str.split(',')
human_GSEs = human_GSEs.explode('series_id')
human_GSEs['Year'] = human_GSEs['submission_date'].map(lambda d: int(d.rpartition(' ')[-1]))
human_GSEs = human_GSEs.groupby('series_id')['Year'].agg(min).to_frame()

#%%
d = human_GSEs.merge(human_rummageo_GSEs, how='right', left_index=True, right_on='GSE')
d.loc[pd.isna(d['Year']), 'Year'] = float('nan')

#%%
GSE_year_lookup = d.groupby('GSE')['Year'].min().to_dict()

#%%
gene_set_hashes = set()
with open('data/2025-08-29-rummageo-human.gmt', 'w') as fw:
  for term, desc, gene_set in tqdm(read_gmt('data/human-geo-auto.gmt')):
    if frozenset(gene_set) in gene_set_hashes: continue # no gene sets
    gse = term.partition('-')[0]
    year = GSE_year_lookup.get(gse)
    if pd.isna(year): continue
    print(term, str(int(year)), *gene_set, sep='\t', file=fw)
    gene_set_hashes.add(frozenset(gene_set))

# %%
mouse_rummageo_GSEs = pd.Series({
  term: gse
  for term, _, _ in read_gmt('data/mouse-geo-auto.gmt')
  for gses in (term.partition('-')[0],)
  for gse in gses.split(',')
}).to_frame('GSE')

#%%
import h5py
f = h5py.File('/home/u8sand/Programs/work/Downloads/mouse_gene_v2.latest.h5', 'r')

mouse_GSEs = pd.DataFrame({
  'submission_date': f['meta/samples/submission_date'].asstr()[:],
  'series_id': f['meta/samples/series_id'].asstr()[:],
})
mouse_GSEs['series_id'] = mouse_GSEs['series_id'].str.split(',')
mouse_GSEs = mouse_GSEs.explode('series_id')
mouse_GSEs['Year'] = mouse_GSEs['submission_date'].map(lambda d: int(d.rpartition(' ')[-1]))
mouse_GSEs = mouse_GSEs.groupby('series_id')['Year'].agg(min).to_frame()

#%%
d = mouse_GSEs.merge(mouse_rummageo_GSEs, how='right', left_index=True, right_on='GSE')
d.loc[pd.isna(d['Year']), 'Year'] = float('nan')

#%%
GSE_year_lookup = d.groupby('GSE')['Year'].min().to_dict()

#%%
gene_set_hashes = set()
with open('data/2025-08-29-rummageo-mouse.gmt', 'w') as fw:
  for term, desc, gene_set in tqdm(read_gmt('data/mouse-geo-auto.gmt')):
    if frozenset(gene_set) in gene_set_hashes: continue # no gene sets
    gse = term.partition('-')[0]
    year = GSE_year_lookup.get(gse)
    if pd.isna(year): continue
    print(term, str(int(year)), *gene_set, sep='\t', file=fw)
    gene_set_hashes.add(frozenset(gene_set))

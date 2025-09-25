import pandas as pd
from gsfm import utils

gene2go = pd.read_csv('data/gene2go.tsv', sep='\t')
pmid_to_year = pd.read_csv('data/pmid_year.csv', converters={'PMID': str}) # produced by fetch_pmids_years
human_geneinfo = pd.read_csv(utils.cached_wget('https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz', 'data/Homo_sapiens.gene_info.gz'), sep='\t')
mouse_geneinfo = pd.read_csv(utils.cached_wget('https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz', 'data/Mus_musculus.gene_info.gz'), sep='\t')
geneinfo = pd.concat([human_geneinfo, mouse_geneinfo]).drop_duplicates(['GeneID'])

gene2go['PubMed'] = gene2go['PubMed'].str.split('|')
gene2go = gene2go.explode('PubMed')
gene2go = gene2go.merge(pmid_to_year, left_on='PubMed', right_on='PMID', how='left')
gene2go = gene2go.merge(geneinfo[['GeneID', 'Symbol']], on='GeneID')
gene2go = gene2go.dropna(subset=['Symbol'])
gene2go['Year'] = gene2go['Year'].apply(lambda s: int(s.split('-')[-1]) if type(s) == str else s)

for category, all_records in gene2go.groupby('Category'):
  with open(f"data/GO_{category}.gmt", 'w') as fw:
    for (GO_ID, GO_term), records in all_records.groupby(['GO_ID', 'GO_term']):
      symbols = records['Symbol'].unique()
      if len(symbols) < 5: continue
      print(f"{GO_term} ({GO_ID})", '', *symbols, sep='\t', file=fw)

for year in range(2000, 2026):
  @utils.if_not_exists(f"data/GO/{year}.gmt")
  def _(file):
    with open(file, 'w') as fw:
      for GO, records in gene2go[gene2go['Year']<year].groupby('GO_ID'):
        symbols = records['Symbol'].unique()
        if len(symbols) < 5: continue
        print(GO, '', *symbols, sep='\t', file=fw)

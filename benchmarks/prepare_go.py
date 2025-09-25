import pandas as pd

gene2go = pd.read_csv('data/gene2go.tsv', sep='\t')
pmid_to_year = pd.read_csv('data/pmid_year.csv', converters={'PMID': str}) # produced by fetch_pmids_years
# curl -LO https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
human_geneinfo = pd.read_csv('data/Homo_sapiens.gene_info.gz', sep='\t')
# curl -LO https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz
mouse_geneinfo = pd.read_csv('data/Mus_musculus.gene_info.gz', sep='\t')
geneinfo = pd.concat([human_geneinfo, mouse_geneinfo]).drop_duplicates(['GeneID'])

geneinfo = {}
gene2go['PubMed'] = gene2go['PubMed'].str.split('|')
gene2go = gene2go.explode('PubMed')
gene2go = gene2go.merge(pmid_to_year, left_on='PubMed', right_on='PMID')
gene2go = gene2go.merge(geneinfo[['GeneID', 'Symbol']], on='GeneID')
gene2go = gene2go.dropna(subset=['Symbol'])
gene2go['Year'] = gene2go['Year'].apply(lambda s: int(s.split('-')[-1]))

gene2go['Year'].value_counts().sort_index().iloc[-26:]

for year in range(2000, 2026):
  with open(f"data/GO/{year}.gmt", 'w') as fw:
    for GO, records in gene2go[gene2go['Year']<year].groupby('GO_ID'):
      symbols = records['Symbol'].unique()
      if len(symbols) < 5: continue
      print(GO, '', *symbols, sep='\t', file=fw)

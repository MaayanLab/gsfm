import pandas as pd

# curl -L https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz | gunzip > data/gene2go.tsv
gene2go = pd.read_csv('data/gene2go.tsv', sep='\t')
gene2go['PubMed'] = gene2go['PubMed'].str.split('|')
pd.Series(gene2go['PubMed'].explode().unique()).to_csv('data/pmids.txt', header=False, index=None)

import pandas as pd
from gsfm import utils

@utils.if_not_exists('data/gene2go.tsv')
def _(file):
  # curl -L https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz | gunzip > data/gene2go.tsv
  import gzip, shutil
  with gzip.open(utils.cached_wget('https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz'), 'rb') as fr:
    with open(file, 'wb') as fw:
      shutil.copyfileobj(fr, fw)

gene2go = pd.read_csv('data/gene2go.tsv', sep='\t')
gene2go['PubMed'] = gene2go['PubMed'].str.split('|')
pd.Series(gene2go['PubMed'].explode().unique()).to_csv('data/pmids.txt', header=False, index=None)

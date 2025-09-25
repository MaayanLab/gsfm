```python
import pandas as pd
gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'r')

gene_gene_similarities.keys()
gene_gene_similarities['geneformer'].index
gene_gene_similarities['scgpt'].index
gene_gene_similarities['geneptsim'].index

human_geneinfo = pd.read_csv('data/Homo_sapiens.gene_info.gz', sep='\t')
mouse_geneinfo = pd.read_csv('data/Mus_musculus.gene_info.gz', sep='\t')
geneinfo = pd.concat([human_geneinfo, mouse_geneinfo]).drop_duplicates(['GeneID'])

def maybe_split(record):
  ''' NCBI Stores Nulls as '-' and lists '|' delimited
  '''
  if record in {'', '-'}:
    return set()
  return set(record.split('|'))
#
def supplement_dbXref_prefix_omitted(ids):
  ''' NCBI Stores external IDS with Foreign:ID while most datasets just use the ID
  '''
  for id in ids:
    # add original id
    yield id
    # also add id *without* prefix
    if ':' in id:
      yield id.split(':', maxsplit=1)[1]
#
geneinfo['All_synonyms'] = [
  set.union(
    maybe_split(row['Symbol']),
    maybe_split(row['Symbol_from_nomenclature_authority']),
    maybe_split(str(row['GeneID'])),
    maybe_split(row['Synonyms']),
    maybe_split(row['Other_designations']),
    maybe_split(row['LocusTag']),
    set(supplement_dbXref_prefix_omitted(maybe_split(row['dbXrefs']))),
  )
  for _, row in geneinfo.iterrows()
]

synonyms, symbols = zip(*{
  (synonym, row['Symbol'])
  for _, row in geneinfo.iterrows()
  for synonym in row['All_synonyms']
})
ncbi_lookup = pd.Series(symbols, index=synonyms)
index_values = ncbi_lookup.index.value_counts()
ambiguous = index_values[index_values > 1].index
ncbi_lookup_disambiguated = ncbi_lookup[(
  (ncbi_lookup.index == ncbi_lookup) | (~ncbi_lookup.index.isin(ambiguous))
)]

gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'r')
geneformer = gene_gene_similarities['geneformer']
gene_gene_similarities.close()

geneformer_index_resolved = geneformer.index.map(ncbi_lookup_disambiguated.get)
geneformer.groupby(ncbi_lookup_disambiguated).mean()
mask = ~pd.isna(geneformer_index_resolved)
geneformer = geneformer.loc[mask, mask]
geneformer.index = geneformer.columns = geneformer_index_resolved[mask]

gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'a')
gene_gene_similarities['geneformer'] = geneformer
gene_gene_similarities.close()


gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'r')
scgpt = gene_gene_similarities['scgpt']
gene_gene_similarities.close()

scgpt_index_resolved = scgpt.index.map(ncbi_lookup_disambiguated.get)
mask = ~pd.isna(scgpt_index_resolved)
scgpt = scgpt.loc[mask, mask]
scgpt.index = scgpt.columns = scgpt_index_resolved[mask]
mask = ~scgpt.index.duplicated()
scgpt = scgpt.loc[mask, mask]

gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'a')
gene_gene_similarities['scgpt'] = scgpt
gene_gene_similarities.close()
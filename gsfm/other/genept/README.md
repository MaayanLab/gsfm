
```python
import pickle
import zipfile

with zipfile.ZipFile(utils.cached_wget('https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1', 'data/GenePT_emebdding_v2.zip'), 'rb') as zip:
  with zip.open('GenePT_gene_protein_embedding_model_3_text.pickle.', 'rb') as fr:
    genept = pd.DataFrame.from_dict(pickle.load(fr)).T

genept = genept[genept.index.isin(all_genes)]

import sklearn.metrics.pairwise
genept_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(genept.values), index=genept.index, columns=genept.index)

gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'a')
gene_gene_similarities['geneptsim'] = genept_gene_gene
gene_gene_similarities.close()
```

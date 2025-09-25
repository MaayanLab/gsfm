# Node2Vec

With [GRAPE](https://github.com/AnacletoLAB/grape) because the graph is too big for node2vec(https://github.com/eliorc/node2vec)

We train node2vec using the rummagene gene gene similarity matrix where edges are made between genes with a similarity > 3 standard deviations from the mean similarity. From the resulting node2vec we extract the embeddings and compute similarity between genes using those embeddings.

## Prepare python environment
```bash
pyenv local 3.8.18
pip install grape
```

## Node2Vec with GRAPE

```python
import pathlib
data_dir = pathlib.Path('data')

import pandas as pd

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'r')
rummagene_gene_gene = gene_gene_similarities['rummagene']
gene_gene_similarities.close()

values = pd.Series(rummagene_gene_gene.values.flatten())
mu, std = values.mean(), values.std()
mu+(3*std)

with open(data_dir/'rummagene_sim_edges_mu_3std.csv', 'w') as fw:
  print('source', 'destination', sep=',', file=fw)
  for a, row in rummagene_gene_gene.iterrows():
    for b in row[row>(mu+(3*std))].index:
      if a == b: continue
      print(a, b, sep=',', file=fw)


from grape import Graph
from grape.embedders import Node2VecGloVeEnsmallen

G = Graph.from_csv(edge_path='rummagene_sim_edges_mu_3std.csv', sources_column='source', destinations_column='destination', directed=False)
embedding = Node2VecGloVeEnsmallen().fit_transform(G)
node2vec_a, node2vec_b = embedding.get_all_node_embedding()
node2vec_a.to_csv(data_dir/'rummagene_sim_node2vec_emb_mu_3std_a.tsv', sep='\t')
node2vec_b.to_csv(data_dir/'rummagene_sim_node2vec_emb_mu_3std_b.tsv', sep='\t')

import sklearn.metrics.pairwise
node2vec_a_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(node2vec_a.values), index=node2vec_a.index, columns=node2vec_a.index)
node2vec_b_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(node2vec_b.values), index=node2vec_b.index, columns=node2vec_b.index)

node2vec_c = pd.read_csv('rummagene_sim_node2vec_emb_0.5.tsv', sep='\t', index_col=0)
node2vec_c_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(node2vec_c.values), index=node2vec_c.index, columns=node2vec_c.index)

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'a')
gene_gene_similarities.keys()
gene_gene_similarities['node2vec_a'] = node2vec_a_gene_gene
gene_gene_similarities['node2vec_b'] = node2vec_b_gene_gene
gene_gene_similarities['node2vec_c'] = node2vec_c_gene_gene
gene_gene_similarities.close()
```

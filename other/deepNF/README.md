## Create edgelist from Rummagene

```python
import pathlib
data_dir = pathlib.Path('data')
import pandas as pd

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'r')
rummagene_gene_gene = gene_gene_similarities['rummagene']
gene_gene_similarities.close()

gene_to_index = { g: i for i, g in enumerate(rummagene_gene_gene.index) }

with open(data_dir/'rummagene_net_edgelist_c.txt', 'w') as fw:
  for a, row in rummagene_gene_gene.iterrows():
    for b, v in row[row>0.5].to_dict().items():
    # for b, v in row[row>(mu+(3*std))].to_dict().items():
      if b == a: continue
      print(gene_to_index[a], gene_to_index[b], v, file=fw, sep='\t')
```

## Run deepNF

From <https://github.com/VGligorijevic/deepNF> with some patches to get it working <https://github.com/VGligorijevic/deepNF/pull/9>

We train deepNF using the rummagene gene gene similarity matrix where edges are made between genes with a similarity > 0.5 similarity. From the resulting deepNF we extract the embeddings and compute similarity between genes using those embeddings.

```bash
pyenv local 3.9
pip install keras tensorflow numpy networkx scikit-learn matplotlib
# patch deepNF.py
python net_embedding.py --model_type ae --nets ../data/rummagene_net_edgelist_c.txt
```

## Load embedding
```python
import pathlib
data_dir = pathlib.Path('data')

import pickle
with open('deepNF/test_results/deepNF_AE_arch_1000-500-1000_features.pckl', 'rb') as fr:
  features = pickle.load(fr)

import h5py
f=h5py.File('deepNF/test_models/deepNF_AE_arch_1000-500-1000.h5', 'r')
f['model_weights']['input_1'].keys()
import pandas as pd

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'r')
rummagene_gene_gene = gene_gene_similarities['rummagene']
gene_gene_similarities.close()

index_to_gene = { i: g for i, g in enumerate(rummagene_gene_gene.index) }

index = {index_to_gene[i] for i in pd.read_csv(data_dir/'rummagene_net_edgelist_c.txt', sep='\t', header=None)[[0, 1]].values.flatten()}

df_features = pd.DataFrame(features,index=index)
df_features.to_csv(data_dir/'deepNF_emb_c.tsv', sep='\t')
# df_features = pd.read_csv('deepNF_emb.tsv', sep='\t', index_col=0)

import pandas as pd
import sklearn.metrics.pairwise
deepNF_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(df_features.values), index=df_features.index, columns=df_features.index)

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'a')
gene_gene_similarities['deepNF_c'] = deepNF_gene_gene
gene_gene_similarities.close()
```

# Geneformer

From: <https://huggingface.co/ctheodoris/Geneformer>

We extract gene embeddings from the pre-trained geneformer model.

```bash
pyenv local 3.10.13
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
ipython
```

```python
import torch
import pandas as pd
from geneformer import EmbExtractor
from geneformer.emb_extractor import get_embs
import geneformer.perturber_utils as pu
import pathlib
data_dir = pathlib.Path('data')
embex = EmbExtractor(model_type="Pretrained", emb_mode="gene")
filtered_input_data = {'input_ids':[list(embex.token_gene_dict.values())]}

model = pu.load_model(
  embex.model_type, embex.num_classes, 'Geneformer-V2-316M', mode="eval"
)
model.bert.embeddings.word_embeddings

emb = model.bert.embeddings.word_embeddings(torch.arange(len(embex.token_gene_dict))[None, :].to('cuda'))
gene_emb = emb.cpu().detach().squeeze().numpy()

df_gene_emb = pd.DataFrame(gene_emb)
df_gene_emb.index = df_gene_emb.index.map(embex.token_gene_dict.get)
df_gene_emb.to_csv(data_dir/'geneformer_embed.tsv', sep='\t')

import sklearn.metrics.pairwise
geneformer_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(df_gene_emb.values), index=df_gene_emb.index, columns=df_gene_emb.index)

from gsfm import utils
ncbi_lookup_disambiguated = utils.get_ncbi_lookup()

geneformer_index_resolved = geneformer_gene_gene.index.map(ncbi_lookup_disambiguated.get)
geneformer_gene_gene.groupby(ncbi_lookup_disambiguated).mean()
mask = ~pd.isna(geneformer_index_resolved)
geneformer_gene_gene = geneformer_gene_gene.loc[mask, mask]
geneformer_gene_gene.index = geneformer_gene_gene.columns = geneformer_index_resolved[mask]

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'a')
gene_gene_similarities['geneformer'] = geneformer_gene_gene
gene_gene_similarities.close()

```

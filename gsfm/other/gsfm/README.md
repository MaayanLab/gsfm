# GSFM Embeddings

Using the GSFM model itself off of hugging face, produce gene embeddings by a couple of methods

```python
from gsfm import Vocab, GSFM
vocab = Vocab.from_pretrained('maayanlab/gsfm-rummagene')
gsfm = GSFM.from_pretrained('maayanlab/gsfm-rummagene')
gsfm.eval()
gsfm.decoder.layers = gsfm.decoder.layers[:1]

import torch
import pandas as pd
gsfm_enc = gsfm(torch.tensor(vocab(vocab.vocab))[:, None]).cpu().detach().numpy()
df_gsfm_enc = pd.DataFrame(gsfm_enc, index=vocab.vocab)
df_gsfm_enc.to_csv('data/gsfm_enc.tsv', sep='\t')

gsfm_emb = gsfm.encode(torch.tensor(vocab(vocab.vocab))[:, None]).cpu().detach().numpy()
df_gsfm_emb = pd.DataFrame(gsfm_emb, index=vocab.vocab)
df_gsfm_emb.to_csv('data/gsfm_emb.tsv', sep='\t')

import sklearn.metrics.pairwise
df_gsfm_enc_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(df_gsfm_enc.values), index=df_gsfm_enc.index, columns=df_gsfm_enc.index)
df_gsfm_emb_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(df_gsfm_emb.values), index=df_gsfm_emb.index, columns=df_gsfm_emb.index)

gene_gene_similarities = pd.HDFStore('data/gene_gene_similarities.h5', 'a')
gene_gene_similarities['gsfm_emb'] = df_gsfm_emb_gene_gene
gene_gene_similarities['gsfm_enc'] = df_gsfm_enc_gene_gene
gene_gene_similarities.close()
```

# scGPT

From: <https://github.com/bowang-lab/scGPT>

Downloaded `best_model.pt`, `vocab.json`, & `args.json` from the whole-human link 2025-09

Here we load the model and extract the pre-trained gene embeddings.

## Prepare python environment
```bash
pyenv local 3.8.18
pip install torch
pip install scgpt "flash-attn<1.0.5"
```

## Extract scGPT Gene Embeddings & Gene-Gene Similarities

```python
import torch
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import json
import pathlib
data_dir = pathlib.Path('data')

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins


model_dir = Path('.')
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Retrieve model parameters from config files
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

gene2idx = vocab.get_stoi()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
)

try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)

import numpy as np
gene_ids = np.array([id for id in gene2idx.values()])
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()

import pandas as pd
gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys())}

df_gene_embeddings = pd.DataFrame(gene_embeddings).T
df_gene_embeddings.to_csv(data_dir/'scGPT_embed.tsv', sep='\t')
# df_gene_embeddings = pd.read_csv('/home/u8sand/Programs/work/experiments/genesetformer/2025-09-04/scGPT/scGPT_embed.tsv', sep='\t', index_col=0)

import sklearn.metrics.pairwise
scgpt_gene_gene = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(df_gene_embeddings.values), index=df_gene_embeddings.index, columns=df_gene_embeddings.index)

from gsfm import utils
ncbi_lookup_disambiguated = utils.get_ncbi_lookup()

scgpt_index_resolved = scgpt_gene_gen.index.map(ncbi_lookup_disambiguated.get)
mask = ~pd.isna(scgpt_index_resolved)
scgpt_gene_gen = scgpt_gene_gen.loc[mask, mask]
scgpt_gene_gen.index = scgpt_gene_gen.columns = scgpt_index_resolved[mask]
mask = ~scgpt_gene_gen.index.duplicated()
scgpt_gene_gen = scgpt_gene_gen.loc[mask, mask]

gene_gene_similarities = pd.HDFStore(data_dir/'gene_gene_similarities.h5', 'a')
gene_gene_similarities['scgpt'] = scgpt_gene_gene
gene_gene_similarities.close()

```

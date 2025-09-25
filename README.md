# GSFM

The gene set foundation model training and benchmarking code.

## Usage

```bash
pip install gsfm@git+https://github.com/MaayanLab/gsfm.git
```

```python
import torch
from gsfm import Vocab, GSFM

# other model variants include:
#  maayanlab/gsfm-rummageo -- trained on rummageo only
#  maayanlab/gsfm-rummage -- trained on both rummagene and rummageo

# load gsfm vocabulary and model weights
vocab = Vocab.from_pretrained('maayanlab/gsfm-rummagene')
gsfm = GSFM.from_pretrained('maayanlab/gsfm-rummagene')
gsfm.eval()

# convert gene symbols into token ids
token_ids = torch.tensor(vocab(['ACE1', 'ACE2']))[None, :]

# use model to predict missing genes from the set
logits = torch.squeeze(gsfm(token_ids))
top_10 = sorted(zip(logits, vocab.vocab))[-10:]
top_10

# get model middle layer
gene_set_encoding = gsfm.encode(token_ids)
gene_set_encoding
```

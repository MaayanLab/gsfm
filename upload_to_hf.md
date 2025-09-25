# Code used to upload relevant trained models to huggingface

```python
import torch
import pathlib
from gsfm import utils

for ckpt in pathlib.Path('lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt'):
  config = utils.model_config_from_ckpt(ckpt)
  model, vocab = utils.model_tokenizer_from_ckpt(ckpt, config=config, map_location=torch.device('cuda'))
  model.push_to_hub(f"maayanlab/gsfm-{config['data']['class_path'].split('.')[-2]}")
  vocab.push_to_hub(f"maayanlab/gsfm-{config['data']['class_path'].split('.')[-2]}")
```

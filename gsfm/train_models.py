#%%
import itertools
params = [
  [('python -m gsfm.main fit',
    '--data.batch_size=128 --trainer.max_epochs=50')],
  [
    ('--data=rummagene_seq2multihot.RummageneDataModule',
     '--model.vocab_size=19589'),
    ('--data=rummageo_seq2multihot.RummaGEODataModule',
     '--model.vocab_size=89890'),
    ('--data=rummage_seq2multihot.RummageDataModule',
     '--model.vocab_size=106972'),
  ],
  [
    ('--model=gsfm.models.DAE.GSFM',
     ''),
    ('--model=gsfm.models.EDAE.GSFM',
     ''),
    ('--model=gsfm.models.EPDAE.GSFM',
     ''),
    ('--model=gsfm.models.mEDAE.GSFM',
     ''),
    ('--model=gsfm.models.VAE.GSFM',
     ''),
  ],
  [
    ('--model.partition=0',
     ''),
    ('--model.partition=0.2',
     ''),
    ('--model.partition=0.5',
     ''),
    ('--model.partition=0.8',
     ''),
  ],
  [
    ('--model.dropout=0.1',
     ''),
    ('--model.dropout=0.2',
     ''),
    ('--model.dropout=0.4',
     ''),
  ],
  [
    ('--model.depth=1',
     ''),
    ('--model.depth=2',
     ''),
    ('--model.depth=3',
     ''),
  ],
  [
    ('--model.d_model=128',
     ''),
    ('--model.d_model=256',
     ''),
    ('--model.d_model=512',
     ''),
    ('--model.d_model=1024',
     ''),
  ],
  [
    ('--model.weighted_loss=none',
     ''),
    ('--model.weighted_loss=balance',
     ''),
    ('--model.weighted_loss=unseen',
     ''),
  ],
]
param_product = []
for P in itertools.product(*params):
  starts, ends = zip(*P)
  param_product.append(' '.join(itertools.chain(starts, ends)))

#%%
print('\n'.join(param_product))

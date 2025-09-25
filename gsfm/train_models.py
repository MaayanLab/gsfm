#%%
import itertools
params = [
  [('cd 2025-09-08;', '')],
  [('../.venv/bin/python main.py fit',
    '--data.batch_size=128 --trainer.max_epochs=50')],
  [
    # ('--data=rummagene_seq2multihot.RummageneDataModule',
    #  '--model.vocab_size=19589'),
    # ('--data=rummageo_seq2multihot.RummaGEODataModule',
    #  '--model.vocab_size=89890'),
    # ('--data=rummage_seq2multihot.RummageDataModule',
    #  '--model.vocab_size=106972'),
    ('--data=rummage_human_seq2multihot.RummageDataModule',
     '--model.vocab_size=57971'),
    ('--data=rummageo_human_seq2multihot.RummaGEODataModule',
     '--model.vocab_size=57539'),
  ],
  [
  #   ('--data.noise_size=0.25',
  #    ''),
  #   ('--data.noise_size=0.5',
  #    ''),
  #   ('--data.noise_size=0.75',
  #    ''),
  #   ('--data.noise_size=1.0',
  #    ''),
    # ('--data.noise_size=2.0',
    #  ''),
    # ('--data.noise_size=4.0',
    #  ''),
    # ('--data.noise_size=8.0',
    #  ''),
    # ('--data.noise_size=16.0',
    #  ''),
    # ('--data.max_size=500',
    #  ''),
    # ('--data.max_size=200',
    #  ''),
    # ('--data.max_size=100',
    #  ''),
    # ('--data.max_size=15',
    #  ''),
    ('--data.data_size=0.25',
     ''),
    ('--data.data_size=0.5',
     ''),
    ('--data.data_size=0.75',
     ''),
  ],
  [
    ('--model=J.GeneSetMaskedEmbeddingAutoEncoder --model.partition=0 --model.dropout=0.2 --model.depth=2 --model.d_model=256 --model.weighted_loss=none',
     ''),
  ],
]
param_product = []
for P in itertools.product(*params):
  starts, ends = zip(*P)
  param_product.append(' '.join(itertools.chain(starts, ends)))

#%%
print('\n'.join(param_product))

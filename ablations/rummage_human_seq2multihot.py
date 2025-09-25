#%%
import torch
import lightning as L
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class RummageDataModule(L.LightningDataModule):
  def __init__(self, batch_size: int = 64):
    super().__init__()
    self.batch_size = batch_size

  def setup(self, stage: str):
    if getattr(self, 'rummage_train', None) is not None: return
    if stage == 'fit':
      self.rummage_train = [
        clean_genes
        for it in [
          utils.read_gmt('data/2025-08-29-rummageo-human.gmt'),
          utils.read_gmt('data/2025-08-29-rummagene.gmt'),
        ]
        for _, _, genes in it
        for clean_genes in (list(set(filter(None, genes))),)
        if len(clean_genes) > 10 and len(clean_genes) < 1000
      ]
    self.tokenizer = lambda gene_set: gene_set.split('\t')
    @utils.if_not_exists('data/2025-08-29-rummage-human-vocab.pickle')
    def _(filename):
      vocab = utils.Vocab.build_vocab_from_iterator((
        gene_set
        for gene_set in self.rummage_train
      ), min_freq=5, specials=special_symbols, special_first=True)
      vocab.set_default_index(UNK_IDX)
      vocab.save(filename)

    self.vocab = utils.Vocab.from_file('data/2025-08-29-rummage-human-vocab.pickle')
    self.vocab_size = len(self.vocab)
    self.text_transform = lambda text: torch.tensor(self.vocab(self.tokenizer(text)))

  def collate_fn(self, raw_batch: list[list[str]]):
    return torch.nn.utils.rnn.pad_sequence([
      torch.tensor(self.vocab(gene_set), dtype=torch.int64)
      for gene_set in raw_batch
    ], padding_value=PAD_IDX, batch_first=True)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.rummage_train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

  # def val_dataloader(self):
  #   return torch.utils.data.DataLoader(self.rummage_val, batch_size=256, collate_fn=RummageDataModule._collate_fn, shuffle=True)

#%%
# M=RummageDataModule()
# M.setup('fit')
# M.vocab_size

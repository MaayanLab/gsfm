#%%
import torch
import lightning as L
import random
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class RummageneDataModule(L.LightningDataModule):
  def __init__(self, batch_size: int = 64, data_size: float = 1.0):
    super().__init__()
    self.batch_size = batch_size
    self.data_size = data_size

  def setup(self, stage: str):
    if stage == 'fit':
      if getattr(self, 'rummagene_train', None) is None:
        self.rummagene_train = [
          clean_genes
          for _, year, genes in utils.read_gmt(f'data/2025-08-29-rummagene.gmt')
          for clean_genes in (list(set(map(str.upper, filter(None, genes)))),)
          if len(clean_genes) > 10 and len(clean_genes) < 1000
        ]
        random.shuffle(self.rummagene_train)
        self.rummagene_train = self.rummagene_train[:int(self.data_size*len(self.rummagene_train))]
    self.tokenizer = lambda gene_set: gene_set.split('\t')
    @utils.if_not_exists('data/2025-08-29-rummagene-vocab.pickle')
    def _(filename):
      vocab = utils.Vocab.build_vocab_from_iterator((
        gene_set
        for gene_set in self.rummagene_train
      ), min_freq=5, specials=special_symbols, special_first=True)
      vocab.set_default_index(UNK_IDX)
      vocab.save(filename)

    self.vocab = utils.Vocab.from_file('data/2025-08-29-rummagene-vocab.pickle')
    self.vocab_size = len(self.vocab)
    self.text_transform = lambda text: torch.tensor(self.vocab(self.tokenizer(text)))

  def collate_fn(self, raw_batch: list[list[str]]):
    return torch.nn.utils.rnn.pad_sequence([
      torch.tensor(self.vocab(gene_set), dtype=torch.int64)
      for gene_set in raw_batch
    ], padding_value=PAD_IDX, batch_first=True)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.rummagene_train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

#%%
# M=RummageneDataModule()
# M.setup('fit')
# M.vocab_size

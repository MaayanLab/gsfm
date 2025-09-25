import torch
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class GSFM(utils.LightningModuleEx):
  def __init__(self, vocab_size, d_model=256, depth=2, dropout=0.2, partition=0.8, weighted_loss='none'):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.depth = depth
    self.dropout = dropout
    self.partition = partition
    self.weighted_loss = weighted_loss
    self.encoder = utils.MLP(vocab_size, *[d_model*(2**(n-1)) for n in range(depth, 1, -1)], d_model, dropout=dropout)
    self.decoder = utils.MLP(d_model, *[d_model*(2**(n-1)) for n in range(1, depth)], vocab_size, dropout=dropout)
    if weighted_loss == 'balance':
      self.bias = torch.nn.Parameter(torch.ones(vocab_size), requires_grad=False)
    self.save_hyperparameters()
  def encode(self, x):
    x = utils.multihot_tensor(x, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
    x[:, PAD_IDX] = 0
    return self.encoder(x)
  def forward(self, x):
    x = self.encode(x)
    x = self.decoder(x)
    return x
  def step(self, batch, batch_idx):
    if self.partition == 0:
      x_idx = y_idx = batch
    else:
      x_idx, y_idx = utils.partition_padded_tensor(batch, self.partition, PAD_IDX, device=self.device)
    y_ = self(x_idx)
    y = utils.multihot_tensor(y_idx, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
    y[:, PAD_IDX] = 0
    if self.partition != 0 and self.weighted_loss == 'unseen':
      pos_weight = 1 - utils.multihot_tensor(x_idx, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
      pos_weight[:, PAD_IDX] = 0
      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif self.weighted_loss == 'balance':
      # we count the number of times we've seen each gene
      #   genes we've seen less frequently will have a higher weight
      self.bias += y.sum(0).long()
      self.bias[PAD_IDX] += y.shape[0]
      pos_weight = ((self.bias[PAD_IDX]-self.bias) / self.bias)[None, :]
      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif self.weighted_loss == 'none':
      criterion = torch.nn.BCEWithLogitsLoss()
    else: raise NotImplementedError()
    loss = criterion(y_, y)
    return loss
  def training_step(self, batch, batch_idx):
    loss = self.step(batch, batch_idx)
    self.log('train_loss', loss)
    return loss
  # def validation_step(self, batch, batch_idx):
  #   loss = self.step(batch, batch_idx)
  #   self.log('val_loss', loss)
  #   return loss
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25)
    return [optimizer], [{
        "scheduler": scheduler,
        "monitor": "train_loss",
        "frequency": 1,
    }]

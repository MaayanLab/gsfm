import torch
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class GSFM(utils.LightningModuleEx):
  def __init__(self, vocab_size, d_model=256, depth=2, dropout=0.2, partition=0.8, weighted_loss=None):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.depth = depth
    self.dropout = dropout
    self.partition = partition
    self.weighted_loss = weighted_loss
    self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
    self.encoder = utils.MLP(d_model, *[d_model*(2**(n-1)) for n in range(depth-1, 1, -1)], d_model, dropout=dropout)
    self.decoder = utils.MLP(d_model*2, *[d_model*(2**n) for n in range(1, depth)], vocab_size, dropout=dropout)
    self.save_hyperparameters()
  def encode(self, x):
    mask = (x != PAD_IDX)[:, :, None]
    x = emb = self.embedding(x)
    emb_mean = (emb*mask).sum(1) / mask.sum(1)
    x = enc = self.encoder(emb)
    enc_mean = (enc*mask).sum(1) / mask.sum(1)
    x = torch.cat([enc_mean, emb_mean], -1)
    return x
  def forward(self, x):
    x = self.encode(x)
    x = self.decoder(x)
    return x
  def training_step(self, batch, batch_idx):
    if self.partition == 0:
      x_idx = y_idx = batch
    else:
      x_idx, y_idx = utils.partition_padded_tensor(batch, self.partition, PAD_IDX, device=self.device)
    y = utils.multihot_tensor(y_idx, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
    y[:, PAD_IDX] = 0
    y_ = self(x_idx)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_, y)
    self.log('loss', loss, prog_bar=True)
    return loss
  def validation_step(self, batch, batch_idx):
    return self.training_step(batch, batch_idx)
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25)
    return [optimizer], [{
        "scheduler": scheduler,
        "monitor": "loss",
        "frequency": 1,
    }]

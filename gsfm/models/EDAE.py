import torch
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class GSFM(utils.LightningModuleEx):
  def __init__(self, vocab_size, d_model=256, depth=2):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
    self.encoder = utils.MLP(*[d_model**n for n in range(1, depth)], d_model)
    self.decoder = utils.MLP(d_model, *[d_model**n for n in range(2, depth)], vocab_size)
    self.save_hyperparameters()
  def encode(self, x):
    mask = (x != PAD_IDX)[:, :, None]
    emb = self.embedding(x)
    enc = self.encoder(emb)
    enc_mean = (enc*mask).sum(1) / mask.sum(1)
    return enc_mean
  def forward(self, x):
    x = self.encode(x)
    x = self.decoder(x)
    return x
  def training_step(self, batch, batch_idx):
    x, y = batch
    is_x = y < 0
    y = torch.where(is_x, 0, y).float()
    pos_weight = torch.where(is_x, 0, 1)
    y_ = self(x)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

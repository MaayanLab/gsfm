import torch
from gsfm import utils

UNK_IDX, PAD_IDX = 0, 1
special_symbols = ['<unk>', '<pad>']

class GSFM(utils.LightningModuleEx):
  def __init__(self, vocab_size, d_model=256, depth=2, dropout=0.2, latent_dim=128):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.depth = depth
    self.dropout = dropout
    self.latent_dim = latent_dim
    self.encoder = utils.MLP(vocab_size, *[d_model*(2**(n-1)) for n in range(depth, 1, -1)], d_model, dropout=dropout)
    self.hidden2mu = torch.nn.Linear(d_model, latent_dim)
    self.hidden2log_var = torch.nn.Linear(d_model, latent_dim)
    self.decoder = utils.MLP(latent_dim, d_model, *[d_model*(2**(n-1)) for n in range(1, depth)], vocab_size, dropout=dropout)
    self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))
    self.save_hyperparameters()

  def encode(self, x):
    x = utils.multihot_tensor(x, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
    x[:, PAD_IDX] = 0
    hidden = self.encoder(x)
    mu = self.hidden2mu(hidden)
    log_var = self.hidden2log_var(hidden)
    return mu, log_var

  def _forward(self, x):
    mu, log_var = self.encode(x)
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    x_ = self.decoder(z)
    return mu, std, z, x_

  def forward(self, x):
    _mu, _std, _z, x_ = self._forward(x)
    return x_

  def reparametrize(self, mu, log_var):
    sigma = torch.exp(0.5*log_var)
    z = torch.randn_like(sigma)
    return mu + sigma*z

  def kl_divergence(self, z, mu, std):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl

  def training_step(self, batch, batch_idx):
    mu, std, z, x_ = self._forward(batch)
    x = utils.multihot_tensor(batch, num_classes=self.vocab_size, device=self.device, dtype=torch.float)
    x[:, PAD_IDX] = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    bce = criterion(x_, x)
    kl = self.kl_divergence(z, mu, std)
    loss = (kl + bce).mean()
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

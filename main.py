from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
  seed_everything(42, workers=True)
  cli = LightningCLI(
    trainer_defaults=dict(
      accelerator='auto',
      devices='auto',
      strategy='auto',
      max_epochs=50,
      callbacks=[ModelCheckpoint(save_top_k=-1, every_n_epochs=5)],
      deterministic=True,
    ),
  )

if __name__ == '__main__':
  main()

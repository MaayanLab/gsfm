#%%
import pandas as pd
import numpy as np
import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from gsfm import utils

#%%
batch_size = 128

def enumerate_x_ys(*, library, vocab):
  for term, _, genes in utils.read_gmt(f'data/{library}.txt'):
    for i in range(5):
      np.random.shuffle(genes)
      x, y = utils.partition(genes, 0.5)
      x_true = utils.multi_hot(torch.tensor(vocab(x)), num_classes=len(vocab))
      y_true = utils.multi_hot(torch.tensor(vocab(y)), num_classes=len(vocab)).numpy()
      if x_true.sum() < 5 or y_true.sum() < 5: continue
      yield term, torch.tensor(vocab(x)), x_true, y_true

def model_benchmark_from_ckpt(ckpt, library):
  config = utils.model_config_from_ckpt(ckpt)
  @utils.if_not_exists(f"data/classic-benchmark/2025-09-02/{library}/model/{pathlib.Path(ckpt).parent.parent.name}/scores.tsv")
  def _(file):
    model, vocab = utils.model_tokenizer_from_ckpt(ckpt, config=config, map_location=torch.device('cuda'))
    scores = []
    library_background = utils.read_gmt_background(f'data/{library}.txt')
    background = utils.multi_hot(torch.tensor(vocab(library_background)), num_classes=len(vocab)).numpy().astype(bool)
    for rows in tqdm(utils.chunked(enumerate_x_ys(library=library, vocab=vocab), batch_size)):
      batch_term, batch_x, batch_x_true, batch_y_true = zip(*rows)
      batch_y_score = torch.sigmoid(model(torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(model.device))).detach().cpu().numpy()
      for i, (term, x_true, y_true) in tqdm(enumerate(zip(batch_term, batch_x_true, batch_y_true))):
        y_score = batch_y_score[i, :]
        # ignore genes not in the background
        y_true = y_true[background & ~(x_true.bool().numpy())]
        y_score = y_score[background & ~(x_true.bool().numpy())]
        t = 0.5
        scores.append(dict(
          name=library,
          model_base=pathlib.Path(ckpt).parent.parent.parent.parent.name,
          model_name=f"{config['model']['class_path']}({pathlib.Path(ckpt).parent.parent.name})",
          data_name=config['data']['class_path'],
          term=term,
          threshold=t,
          roc_auc=roc_auc_score(y_true, y_score),
          ap=average_precision_score(y_true, y_score),
          recall=recall_score(y_true, y_score>=t),
          f1=f1_score(y_true, y_score>=t),
          es=utils.ES(y_true, y_score),
          n=y_true.shape[0],
          n_true=y_true.sum(),
          n_pred=(y_score>=t).sum(),
          n_correct=(y_score[y_true.astype(bool)]>=t).sum(),
        ))
    pd.DataFrame(scores).to_csv(file, sep='\t', index=None)

#%%
if __name__ == '__main__':
  libraries = [
    'GWAS_Catalog_2023',
    'GO_Biological_Process_2023',
    'ChEA_2022',
    'KEGG_2021_Human',
  ]
  with ThreadPoolExecutor(max_workers=6) as pool:
    for library in tqdm(libraries):
      for ckpt in tqdm(pathlib.Path('../2025-05-27/lightning_logs').absolute().glob('*/checkpoints/epoch=49-*.ckpt')):
        utils.mp_run(model_benchmark_from_ckpt, ckpt, library)

      for ckpt in tqdm(pathlib.Path('lightning_logs').glob('*/checkpoints/epoch=49-*.ckpt')):
        pool.submit(utils.mp_run, model_benchmark_from_ckpt, ckpt, library)

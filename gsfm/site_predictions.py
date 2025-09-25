import torch
import numpy as np
import pandas as pd
import pathlib
import joblib
from gsfm import utils, GSFM, Vocab
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from tqdm.auto import tqdm

batch_size = 256
jobs = 6

def enumerate_xs(*, library, vocab):
  for term, _, genes in utils.read_gmt(library):
    x_true = utils.multi_hot(torch.tensor(vocab(genes)), num_classes=len(vocab))
    if x_true.sum() < 5: continue
    yield term, torch.tensor(vocab(genes))

def enumerate_x_ys(*, library, vocab):
  for term, _, genes in utils.read_gmt(library):
    if len(genes) < 5: continue
    for i in range(5):
      np.random.shuffle(genes)
      x, y = utils.partition(genes, 0.5)
      x_true = utils.multi_hot(torch.tensor(vocab(x)), num_classes=len(vocab))
      y_true = utils.multi_hot(torch.tensor(vocab(y)), num_classes=len(vocab)).numpy()
      if x_true.sum() < 5 or y_true.sum() < 5: continue
      yield term, torch.tensor(vocab(x)), x_true, y_true

gmts = [
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/LINCS/XMT/2022-12-13/LINCS_XMT_2022-12-13_LINCS_L1000_CRISPR_KO_Consensus_Sigs.gmt', 'data/LINCS_L1000_CRISPR_KO_Consensus_Sigs.gmt'),
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/LINCS/XMT/2022-12-13/LINCS_XMT_2022-12-13_LINCS_L1000_Chem_Pert_Consensus_Sigs.gmt', 'data/LINCS_L1000_Chem_Pert_Consensus_Sigs.gmt'),
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/IDG/XMT/2022-12-13/IDG_XMT_2022-12-13_IDG_Drug_Targets_2022.gmt', 'data/IDG_Drug_Targets_2022.gmt'),
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/MoTrPAC/XMT/2024-03-05/MoTrPAC_Endurance_Trained_Rats_2023.gmt', 'data/MoTrPAC_Endurance_Trained_Rats_2023.gmt'),
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/KOMP2/XMT/2022-12-13/KOMP2_XMT_2022-12-13_KOMP2_Mouse_Phenotypes_2022.gmt', 'data/KOMP2_Mouse_Phenotypes_2022.gmt'),
  utils.cached_wget('https://cfde-drc.s3.amazonaws.com/LINCS/XMT/2024-09-14/LINCS_L1000_Consensus_Median_Signatures.gmt', 'data/LINCS_L1000_Consensus_Median_Signatures.gmt'),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=OMIM_Disease', f"data/OMIM_Disease.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MGI_Mammalian_Phenotype_Level_4_2024', f"data/MGI_Mammalian_Phenotype_Level_4_2024.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human', f"data/KEGG_2021_Human.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEA_2015', f"data/KEA_2015.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Human_Phenotype_Ontology', f"data/Human_Phenotype_Ontology.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GWAS_Catalog_2025', f"data/GWAS_Catalog_2025.gmt"),
  utils.cached_wget('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ChEA_2022', f"data/ChEA_2022.gmt"),
  f"data/HuBMAP_Azimuth.gmt",
  f"data/HuBMAP_ASCTpB.gmt",
  f"data/GO_Component.gmt",
  f"data/GO_Function.gmt",
  f"data/GO_Process.gmt",
]

def do_predictions(repo_id, library):
  @utils.if_not_exists(f"data/site-predictions/{pathlib.Path(library).name}/{repo_id}/preds.tsv")
  def _(filename):
    vocab = Vocab.from_pretrained(repo_id)
    model = GSFM.from_pretrained(repo_id)
    model.eval()
    y_scores = {}
    for rows in utils.chunked(enumerate_xs(library=library, vocab=vocab), batch_size):
      batch_term, batch_x = zip(*rows)
      batch_y_score = model(torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(model.device)).detach().cpu().numpy()
      for i, term in enumerate(batch_term):
        y_scores[term] = batch_y_score[i, :]
    pd.DataFrame(y_scores, index=vocab.vocab).to_csv(filename, sep='\t', float_format="%.5f")
  
  @utils.if_not_exists(f"data/site-predictions/{pathlib.Path(library).name}/{repo_id}/eval.tsv")
  def _(filename):
    vocab = Vocab.from_pretrained(repo_id)
    model = GSFM.from_pretrained(repo_id)
    model.eval()
    scores = []
    np.random.seed(42)
    library_background = utils.read_gmt_background(library)
    background = utils.multi_hot(torch.tensor(vocab(library_background)), num_classes=len(vocab)).numpy().astype(bool)
    for rows in utils.chunked(enumerate_x_ys(library=library, vocab=vocab), batch_size):
      batch_term, batch_x, batch_x_true, batch_y_true = zip(*rows)
      batch_y_score = torch.sigmoid(model(torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(model.device))).detach().cpu().numpy()
      for i, (term, x_true, y_true) in tqdm(enumerate(zip(batch_term, batch_x_true, batch_y_true))):
        y_score = batch_y_score[i, :]
        # ignore genes not in the background
        y_true = y_true[background & ~(x_true.bool().numpy())]
        y_score = y_score[background & ~(x_true.bool().numpy())]
        t = 0.5
        scores.append(dict(
          name=repo_id,
          library=library,
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
    pd.DataFrame(scores).to_csv(filename, sep='\t', index=False)

#%%
if __name__ == '__main__':
  for _ in tqdm(joblib.Parallel(jobs, return_as='generator_unordered')([
    joblib.delayed(do_predictions)(repo_id, library)
    for repo_id in ['maayanlab/gsfm-rummagene','maayanlab/gsfm-rummage','maayanlab/gsfm-rummageo']
    for library in gmts
  ])): pass

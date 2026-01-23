#%%
import pathlib
import pandas as pd
import seaborn as sns

df = pd.concat([
  pd.read_csv(f, sep='\t')
  for f in pathlib.Path('data/classic-benchmark/2025-09-02').rglob('scores.tsv')
])
df['model_name_with_data'] = df.apply(lambda row: f"{row['model_name'][:-1]}, {row['data_name']})", axis=1)

#%%
import yaml
data = {}
for d in ['2025-05-27', '2025-09-02']:
  for f in pathlib.Path(f"genesetformer/{d}/lightning_logs").expanduser().glob('*/config.yaml'):
    with f.open('r') as fr:
      config = yaml.safe_load(fr)
    data[f.parent.name] = dict(
      data_class_path=config['data']['class_path'],
      data_size=config['data']['init_args'].get('data_size', 1),
      data_max_size=config['data']['init_args'].get('max_size', 1000),
      data_noise_size=config['data']['init_args'].get('noise_size', 0),
    )
#
df_versions = pd.DataFrame(data).T
import re
def _(m):
  m = re.search(r'\((version_\d+)\)', m)
  if m: return m.group(1)
  else: return None
df['model_version'] = df['model_name'].apply(_)
df_versions.index.name = 'model_version'
df_ = df.merge(df_versions, on='model_version', how='inner')
df_
# %%
df_[(
  (df_['data_class_path']=='rummagene_setnoise.RummageneDataModule')
  |(df_['data_class_path']=='rummagene_seq2multihot.RummageneDataModule')
)].groupby(['name', 'data_noise_size'])['roc_auc'].median().unstack().to_csv('data/data-noise-size.csv', float_format='%.2g')


# %%
df_[(
  (df_['data_class_path']=='rummagene_setsize.RummageneDataModule')
  |(df_['data_class_path']=='rummagene_seq2multihot.RummageneDataModule')
  # |(df_['data_class_path']=='rummage_seq2multihot.RummageDataModule')
)].groupby(['name', 'data_max_size'])['roc_auc'].median().unstack().to_csv('data/data-max-size.csv', float_format='%.2g')

# %%
df_[(
  (df_['data_class_path']=='rummagene_datasize.RummageneDataModule')
  |(df_['data_class_path']=='rummagene_seq2multihot.RummageneDataModule')
  # |(df_['data_class_path']=='rummage_seq2multihot.RummageDataModule')
)].groupby(['name', 'data_size'])['roc_auc'].median().unstack().to_csv('data/data-size.csv', float_format='%.2g')

# %%
df_[(
  ((df_['data_class_path']=='rummagene_no_mito.RummageneDataModule')
  |(df_['data_class_path']=='rummagene_seq2multihot.RummageneDataModule')
  # |(df_['data_class_path']=='rummage_seq2multihot.RummageDataModule')
  )
  &(df_['term'].str.lower().str.contains('mitochondria'))
)].groupby(['name', 'data_class_path'])['roc_auc'].median().unstack().to_csv('data/mitochondrial.csv', float_format='%.2g')


#%%

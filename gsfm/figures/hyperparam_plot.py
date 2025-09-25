#%%
import pandas as pd
import plotly.express as px
import itertools

#%%
vals = {
  # 'dataset': ['Rummagene', 'RummaGEO', 'RummaGEO/Gene'],
  'type': ['DAE', 'VAE', 'mEDAE', 'EDAE', 'EPDAE'],
  'dropout': [0.1, 0.2, 0.4],
  'partition': [0.0, 0.2, 0.5, 0.8],
  'depth': [1, 2, 3],
  # 'n_layers': [1, 2, 3],
  'd_model': [256, 1024, 128, 512],
  'weighted_loss': {'balance', 'unseen', 'none'},
  'epochs': list(range(0, 51, 10)),
}
chosen_vals = {
  # 'dataset': 'Rummagene',
  'type': 'DAE',
  'dropout': 0.2,
  'partition': 0.0,
  # 'n_layers': 2,
  'depth': 2,
  'd_model': 256,
  'weighted_loss': 'none',
  'epochs': 50
}
#%%
df_chosen = pd.DataFrame([chosen_vals])

# If you want to plot the entire Cartesian product (careful, it’s huge):
# product = list(itertools.product(*vals.values()))
# df_all = pd.DataFrame(product, columns=vals.keys())

# Instead, we’ll just use unique values per parameter to show axes ranges
# Plot chosen line explicitly
df_plot = df_chosen.copy()
df_plot["is_chosen"] = "chosen"

# Add random sample of possible candidates (optional)
sampled = []
for j in range(50):  # 20 random combinations for context
    sampled.append({k: list(v)[j % len(v)] for i, (k,v) in enumerate(vals.items())})
df_sample = pd.DataFrame(sampled)
df_sample["is_chosen"] = "candidate"

# Combine
df_final = pd.concat([df_sample, df_plot], ignore_index=True)
df_final["color_val"] = df_final["is_chosen"].apply(lambda x: 1 if x=="chosen" else 0)

#%%
fig = px.parallel_categories(
    df_final,
    dimensions=list(vals.keys()),
    color="color_val",
    color_continuous_scale=[[0, "lightgray"], [1, "red"]],
    range_color=[0, 1]
)
fig.update_coloraxes(showscale=False)
fig.write_image('fig-3.pdf')
fig.show()

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
with open('SiC_lengths.npy', 'rb') as f:
    SiC_lengths = np.load(f)

with open('SiC_att_0.npy', 'rb') as f:
    SiC_att_0 = np.load(f)

# %%
plt.scatter(SiC_lengths, SiC_att_0, s=1, marker='.')

# %%

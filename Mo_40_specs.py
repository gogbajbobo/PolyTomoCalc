# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from _handler_funcs import extract_spectrum_data

# %%
colibr_path = '/Users/grimax/Documents/Science/xtomo/spectroTomo/Mo source spectra/Mo_BSV-45_spectra 2023/colibr.txt'
spec_df = pd.read_csv(colibr_path, sep='\s+', header=None, names=['channel', 'count', 'keV'])[['keV']]

specs_path = '/Users/grimax/Documents/Science/xtomo/spectroTomo/Mo source spectra/Mo_BSV-45_spectra 2023/no_filter/'
spec_df['nf_40x5'] = extract_spectrum_data(f'{specs_path}Mo_BSV-45_40x5_g20_20sec.mca')
spec_df['nf_40x10'] = extract_spectrum_data(f'{specs_path}Mo_BSV-45_40x10_g20_20sec.mca')
spec_df['nf_40x15'] = extract_spectrum_data(f'{specs_path}Mo_BSV-45_40x15_g20_20sec.mca')
spec_df['nf_40x20'] = extract_spectrum_data(f'{specs_path}Mo_BSV-45_40x20_g20_20sec.mca')

sigma = 1

spec_df['nf_gauss_5'] = sp.ndimage.gaussian_filter(spec_df['nf_40x5'], sigma=sigma)
spec_df['nf_gauss_10'] = sp.ndimage.gaussian_filter(spec_df['nf_40x10'], sigma=sigma)
spec_df['nf_gauss_15'] = sp.ndimage.gaussian_filter(spec_df['nf_40x15'], sigma=sigma)
spec_df['nf_gauss_20'] = sp.ndimage.gaussian_filter(spec_df['nf_40x20'], sigma=sigma)

# spec_df = spec_df[27:566].reset_index().drop(labels='index', axis=1)
spec_df = spec_df[65:].reset_index().drop(labels='index', axis=1)
spec_df['nf_gauss_5'] /= spec_df['nf_gauss_5'].sum()
spec_df['nf_gauss_10'] /= spec_df['nf_gauss_10'].sum()
spec_df['nf_gauss_15'] /= spec_df['nf_gauss_15'].sum()
spec_df['nf_gauss_20'] /= spec_df['nf_gauss_20'].sum()

spec_df

# %%
spec_df[150:250].plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], logy=True)

# %%

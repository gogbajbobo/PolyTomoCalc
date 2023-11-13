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
# spec_df['nf_gauss_5'] /= spec_df['nf_gauss_5'].sum()
# spec_df['nf_gauss_10'] /= spec_df['nf_gauss_10'].sum()
# spec_df['nf_gauss_15'] /= spec_df['nf_gauss_15'].sum()
# spec_df['nf_gauss_20'] /= spec_df['nf_gauss_20'].sum()

spec_df

# %%
spec_df.plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], logy=True)

# %%
plt.plot(spec_df['keV'], spec_df['nf_gauss_5'], label='5 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_10'], label='10 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_15'], label='15 mA')
plt.ylim(0, 5000)
# plt.xlim(0, 50)
# plt.yscale('log')
plt.xlabel('Energy, keV')
plt.ylabel('Photons, counts')
plt.legend()
plt.grid()
plt.show()

# %%
plt.plot(spec_df['keV'], spec_df['nf_gauss_5'], label='5 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_10'], label='10 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_15'], label='15 mA')
# plt.ylim(1e-5, 0.1)
# plt.xlim(0, 50)
plt.yscale('log')
plt.xlabel('Energy, keV')
plt.ylabel('Photons, counts')
plt.legend()
plt.grid()
plt.show()

# %%
plt.plot(spec_df['keV'], spec_df['nf_gauss_5']/spec_df['nf_gauss_5'].sum(), label='5 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_10']/spec_df['nf_gauss_10'].sum(), label='10 mA')
plt.plot(spec_df['keV'], spec_df['nf_gauss_15']/spec_df['nf_gauss_15'].sum(), label='15 mA')
# plt.ylim(1e-5, 0.1)
# plt.xlim(0, 50)
plt.yscale('log')
plt.xlabel('Energy, keV')
plt.ylabel('Intensity, a.u.')
plt.legend()
plt.grid()
plt.show()

# %%
plt.plot(spec_df[150:250]['keV'], spec_df[150:250]['nf_gauss_5'], label='5 mA')
plt.plot(spec_df[150:250]['keV'], spec_df[150:250]['nf_gauss_10'], label='10 mA')
plt.plot(spec_df[150:250]['keV'], spec_df[150:250]['nf_gauss_15'], label='15 mA')
# plt.ylim(1e-5, 0.1)
# plt.xlim(0, 50)
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# %%
spec_df[150:250].plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], logy=True)

# %%
spec_df[150:250].plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], grid=True)

# %%
spec_df[180:184].plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], grid=True)

# %%
spec_df[180:184][['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15']]

# %%
spec_df[180:184][['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15']].max()

# %%
spec_df[210:215].plot('keV', ['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15'], grid=True)

# %%
spec_df[210:215][['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15']]

# %%
spec_df[210:215][['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15']].max()

# %%
spec_df[180:184][['nf_gauss_5', 'nf_gauss_10', 'nf_gauss_15']].max()

# %%
0.047504/0.009284

# %%
0.036493/0.007822

# %%
0.027468/0.006289

# %%

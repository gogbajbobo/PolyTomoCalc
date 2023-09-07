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
import xraydb
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart

# %%
spectrum1 = pd.read_csv('/Users/grimax/Desktop/tmp/SiC_samples/Mo_source_spectra.csv', sep=';', decimal=',')
# spectrum1 = spectrum1[27:896].reset_index().drop(labels=['index', 'Unnamed: 0'], axis=1)
spectrum1 = spectrum1.reset_index().drop(labels=['index', 'Unnamed: 0'], axis=1)

sigma = 1

spectrum1['nf_gauss'] = sp.ndimage.gaussian_filter(spectrum1['no filter'], sigma=sigma)
spectrum1['1,8_gauss'] = sp.ndimage.gaussian_filter(spectrum1['1,8'], sigma=sigma)
spectrum1['3,24_gauss'] = sp.ndimage.gaussian_filter(spectrum1['3,24'], sigma=sigma)

spectrum1.plot('keV', ['nf_gauss', '1,8_gauss', '3,24_gauss'], logy=True)

# %%
thresh = 220
spectrum1[thresh:].plot('keV', ['nf_gauss', '1,8_gauss', '3,24_gauss'], logy=True)
spectrum1[:thresh].plot('keV', ['nf_gauss', '1,8_gauss', '3,24_gauss'], logy=False)

# %%
t = spectrum1[:thresh]['3,24_gauss'].to_numpy()
tt = spectrum1[:thresh]['1,8_gauss'].to_numpy()
ttt = spectrum1[:thresh]['nf_gauss'].to_numpy()

# %%
sl = slice(27, -1)
# print(t[sl])
plt.plot(t[sl])
# plt.plot(sp.ndimage.gaussian_filter(t[sl], sigma=10))
# plt.plot(sp.ndimage.median_filter(t[sl], size=20))
t_m = sp.ndimage.gaussian_filter(sp.ndimage.median_filter(t[sl], size=20), sigma=10)
plt.plot(t_m)

plt.plot(tt[sl])
tt_m = sp.ndimage.gaussian_filter(sp.ndimage.median_filter(tt[sl], size=20), sigma=10)
plt.plot(tt_m)

# %%
plt.plot(tt_m)
plt.plot(t_m * tt_m[0] / t_m[0])

# %%
ttt_m = sp.ndimage.gaussian_filter(sp.ndimage.median_filter(ttt[sl], size=20), sigma=10)

plt.plot(ttt[sl])
plt.plot(ttt_m)
plt.plot(t_m * ttt_m[0] / t_m[0])

# %%
spec_nf = spectrum1['nf_gauss'].to_numpy()
spec_nf_corrected = np.copy(spec_nf)
spec_nf_noise = t_m * ttt_m[0] / t_m[0]
spec_nf_noise

sl = slice(27, 192+27)
spec_nf_corrected[sl] = spec_nf_corrected[sl] - spec_nf_noise

spec_nf_corrected[spec_nf_corrected < 0] = 0

plt.plot(spec_nf)
plt.plot(spec_nf_corrected)
plt.yscale('log')

# %%
spec_nf_corrected_1 = np.copy(spec_nf_corrected)
spec_nf_corrected_1[:117] = 0
spec_nf_corrected_1[900:] = 0
plt.plot(spectrum1['keV'], spec_nf_corrected_1)
plt.yscale('log')

# %%
np.save('Mo_spec_poly_50', spec_nf_corrected_1)

# %%

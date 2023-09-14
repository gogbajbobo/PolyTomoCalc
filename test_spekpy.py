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
import spekpy
import matplotlib.pyplot as plt
import numpy as np
import xraydb
from spec_gen import generate_spectrum

# %%
s = spekpy.Spek(kvp=45, dk=0.2, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(1024)])

_, s = generate_spectrum(40, 45, 'Mo', energies=en_keV)
att_air = np.exp(-xraydb.material_mu('air', en_keV*1000) * 144)

s /= s.sum()
sf = s * att_air
sf /= sf.sum()

plt.plot(en_keV, sf)
plt.grid()
plt.ylim([1e-6, 5e-1])
plt.yscale('log')

# %%
plt.plot(energies, intensities)
plt.plot(en_keV, sf)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%

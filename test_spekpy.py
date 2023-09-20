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
from _handler_funcs import generate_spectrum

# %%
en_step = (19.608 - 17.479) / (416 - 371)

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)

energies_1, intensities_1 = s.get_spectrum(flu=False)
intensities_1 /= intensities_1.sum()

plt.plot(energies_1, intensities_1)

plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
_, s = generate_spectrum(40, 45, 'Mo', energies=energies)
att_air = np.exp(-xraydb.material_mu('air', energies*1000) * 144)

s /= s.sum()
sf = s * att_air
sf /= sf.sum()

plt.plot(energies, sf)
plt.grid()
plt.ylim([1e-6, 5e-1])
plt.yscale('log')

# %%
plt.plot(energies, intensities)
plt.plot(energies_1, intensities_1)
plt.plot(energies, sf)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
input_path = 'Mo_spec_poly_45.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45 = np.load(f).astype(float)
    spec_Mo_45 /= spec_Mo_45.sum()

input_path = 'Mo_spec_poly_45_energies.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45_energies = np.load(f).astype(float)

idx_min = np.where(spec_Mo_45_energies < energies[0])[0][-1]
idx_max = np.where(spec_Mo_45_energies > energies[-1])[0][0]

spec_Mo_45_energies_1 = spec_Mo_45_energies[idx_min:idx_max]
spec_Mo_45_1 = spec_Mo_45[idx_min:idx_max]
spec_Mo_45_1 /= spec_Mo_45_1.sum()

spec_Mo_45_2 = spec_Mo_45_1 * spec_Mo_45_energies_1
spec_Mo_45_2 /= spec_Mo_45_2.sum()

# %%
plt.plot(energies, intensities)
plt.plot(energies_1, intensities_1)
plt.plot(energies, sf)
plt.plot(spec_Mo_45_energies_1, spec_Mo_45_1)
plt.plot(spec_Mo_45_energies_1, spec_Mo_45_2)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%

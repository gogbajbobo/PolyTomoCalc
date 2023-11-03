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
import spekpy
import matplotlib.pyplot as plt
import numpy as np
import xraydb
from _handler_funcs import generate_spectrum

# %%
en_step = (19.608 - 17.479) / (416 - 371)

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

s.filter('Air', 1440)
_, intensities_with_air = s.get_spectrum()
intensities_with_air /= intensities_with_air.sum()

att_air = np.exp(-xraydb.material_mu('air', energies * 1000) * 144)
intensities_with_air_2 = intensities * att_air
intensities_with_air_2 /= intensities_with_air_2.sum()

plt.plot(energies, intensities)
plt.plot(energies, intensities_with_air)
plt.plot(energies, intensities_with_air_2)

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
plt.plot(energies, intensities_with_air)
plt.grid()
plt.ylim([1e-6, 5e-1])
plt.yscale('log')

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

# %%
plt.plot(energies, intensities_with_air)
plt.plot(energies, sf)
plt.plot(spec_Mo_45_energies_1, spec_Mo_45_1)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
whos

# %%
s = spekpy.Spek(kvp=40, dk=0.25, targ='Mo')
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

s.filter('Air', 1440)
_, intensities_with_air = s.get_spectrum()
intensities_with_air /= intensities_with_air.sum()

plt.figure(figsize=(10, 5))

# plt.plot(energies, intensities)
plt.plot(energies, intensities_with_air)

plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
s = spekpy.Spek(kvp=50, dk=0.25, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()

s.filter('Al', 1)
_, intensities_1 = s.get_spectrum()
# intensities_1 /= intensities_1.sum()

s.filter('Al', 2)
_, intensities_2 = s.get_spectrum()
# intensities_2 /= intensities_2.sum()

fig, ax = plt.subplots(1, 3, figsize=(12, 2))
# ax[0].plot(energies, intensities)
ax[0].fill(energies, intensities)
ax[0].set_ylim([2e4, 3e8])
ax[0].set_yscale('log')
# ax[1].plot(energies, intensities_1)
ax[1].fill(energies, intensities_1)
ax[1].set_ylim([2e4, 3e8])
ax[1].set_yscale('log')
# ax[2].plot(energies, intensities_2)
ax[2].fill(energies, intensities_2)
ax[2].set_ylim([2e4, 3e8])
ax[2].set_yscale('log')
ax[1].set_xlabel('Energy, keV')
ax[0].set_ylabel('Intensity, counts')

plt.show()


# %%
s = spekpy.Spek(kvp=50, dk=1, targ='Mo', brem=False)
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()

s.filter('Al', 1)
_, intensities_1 = s.get_spectrum()
# intensities_1 /= intensities_1.sum()

s.filter('Al', 2)
_, intensities_2 = s.get_spectrum()
# intensities_2 /= intensities_2.sum()

fig, ax = plt.subplots(1, 3, figsize=(12, 2))
ax[0].plot(energies, intensities)
ax[0].set_xlim([17.2, 17.8])
ax[0].set_ylim([2e4, 3e8])
ax[0].set_yscale('log')
ax[1].plot(energies, intensities_1)
ax[1].set_xlim([17.2, 17.8])
ax[1].set_ylim([2e4, 3e8])
ax[1].set_yscale('log')
ax[2].plot(energies, intensities_2)
ax[2].set_xlim([17.2, 17.8])
ax[2].set_ylim([2e4, 3e8])
ax[2].set_yscale('log')
ax[1].set_xlabel('Energy, keV')
ax[0].set_ylabel('Intensity, counts')

plt.show()


# plt.figure(figsize=(10, 5))

# plt.plot(energies, intensities, linewidth=2)
# plt.xlim([17.2, 17.8])
# plt.ylim([2e4, 3e8])
# plt.yscale('log')
# plt.xlabel('Energy, keV')
# plt.ylabel('Intensity, counts')
# # plt.grid()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(energies, intensities_1, linewidth=2)
# plt.xlim([17.2, 17.8])
# plt.ylim([2e4, 3e8])
# plt.yscale('log')
# plt.xlabel('Energy, keV')
# plt.ylabel('Intensity, counts')
# # plt.grid()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(energies, intensities_2, linewidth=2)
# plt.xlim([17.2, 17.8])
# plt.ylim([2e4, 3e8])
# plt.yscale('log')
# plt.xlabel('Energy, keV')
# plt.ylabel('Intensity, counts')
# # plt.grid()
# plt.show()

# %%
s = spekpy.Spek(kvp=50, dk=0.25, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()

plt.fill(energies, intensities)
plt.ylim([2e4, 3e8])
plt.yscale('log')
plt.xlabel('Energy, keV')
plt.ylabel('Photons, counts')

plt.show()


# %%
input_path = 'Mo_spec_poly_50.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50 = np.load(f).astype(float)
    spec_Mo_50 /= spec_Mo_50.sum()

input_path = 'Mo_spec_poly_50_energies.npy'
with open(input_path, 'rb') as f:
    Mo_spec_poly_50_energies = np.load(f).astype(float)


en_step = np.mean(Mo_spec_poly_50_energies[1:] - Mo_spec_poly_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(Mo_spec_poly_50_energies, spec_Mo_50, label='Experiment', c='green')
plt.plot([], [])
plt.plot(energies, intensities, label='Model')
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.xlabel('Energy, keV')
plt.ylabel('Intensity, a.u.')
plt.legend()
plt.grid()

# %%
en_slice = slice(250, 300)
# plt.plot(energies[en_slice], intensities[en_slice])
max1 = np.max(intensities[en_slice])

en_slice = slice(300, 400)
plt.plot(energies[en_slice], intensities[en_slice])
max2 = np.max(intensities[en_slice])

print(max1, max2, max1/max2)

# %%

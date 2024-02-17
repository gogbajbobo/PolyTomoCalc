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
import xraydb
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from skimage.io import imread

import spekpy


# %% [markdown]
# ## **Spectram tomography results**
#
# Alpha, beta, poly slices

# %% editable=true slideshow={"slide_type": ""}
v_index = '048'

im_alpha_0 = imread(f'/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/0/tomo_k_alpha{v_index}.tif')
im_beta_0 = imread(f'/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/0/tomo_k_beta{v_index}.tif')
im_poly_0 = imread(f'/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/0/tomo_poly{v_index}.tif')

im_alpha_0[im_alpha_0 < 0] = 0
im_beta_0[im_beta_0 < 0] = 0
im_poly_0[im_poly_0 < 0] = 0

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(im_alpha_0)
ax[0].set_title('MoKα', fontsize=18)
ax[0].axhline(80, linewidth=4, c='white', alpha=0.5)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_beta_0)
ax[1].set_title('MoKβ', fontsize=18)
ax[1].axhline(80, linewidth=4, c='white', alpha=0.5)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(im_poly_0)
ax[2].set_title('Poly', fontsize=18)
ax[2].axhline(80, linewidth=4, c='white', alpha=0.5)
plt.colorbar(im2, ax=ax[2])
plt.show()

# %% jupyter={"source_hidden": true}
# im_alpha_1 = imread('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/1/rec_alpha_118.tif')
# im_beta_1 = imread('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/1/rec_beta_118.tif')
# im_poly_1 = imread('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Poly_NaCl_slices/1/rec_poly_118.tif')

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# im0 = ax[0].imshow(im_alpha_1)
# plt.colorbar(im0, ax=ax[0])
# im1 = ax[1].imshow(im_beta_1)
# plt.colorbar(im1, ax=ax[1])
# im2 = ax[2].imshow(im_poly_1)
# plt.colorbar(im2, ax=ax[2])
# plt.show()

# %%
Mo_lines = xraydb.xray_lines('Mo')
Mo_lines

# %%
alpha_lines = [Mo_lines[line] for line in ['Ka1', 'Ka2', 'Ka3']]
beta_lines = [Mo_lines[line] for line in ['Kb1', 'Kb2', 'Kb3', 'Kb5']]

def lines_energies(lines):
    return [line.energy for line in lines]

def lines_intensities(lines):
    return [line.intensity for line in lines]


Mo_Ka_average = np.average(lines_energies(alpha_lines), weights=lines_intensities(alpha_lines))
Mo_Kb_average = np.average(lines_energies(beta_lines), weights=lines_intensities(beta_lines))

print(lines_energies(alpha_lines), lines_energies(beta_lines))
print(lines_intensities(alpha_lines), lines_intensities(beta_lines))
print(Mo_Ka_average, Mo_Kb_average)
Mo_Ka_NaCl_mu = xraydb.material_mu('NaCl', Mo_Ka_average) / 10
Mo_Kb_NaCl_mu = xraydb.material_mu('NaCl', Mo_Kb_average) / 10

print(Mo_Ka_NaCl_mu)
print(Mo_Kb_NaCl_mu)

# %%
h_line = 80
h_line_width = 10
h_slice = slice(h_line - h_line_width//2, h_line + h_line_width//2)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(np.mean(im_alpha_0[h_slice], axis=0))
ax[0].axhline(Mo_Ka_NaCl_mu, c='red')
ax[0].text(90, 1.65, r'µ$_{NaCl}$@MoKα', fontsize=16, c='red')
ax[0].set_ylim(0, 2)
ax[0].set_title('MoKα', fontsize=18)
ax[0].set_ylabel('µ, 1/mm', fontsize=18)
ax[1].plot(np.mean(im_beta_0[h_slice], axis=0))
ax[1].axhline(Mo_Kb_NaCl_mu, c='red')
ax[1].text(90, 1.17, r'µ$_{NaCl}$@MoKβ', fontsize=16, c='red')
ax[1].set_ylim(0, 1.4)
ax[1].set_title('MoKβ', fontsize=18)
ax[2].plot(np.mean(im_poly_0[h_slice], axis=0))
ax[2].axhline(0.4, c='red')
ax[2].text(100, 0.375, r'µ$_{NaCl}@\bf???$', fontsize=16, c='red')
ax[2].set_ylim(0, 0.45)
ax[2].set_title('Poly', fontsize=18)
plt.show()

# %% [markdown]
# ## **SiC samples**
#
# No filter, Al-filter 1.8mm, Al-filter 3.24mm

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_bh_corr_2.57_optimal.npy'
with open(input_path, 'rb') as f:
  bim_SiC = np.load(f)

bim_SiC = gaussian(median(bim_SiC))
thresh = threshold_otsu(bim_SiC)
print(f'thresh: {thresh}')
bim_SiC = (bim_SiC > thresh).astype(int)

# bim_SiC = gaussian(median(bim_SiC)) > 2

plt.imshow(bim_SiC)
plt.axis('off')
plt.show()

# %%
h_line_SiC = 550

def process_porous_pic(im, bim):
    im = gaussian(median(im))
    im[im < 0] = 0
    im[bim == 0] = 0
    return im

def show_porous_pic_with_profile(im, h_profile_line):

    plt.imshow(im)
    plt.axhline(h_profile_line, linewidth=4, c='white', alpha=0.5)
    plt.colorbar()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(im[h_profile_line])
    plt.grid()
    plt.show()


input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_bh_corr_1.0.npy'
with open(input_path, 'rb') as f:
  im_SiC_0 = np.load(f)

im_SiC_0 = process_porous_pic(im_SiC_0, bim_SiC)

plt.imshow(im_SiC_0)
plt.axis('off')
plt.show()

show_porous_pic_with_profile(im_SiC_0, h_line_SiC)

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/b839cf5f-bde2-4822-9ac6-eef01eace089_bh_corr_1.0.npy'
with open(input_path, 'rb') as f:
  im_SiC_18 = np.load(f)

im_SiC_18 = process_porous_pic(im_SiC_18, bim_SiC)

show_porous_pic_with_profile(im_SiC_18, h_line_SiC)

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/746545ab-e1c6-4001-aff0-0fd1f0de1882_bh_corr_1.0.npy'
with open(input_path, 'rb') as f:
  im_SiC_324 = np.load(f)

im_SiC_324 = process_porous_pic(im_SiC_324, bim_SiC)

show_porous_pic_with_profile(im_SiC_324, h_line_SiC)

# %% [markdown]
# ## **Source spectrums** (SiC samples)
#
# Mo-anode 50keV no filter, 1.8mm and 3.24mm Al-filter

# %%
# Mo_spec_*.npy should be created in spectrum_analysis.ipynb first

input_path = 'Mo_spec_poly_50_energies.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_energies = np.load(f)

# spec_Mo_50_energies = spec_Mo_50_energies[spec_Mo_50_energies > 0]
spec_Mo_50_energies = spec_Mo_50_energies[27:]

input_path = 'Mo_spec_poly_50.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_0 = np.load(f)[-spec_Mo_50_energies.size:].astype(float)
    spec_Mo_50_0 /= spec_Mo_50_0.sum()

input_path = 'Mo_spec_poly_50_18.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_18 = np.load(f)[-spec_Mo_50_energies.size:].astype(float)
    spec_Mo_50_18 /= spec_Mo_50_18.sum()

input_path = 'Mo_spec_poly_50_324.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_324 = np.load(f)[-spec_Mo_50_energies.size:].astype(float)
    spec_Mo_50_324 /= spec_Mo_50_324.sum()


plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, spec_Mo_50_0, label='no filter')
plt.plot(spec_Mo_50_energies, spec_Mo_50_18, label='Al filter, 1.8 mm')
plt.plot(spec_Mo_50_energies, spec_Mo_50_324, label='Al filter, 3.24 mm')
plt.ylim(1e-5, 1e-1)
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

# %%
input_path = 'Mo_spec_poly_50.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_0_counts = np.load(f)[-spec_Mo_50_energies.size:].astype(float)


plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, spec_Mo_50_0_counts)
# plt.ylim(1e-8, 1e-1)
plt.yscale('log')
plt.xlabel('Energy, keV', fontsize=16)
plt.ylabel('Photons, count', fontsize=16)
# plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, spec_Mo_50_0_counts)
# plt.ylim(1e-8, 1e-1)
plt.yscale('log')
plt.xlabel('Energy, keV', fontsize=16)
plt.ylabel('Photons, count', fontsize=16)
plt.grid()
plt.show()

# %%
air_mu = xraydb.material_mu('Air', [17500, 19600]) / 10
air_t = np.exp(-air_mu * 1440)

slice_index = slice(270, 300)
# plt.plot(spec_Mo_50_energies[slice_index], spec_Mo_50_0_counts[slice_index])
# plt.show()

max1 = np.max(spec_Mo_50_0_counts[slice_index])

slice_index = slice(300, 350)
# plt.plot(spec_Mo_50_energies[slice_index], spec_Mo_50_0_counts[slice_index])
# plt.show()

max2 = np.max(spec_Mo_50_0_counts[slice_index])

print(max1, max2, air_t)

initial_intensity = np.array([max1, max2]) / air_t
print(initial_intensity)
print(max1/max2)
print(initial_intensity[0]/initial_intensity[1])

# %% [markdown]
# ## **Gadolinium oxysulfide (GOS)**

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus_50 = xraydb.material_mu('GOS', spec_Mo_50_energies * 1000) / 10
GOS_t_50 = np.exp(-GOS_mus_50 * 22 * 0.001) # (22 * 0.001)mm == 22µm

# %%
# beta = 3.7
# en_gap = 4.6 # eV
# GOS_n_p_50 = spec_Mo_50_energies * 1000 / (beta * en_gap)

qe = 60 # photon/keV
GOS_n_p_50 = spec_Mo_50_energies * qe

# GOS_n_p_50 = np.ones(spec_Mo_50_energies.size)

GOS_eff_50 = GOS_n_p_50 * (1 - GOS_t_50)

fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax[0].plot(spec_Mo_50_energies, GOS_n_p_50)
ax[0].grid()
ax[0].set_ylabel('Photons, count', fontsize=16)
ax[1].plot(spec_Mo_50_energies, 1-GOS_t_50)
ax[1].grid()
ax[1].set_ylabel('Attenuation, a.u.', fontsize=16)
ax[2].plot(spec_Mo_50_energies, GOS_eff_50)
ax[2].grid()
ax[2].set_ylabel('Photons, count', fontsize=16)
ax[1].set_xlabel('Energy, keV', fontsize=16)
plt.show()

plt.plot(spec_Mo_50_energies, GOS_eff_50)
plt.grid()
plt.show()

# %% [markdown]
# ## **Effective spectrum Mo 50keV**

# %%
effective_spec_Mo_50_0 = spec_Mo_50_0 * GOS_eff_50
effective_spec_Mo_50_18 = spec_Mo_50_18 * GOS_eff_50
effective_spec_Mo_50_324 = spec_Mo_50_324 * GOS_eff_50

effective_spec_Mo_50_0 /= effective_spec_Mo_50_0.sum()
effective_spec_Mo_50_18 /= effective_spec_Mo_50_18.sum()
effective_spec_Mo_50_324 /= effective_spec_Mo_50_324.sum()

plt.plot(spec_Mo_50_energies, spec_Mo_50_0)
plt.plot(spec_Mo_50_energies, effective_spec_Mo_50_0)
# plt.plot(spec_Mo_50_energies, spec_Mo_50_18)
# plt.plot(spec_Mo_50_energies, effective_spec_Mo_50_18)
# plt.plot(spec_Mo_50_energies, spec_Mo_50_324)
# plt.plot(spec_Mo_50_energies, effective_spec_Mo_50_324)
plt.yscale('log')
plt.grid()
plt.show()

# %% [markdown]
# ## **SiC poly attenuation**

# %%
xraydb.add_material('SiC','SiC', 3.21)
print(f"SiC attenuation at MoKa1: { xraydb.material_mu('SiC', Mo_lines['Ka1'].energy) / 10 } 1/mm")

# %%
att_SiC = xraydb.material_mu('SiC', spec_Mo_50_energies * 1000) / 10

plt.plot(spec_Mo_50_energies, att_SiC)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

# %%
poly_mu_SiC = (att_SiC * spec_Mo_50_0).sum()
poly_mu_SiC_effective = (att_SiC * effective_spec_Mo_50_0).sum()
print(poly_mu_SiC)
print(poly_mu_SiC_effective)

# %%
transmissions_at_depths = np.exp(np.outer(-att_SiC, [0, 0.1, 0.3, 1.0, 3.0, 10.0])).T

plt.plot(spec_Mo_50_energies, transmissions_at_depths.T)
plt.show()

# %%
passed_spectrums = transmissions_at_depths * spec_Mo_50_0

plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, passed_spectrums.T)
plt.ylim(1e-10, 1e-1)
plt.yscale('log')
plt.grid()
plt.show()

passed_spectrums = np.array([pass_spec/pass_spec.sum() for pass_spec in passed_spectrums])
plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, passed_spectrums.T)
plt.ylim(1e-10, 1e-1)
plt.yscale('log')
plt.grid()
plt.show()


# %% [markdown]
# ## **SiC poly µ at depths calculation**

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm
total_lenght = 10 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_SiC_at_depths = np.exp(np.outer(-att_SiC, length_ticks)).T

passed_spectrums_50_0 = transmissions_SiC_at_depths * spec_Mo_50_0
passed_spectrums_50_18 = transmissions_SiC_at_depths * spec_Mo_50_18
passed_spectrums_50_324 = transmissions_SiC_at_depths * spec_Mo_50_324

passed_spectrums_50_0_eff = transmissions_SiC_at_depths * effective_spec_Mo_50_0
passed_spectrums_50_18_eff = transmissions_SiC_at_depths * effective_spec_Mo_50_18
passed_spectrums_50_324_eff = transmissions_SiC_at_depths * effective_spec_Mo_50_324

p_specs_norm_50_0 = (passed_spectrums_50_0.T / np.sum(passed_spectrums_50_0, axis=1)).T
p_specs_norm_50_18 = (passed_spectrums_50_18.T / np.sum(passed_spectrums_50_18, axis=1)).T
p_specs_norm_50_324 = (passed_spectrums_50_324.T / np.sum(passed_spectrums_50_324, axis=1)).T

p_specs_norm_50_0_eff = (passed_spectrums_50_0_eff.T / np.sum(passed_spectrums_50_0_eff, axis=1)).T
p_specs_norm_50_18_eff = (passed_spectrums_50_18_eff.T / np.sum(passed_spectrums_50_18_eff, axis=1)).T
p_specs_norm_50_324_eff = (passed_spectrums_50_324_eff.T / np.sum(passed_spectrums_50_324_eff, axis=1)).T

plt.plot(length_ticks, np.sum(passed_spectrums_50_0, axis=1), label='50_0')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_0, axis=1))
plt.plot(length_ticks, np.sum(passed_spectrums_50_18, axis=1), label='50_18')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_18, axis=1))
plt.plot(length_ticks, np.sum(passed_spectrums_50_324, axis=1), label='50_324')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_324, axis=1))

plt.plot(length_ticks, np.sum(passed_spectrums_50_0_eff, axis=1), label='50_0_eff')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_0_eff, axis=1))
plt.plot(length_ticks, np.sum(passed_spectrums_50_18_eff, axis=1), label='50_18_eff')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_18_eff, axis=1))
plt.plot(length_ticks, np.sum(passed_spectrums_50_324_eff, axis=1), label='50_324_eff')
# plt.plot(length_ticks, np.sum(p_specs_norm_50_324_eff, axis=1))

plt.legend()
plt.grid()
plt.show()

print(p_specs_norm_50_0.shape)


# %%
def calc_mu_at_depths(spectrums):
    spec_sums = np.sum(spectrums, axis=1)
    mu_at_depths = -np.log(spec_sums[1:] / spec_sums[:-1]) / voxel_size
    return mu_at_depths

# %% jupyter={"source_hidden": true}
### This is differential µ calculation, but we have to use effective µ (from 9µm length attenuation)

# poly_SiC_mu_map_50_0 = p_specs_norm_50_0 * att_SiC
# poly_SiC_mu_at_depth_50_0 = np.sum(poly_SiC_mu_map_50_0, axis=1)
# poly_SiC_mu_map_50_18 = p_specs_norm_50_18 * att_SiC
# poly_SiC_mu_at_depth_50_18 = np.sum(poly_SiC_mu_map_50_18, axis=1)
# poly_SiC_mu_map_50_324 = p_specs_norm_50_324 * att_SiC
# poly_SiC_mu_at_depth_50_324 = np.sum(poly_SiC_mu_map_50_324, axis=1)

# poly_SiC_mu_map_50_0_eff = p_specs_norm_50_0_eff * att_SiC
# poly_SiC_mu_at_depth_50_0_eff = np.sum(poly_SiC_mu_map_50_0_eff, axis=1)
# poly_SiC_mu_map_50_18_eff = p_specs_norm_50_18_eff * att_SiC
# poly_SiC_mu_at_depth_50_18_eff = np.sum(poly_SiC_mu_map_50_18_eff, axis=1)
# poly_SiC_mu_map_50_324_eff = p_specs_norm_50_324_eff * att_SiC
# poly_SiC_mu_at_depth_50_324_eff = np.sum(poly_SiC_mu_map_50_324_eff, axis=1)

# print(poly_SiC_mu_map_50_0.shape)
# print(poly_SiC_mu_at_depth_50_0.shape)


# %%
poly_SiC_mu_at_depth_50_0 = calc_mu_at_depths(passed_spectrums_50_0)
poly_SiC_mu_at_depth_50_18 = calc_mu_at_depths(passed_spectrums_50_18)
poly_SiC_mu_at_depth_50_324 = calc_mu_at_depths(passed_spectrums_50_324)

poly_SiC_mu_at_depth_50_0_eff = calc_mu_at_depths(passed_spectrums_50_0_eff)
poly_SiC_mu_at_depth_50_18_eff = calc_mu_at_depths(passed_spectrums_50_18_eff)
poly_SiC_mu_at_depth_50_324_eff = calc_mu_at_depths(passed_spectrums_50_324_eff)

print(poly_SiC_mu_at_depth_50_0.shape)

plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_0, label='50_0')
plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_18, label='50_18')
plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_324, label='50_324')

plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_0_eff, label='50_0_eff')
plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_18_eff, label='50_18_eff')
plt.plot(length_ticks[1:], poly_SiC_mu_at_depth_50_324_eff, label='50_324_eff')

# plt.yscale('log')
plt.legend()
plt.grid()
plt.show()


# %%
def fill_im_with_poly_mu(fill_values, bin_im):
    out_im = np.zeros(bin_im.shape)
    for y, bin_row in enumerate(bin_im):
      for i, x in enumerate(np.where(bin_row > 0)[0]):
        out_im[y, x] = fill_values[i] * bin_row[x]
    return out_im


# %%
out_im = fill_im_with_poly_mu(poly_SiC_mu_at_depth_50_0_eff, bim_SiC)
# plt.imshow(out_im, norm=LogNorm())
plt.imshow(out_im)
plt.colorbar()
plt.show()


# %% [markdown]
# ## **Naive reconstruction**

# %%
def naive_reconstruction(bin_im, poly_mu_at_depth):
    att_sum_obj = np.zeros(bin_im.shape)
    angles = np.arange(0, 360, 18)
    
    for angle in angles:
        rot_bin_im = rotate(bin_im, angle, order=1, preserve_range=True)
        out_im = fill_im_with_poly_mu(poly_mu_at_depth, rot_bin_im)
        att_sum_obj += rotate(out_im, -angle, order=1, preserve_range=True)
    
    att_sum_obj /= angles.size
    att_sum_obj[bin_im == 0] = 0

    return att_sum_obj


# %%
def show_porous_profiles(images, h_profile_line):
    plt.figure(figsize=(10, 5))
    for im in images:
        plt.plot(im[h_profile_line])
    plt.grid()
    plt.show()


# %%
naive_recon_obj = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_0)
naive_recon_obj_eff = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_0_eff)

vmax = np.max(np.maximum(np.maximum(im_SiC_0, naive_recon_obj), naive_recon_obj_eff))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(im_SiC_0, vmax=vmax)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(naive_recon_obj, vmax=vmax)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(naive_recon_obj_eff, vmax=vmax)
plt.colorbar(im2, ax=ax[2])
plt.show()

show_porous_profiles([im_SiC_0, naive_recon_obj], h_line_SiC)
show_porous_profiles([im_SiC_0, naive_recon_obj_eff], h_line_SiC)

# %%
naive_recon_obj = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_18)
naive_recon_obj_eff = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_18_eff)

vmax = np.max(np.maximum(np.maximum(im_SiC_0, naive_recon_obj), naive_recon_obj_eff))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(im_SiC_0, vmax=vmax)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(naive_recon_obj, vmax=vmax)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(naive_recon_obj_eff, vmax=vmax)
plt.colorbar(im2, ax=ax[2])
plt.show()

show_porous_profiles([im_SiC_0, naive_recon_obj], h_line_SiC)
show_porous_profiles([im_SiC_0, naive_recon_obj_eff], h_line_SiC)

# %%
naive_recon_obj = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_324)
naive_recon_obj_eff = naive_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_324_eff)

vmax = np.max(np.maximum(np.maximum(im_SiC_0, naive_recon_obj), naive_recon_obj_eff))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(im_SiC_0, vmax=vmax)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(naive_recon_obj, vmax=vmax)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(naive_recon_obj_eff, vmax=vmax)
plt.colorbar(im2, ax=ax[2])
plt.show()

show_porous_profiles([im_SiC_0, naive_recon_obj], h_line_SiC)
show_porous_profiles([im_SiC_0, naive_recon_obj_eff], h_line_SiC)


# %% [markdown]
# ## **µ-filled reconstruction**

# %%
def mu_filled_sum_reconstruction(bin_im, poly_mu_at_depth):
    angles = np.arange(0, 180, 1)
    sinogram = np.zeros((angles.size, bin_im.shape[0]))
    
    for j, angle in enumerate(angles):
      rot_bin_im = rotate(bin_im, angle, order=1, preserve_range=True)
      out_im = fill_im_with_poly_mu(poly_mu_at_depth, rot_bin_im)
      sinogram[j] = np.sum(out_im, axis=1)

    recon = gaussian(iradon(sinogram.T, theta=angles).T)
    recon[bin_im == 0] = 0
    recon[recon < 0] = 0

    return recon, sinogram
    
recon_SiC_50_0, sinogram_50_0 = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_0)

# %%
plt.imshow(sinogram_50_0, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_0, axis=1))
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_0)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_0)
plt.colorbar(im1, ax=ax[1])
plt.show()

# %%
show_porous_profiles([im_SiC_0, recon_SiC_50_0], h_line_SiC)

# %%
recon_SiC_50_0_eff, sinogram_50_0_eff = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_0_eff)

# %%
plt.imshow(sinogram_50_0_eff[:1].T, aspect=0.01)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.colorbar()
plt.show()


# %%
plt.imshow(sinogram_50_0_eff, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_0_eff, axis=1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_0_eff)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_0)
plt.colorbar(im1, ax=ax[1])
plt.show()

show_porous_profiles([im_SiC_0, recon_SiC_50_0_eff], h_line_SiC)

# %%
plt.figure(figsize=(10, 5))
plt.plot(im_SiC_0[h_line_SiC], label='Эксперимент')
plt.plot(recon_SiC_50_0_eff[h_line_SiC], label='Моделирование')
plt.xlabel('voxels')
plt.ylabel('Attenuation µ, 1/mm')
plt.legend()
plt.grid()
plt.show()

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[1, 0.8, 1])

im0 = ax[0].imshow(im_SiC_0)
ax[0].axhline(im_SiC_0.shape[0] // 2, linewidth=4, c='white', alpha=0.5)
plt.colorbar(im0, ax=ax[0])
ax[0].set_title('а', fontsize=24)

ax[1].imshow(bim_SiC)
# plt.colorbar(im1, ax=ax[1])
ax[1].set_title('б', fontsize=24)

im2 = ax[2].imshow(recon_SiC_50_0_eff)
plt.colorbar(im2, ax=ax[2])
ax[2].set_title('в', fontsize=24)

plt.show()


# %%
default_blue_color = u'#1f77b4'

plt.figure(figsize=(10, 5))
plt.plot(im_SiC_0[h_line_SiC], label='Эксперимент', c='black', linewidth=0.5)
plt.plot(recon_SiC_50_0_eff[h_line_SiC], label='Моделирование', c=default_blue_color, linewidth=0.75, linestyle=(0, (7.5, 5)))
plt.xlabel('воксели')
plt.ylabel('Коэффициент ослабления µ, 1/мм')
plt.legend()
plt.grid()
plt.show()

# %%
recon_SiC_50_18, sinogram_50_18 = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_18)

# %%
plt.imshow(sinogram_50_18, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_18, axis=1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_18)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_18)
plt.colorbar(im1, ax=ax[1])
plt.show()

show_porous_profiles([im_SiC_18, recon_SiC_50_18], h_line_SiC)

# %%
recon_SiC_50_18_eff, sinogram_50_18_eff = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_18_eff)

# %%
plt.imshow(sinogram_50_18_eff, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_18_eff, axis=1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_18_eff)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_18)
plt.colorbar(im1, ax=ax[1])
plt.show()

show_porous_profiles([im_SiC_18, recon_SiC_50_18_eff], h_line_SiC)

# %%
recon_SiC_50_324, sinogram_50_324 = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_324)

# %%
plt.imshow(sinogram_50_324, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_324, axis=1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_324)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_18)
plt.colorbar(im1, ax=ax[1])
plt.show()

show_porous_profiles([im_SiC_18, recon_SiC_50_324], h_line_SiC)

# %%
recon_SiC_50_324_eff, sinogram_50_324_eff = mu_filled_sum_reconstruction(bim_SiC, poly_SiC_mu_at_depth_50_324_eff)

# %%
plt.imshow(sinogram_50_324_eff, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(np.sum(sinogram_50_324_eff, axis=1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon_SiC_50_324_eff)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_18)
plt.colorbar(im1, ax=ax[1])
plt.show()

show_porous_profiles([im_SiC_18, recon_SiC_50_324_eff], h_line_SiC)


# %% [markdown]
# ## **Reconstruction with spectrum attenuation calculation and GOS**

# %%
def calc_object_mus_from_spectrum(bin_im, exp_im, spectrum, mat_att, voxel_size, GOS_eff=None, h_line=None):

    angles = np.arange(0, 180, 1)
    sino_shape = (angles.size, bin_im.shape[0])
    sinogram_wo_GOS = np.zeros(sino_shape)

    have_GOS_eff = GOS_eff is not None
    if have_GOS_eff:
        sinogram_GOS = np.zeros(sino_shape)
    
    current_spec_GOS = spectrum * GOS_eff if have_GOS_eff else spectrum
    
    for j, angle in enumerate(angles):
        
        r_im = rotate(bin_im, angle, order=1, preserve_range=True)
        ray_sums = np.sum(r_im, axis=1) * voxel_size
        trans_sums = np.exp(np.outer(-mat_att, ray_sums)).T
        
        passed_specs_wo_GOS = trans_sums * spectrum
        if have_GOS_eff:
            passed_specs_GOS = passed_specs_wo_GOS * GOS_eff
            passed_specs_GOS /= current_spec_GOS.sum()
        
        passed_intensity_wo_GOS = np.sum(passed_specs_wo_GOS, axis=1)
        if have_GOS_eff:
            passed_intensity_GOS = np.sum(passed_specs_GOS, axis=1)
        
        sinogram_wo_GOS[j] = -np.log(passed_intensity_wo_GOS)
        if have_GOS_eff:
            sinogram_GOS[j] = -np.log(passed_intensity_GOS)
        
    recon_wo_GOS = gaussian(iradon(sinogram_wo_GOS.T, theta=angles).T)
    recon_wo_GOS[bin_im == 0] = 0
    recon_wo_GOS[recon_wo_GOS < 0] = 0
    recon_wo_GOS /= voxel_size # convert to 1/mm values

    if have_GOS_eff:
        recon_GOS = gaussian(iradon(sinogram_GOS.T, theta=angles).T)
        recon_GOS[bin_im == 0] = 0
        recon_GOS[recon_GOS < 0] = 0
        recon_GOS /= voxel_size # convert to 1/mm values
    
    fig, ax = plt.subplots(1, 3 if have_GOS_eff else 2, figsize=(15 if have_GOS_eff else 10, 5))
    im0 = ax[0].imshow(exp_im)
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(recon_wo_GOS)
    plt.colorbar(im1, ax=ax[1])
    if have_GOS_eff:
        im2 = ax[2].imshow(recon_GOS)
        plt.colorbar(im2, ax=ax[2])
    plt.show()
    
    row = h_line or bin_im.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(recon_wo_GOS[row], linewidth=.5, c='green')
    if have_GOS_eff:
        plt.plot(recon_GOS[row], linewidth=.5)
    
    # exp_row = np.sum(exp_im[row-10:row+10], axis=0) / exp_im[row-10:row+10].shape[0]
    exp_row = exp_im[row]
    
    plt.plot(exp_row, linewidth=1, c='gray')
    plt.ylabel('Коэффициент ослабления, 1/мм')
    plt.grid(color='gray')
    plt.show()
    
    return (recon_wo_GOS, recon_GOS) if have_GOS_eff else recon_wo_GOS



# %%
_ = calc_object_mus_from_spectrum(bim_SiC, im_SiC_0, spec_Mo_50_0, att_SiC, voxel_size, GOS_eff_50, h_line_SiC)

# %%
en_step = np.mean(spec_Mo_50_energies[1:] - spec_Mo_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.plot(spec_Mo_50_energies, spec_Mo_50_0)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %% [markdown]
# ## **Iohexol samles**
#
# Mo-anode 45keV spectrum

# %% [markdown]
# ## **GOS eff for Ihx**

# %%
GOS_mus_45 = xraydb.material_mu('GOS', spec_Mo_45_energies * 1000) / 10
GOS_t_45 = np.exp(-GOS_mus_45 * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV

GOS_n_p_45 = spec_Mo_45_energies * qe
# GOS_n_p_45 = np.ones(spec_Mo_45_energies.size)

GOS_eff_45 = GOS_n_p_45 * (1 - GOS_t_45)

plt.plot(spec_Mo_45_energies, GOS_eff_45)
plt.grid()
plt.show()

# %%
# see spec_edit.ipynb to load, handle and save spectrum

input_path = 'Mo_spec_poly_45.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45 = np.load(f).astype(float)
    # spec_Mo_45 /= spec_Mo_45.sum()

input_path = 'Mo_spec_poly_45_energies.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45_energies = np.load(f).astype(float)

# en_step = (19.608 - 17.479) / (416 - 371)
# spec_Mo_45_energies = np.array([17.479 + (i - 371) * en_step for i in np.arange(spec_Mo_45.shape[0])])

plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.yscale('log')
plt.grid()
plt.show()

# %%
effective_spec_Mo_45 = spec_Mo_45 * GOS_eff_45
effective_spec_Mo_45 /= effective_spec_Mo_45.sum()

plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.plot(spec_Mo_45_energies, effective_spec_Mo_45)
plt.yscale('log')
plt.grid()
plt.show()


# %% [markdown]
# ## **Calc Iohexol water solution density**

# %%
def calc_Ihx_solution(ihx_parts=1, wat_parts=1):

    # ihx_parts = 5 #4.17
    # wat_parts = 60
    
    ihx_density = 2.2 #g/cm3 — Iohexol
    wat_density = 0.997 # — Water
    ihx_weight = 0.647 #g — Ihx in 1ml of water (fabric solution)
    ihx_density_solution_1 = ihx_weight + wat_density * (1 - ihx_weight/ihx_density)
    print('factory solution density:', ihx_density_solution_1)
    
    ihx_density_solution_2 = (ihx_density_solution_1 * ihx_parts + wat_density * wat_parts) / (ihx_parts + wat_parts)
    
    print('solution density:', ihx_density_solution_2)
    
    ihx_molar_mass = 821.144 #g/mol
    wat_molar_mass = 18.01528
    
    ihx_mol = ihx_weight * ihx_parts / ihx_molar_mass # mol
    wat_mol = (wat_density * (1 - ihx_weight/ihx_density) * ihx_parts + wat_density * wat_parts) / wat_molar_mass
    # print(ihx_mol)
    # print(wat_mol)
    
    mol_ratio = wat_mol / ihx_mol
    # print(mol_ratio)
    hydrogen_coeff = int(np.rint(mol_ratio * 2))
    oxygen_coeff = int(np.rint(mol_ratio))
    
    print('H2O:', hydrogen_coeff, oxygen_coeff)

    return hydrogen_coeff, oxygen_coeff, ihx_density_solution_2


# %% [markdown]
# ## **Ihx samples**

# %% [markdown]
# ## Water/Iohexol 1/12

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/e82c1068-5c0f-40c3-9dba-4e811b566344.npy'
with open(input_path, 'rb') as f:
  im_ihx_1_12_mono = np.load(f)

# print('im_ihx_1_12_mono', im_ihx_1_12_mono.shape)

im_ihx_1_12_mono = im_ihx_1_12_mono[40:1260, 40:1260]
im_ihx_1_12_mono[im_ihx_1_12_mono < 0] = 0

# print('im_ihx_1_12_mono', im_ihx_1_12_mono.shape)

plt.imshow(im_ihx_1_12_mono)
plt.colorbar()
plt.show()

plt.plot(im_ihx_1_12_mono[650])
plt.plot(sp.ndimage.gaussian_filter(im_ihx_1_12_mono[650], sigma=2))
plt.show()

print(np.mean(im_ihx_1_12_mono[650, 200:1000]))

# %%
# h_c, o_c, d = calc_Ihx_solution(5, 60) #4.17:60
h_c, o_c, d = calc_Ihx_solution(4.5, 60) #4.17:60

xraydb.add_material('iohexol_1_12', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_mu_1_12 = xraydb.material_mu('iohexol_1_12', spec_Mo_45_energies*1000) / 10
plt.plot(spec_Mo_45_energies, iohexol_mu_1_12)
plt.yscale('log')
plt.show()

spec_45_MoKa_idx = 371 # 371 — index for 17.479 keV
iohexol_MoKa_mu_1_12 = iohexol_mu_1_12[spec_45_MoKa_idx]
print(iohexol_MoKa_mu_1_12)

# %%
bim_ihx_1_12_mono = (gaussian(im_ihx_1_12_mono, sigma=3) > 0.1).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_ihx_1_12_mono))
ax[1].imshow(bim_ihx_1_12_mono)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12_mono, gaussian(im_ihx_1_12_mono), np.array([1]), iohexol_MoKa_mu_1_12, voxel_size)

# %%
poly_mu_ihx_1_12 = (iohexol_mu_1_12 * spec_Mo_45).sum()
poly_mu_ihx_1_12_eff = (iohexol_mu_1_12 * effective_spec_Mo_45).sum()
print(poly_mu_ihx_1_12)
print(poly_mu_ihx_1_12_eff)

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm
total_lenght = 10 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_ihx_1_12_at_depths = np.exp(np.outer(-iohexol_mu_1_12, length_ticks)).T

passed_spectrums_45 = transmissions_ihx_1_12_at_depths * spec_Mo_45
passed_spectrums_45_eff = transmissions_ihx_1_12_at_depths * effective_spec_Mo_45

print(passed_spectrums_45.shape, passed_spectrums_45_eff.shape)

plt.plot(passed_spectrums_45[0])
plt.plot(passed_spectrums_45_eff[0])
plt.yscale('log')
plt.show()

# %%
poly_ihx_1_12_mu_at_depth = calc_mu_at_depths(passed_spectrums_45)
poly_ihx_1_12_mu_at_depth_eff = calc_mu_at_depths(passed_spectrums_45_eff)

plt.plot(length_ticks[1:], poly_ihx_1_12_mu_at_depth, label='45')
plt.plot(length_ticks[1:], poly_ihx_1_12_mu_at_depth_eff, label='45_eff')

# plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/dbace4ca-3ba6-4a8a-b191-d52fe70c8a4f.npy'
with open(input_path, 'rb') as f:
  im_ihx_1_12 = np.load(f)

# print('im_ihx_1_12', im_ihx_1_12.shape)

im_ihx_1_12 = im_ihx_1_12[40:1260, 40:1260]
im_ihx_1_12[im_ihx_1_12 < 0] = 0

# print('im_ihx_1_12', im_ihx_1_12.shape)

plt.imshow(im_ihx_1_12)
plt.colorbar()
plt.show()

plt.plot(im_ihx_1_12[650])
plt.plot(sp.ndimage.gaussian_filter(im_ihx_1_12[650], sigma=2))
plt.show()

print(np.mean(im_ihx_1_12[650, 200:1000]))

# %%
bim_ihx_1_12 = (gaussian(im_ihx_1_12, sigma=3) > 0.1).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_ihx_1_12))
ax[1].imshow(bim_ihx_1_12)
plt.show()

# %%
out_im = fill_im_with_poly_mu(poly_ihx_1_12_mu_at_depth_eff,bim_ihx_1_12)
# plt.imshow(out_im, norm=LogNorm())
plt.imshow(out_im)
plt.colorbar()
plt.show()

# %%
naive_recon_obj = naive_reconstruction(bim_ihx_1_12, poly_ihx_1_12_mu_at_depth)
naive_recon_obj_eff = naive_reconstruction(bim_ihx_1_12, poly_ihx_1_12_mu_at_depth_eff)

vmax = np.max(np.maximum(np.maximum(im_ihx_1_12, naive_recon_obj), naive_recon_obj_eff))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(im_ihx_1_12, vmax=vmax)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(naive_recon_obj, vmax=vmax)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(naive_recon_obj_eff, vmax=vmax)
plt.colorbar(im2, ax=ax[2])
plt.show()

show_porous_profiles([im_ihx_1_12, naive_recon_obj], im_ihx_1_12.shape[0]//2)
show_porous_profiles([im_ihx_1_12, naive_recon_obj_eff], im_ihx_1_12.shape[0]//2)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spec_Mo_45, iohexol_mu_1_12, voxel_size, GOS_eff_45)

# %% [markdown]
# ## Water/Iohexol 1/1

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/cd130f11-38b7-4de8-ad96-85f30f8a6105.npy'
with open(input_path, 'rb') as f:
  im_ihx_1_1 = np.load(f)

# print('im_ihx_1_1', im_ihx_1_1.shape)

im_ihx_1_1 = im_ihx_1_1[40:1260, 40:1260]
im_ihx_1_1[im_ihx_1_1 < 0] = 0

# print('im_ihx_1_1', im_ihx_1_1.shape)

plt.imshow(im_ihx_1_1)
plt.colorbar()
plt.show()

plt.plot(im_ihx_1_1[650])
plt.plot(sp.ndimage.gaussian_filter(im_ihx_1_1[650], sigma=2))
plt.show()

print(np.mean(im_ihx_1_1[650, 200:1000]))

# %%
bim_ihx_1_1 = (gaussian(im_ihx_1_1, sigma=3) > 0.15).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_ihx_1_1))
ax[1].imshow(bim_ihx_1_1)
plt.show()

# %%
h_c, o_c, d = calc_Ihx_solution(0.85, 1)

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_1_1_mu = xraydb.material_mu('iohexol', spec_Mo_45_energies*1000) / 10
plt.plot(spec_Mo_45_energies, iohexol_1_1_mu)
plt.yscale('log')
# plt.xscale('log')
plt.show()

# spec_MoKa_idx = 348 # 348 — index for 17.479 keV
iohexol_MoKa_mu = iohexol_1_1_mu[spec_45_MoKa_idx]
print(iohexol_MoKa_mu)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spec_Mo_45, iohexol_1_1_mu, voxel_size, GOS_eff_45)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), effective_spec_Mo_45, iohexol_1_1_mu, voxel_size)

# %% [markdown]
# ## **generate SpekPy spectrum**

# %%
en_step = np.mean(spec_Mo_45_energies[1:] - spec_Mo_45_energies[:-1])
en_step

# %%
s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
idx_min = np.where(spec_Mo_45_energies < energies[0])[0][-1]
idx_max = np.where(spec_Mo_45_energies > energies[-1])[0][0]
spec_Mo_45_energies_1 = spec_Mo_45_energies[idx_min:idx_max]
spec_Mo_45_1 = spec_Mo_45[idx_min:idx_max]
spec_Mo_45_1 /= spec_Mo_45_1.sum()

# %%
plt.plot(energies, intensities)
plt.plot(spec_Mo_45_energies_1, spec_Mo_45_1)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
spekpy_Mo_45_energies = np.copy(energies)
spekpy_Mo_45 = np.copy(intensities)

# %% [markdown]
# ## **GOS efficiensy**

# %%
GOS_mus_45_spekpy = xraydb.material_mu('GOS', spekpy_Mo_45_energies * 1000) / 10
GOS_t_45_spekpy = np.exp(-GOS_mus_45_spekpy * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV

GOS_n_p_45_spekpy = spekpy_Mo_45_energies * qe
# GOS_n_p_45 = np.ones(spekpy_Mo_45_energies.size)

GOS_eff_45_spekpy = GOS_n_p_45_spekpy * (1 - GOS_t_45_spekpy)

plt.plot(spekpy_Mo_45_energies, GOS_eff_45_spekpy)
plt.grid()
plt.show()

# %%
# h_c, o_c, d = calc_Ihx_solution(5, 60) #4.17:60
h_c, o_c, d = calc_Ihx_solution(4.5, 60) #4.17:60

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_1_12_mu_spekpy = xraydb.material_mu('iohexol', spekpy_Mo_45_energies*1000) / 10
plt.plot(spekpy_Mo_45_energies, iohexol_1_12_mu_spekpy)
plt.yscale('log')
plt.show()

spekpy_MoKa_idx = 348 # 348 — index for 17.479 keV
iohexol_MoKa_mu_spekpy = iohexol_1_12_mu_spekpy[spekpy_MoKa_idx]
print(iohexol_MoKa_mu_spekpy)

# %%
_ = calc_object_mus_from_spectrum(
    bim_ihx_1_12, 
    gaussian(im_ihx_1_12), 
    spekpy_Mo_45, 
    iohexol_1_12_mu_spekpy, 
    voxel_size, 
    GOS_eff_45_spekpy
)

# %%
h_c, o_c, d = calc_Ihx_solution(0.85, 1)

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_1_1_mu_spekpy = xraydb.material_mu('iohexol', spekpy_Mo_45_energies*1000) / 10
plt.plot(spekpy_Mo_45_energies, iohexol_1_1_mu_spekpy)
plt.yscale('log')
# plt.xscale('log')
plt.show()

iohexol_MoKa_mu_spekpy = iohexol_1_1_mu_spekpy[spekpy_MoKa_idx]
print(iohexol_MoKa_mu_spekpy)

# %%
_ = calc_object_mus_from_spectrum(
    bim_ihx_1_1, 
    gaussian(im_ihx_1_1), 
    spekpy_Mo_45, 
    iohexol_1_1_mu_spekpy, 
    voxel_size, 
    GOS_eff_45_spekpy
)

# %%
whos

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## **Generate Mo 50keV spectrum**

# %%
en_step = np.mean(spec_Mo_50_energies[1:] - spec_Mo_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.plot(spec_Mo_50_energies, spec_Mo_50_0)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
idx_min = 0
idx_max = np.where(spec_Mo_50_energies > energies[-1])[0][0]
spec_Mo_50_energies_1 = spec_Mo_50_energies[idx_min:idx_max]
spec_Mo_50_0_1 = spec_Mo_50_0[idx_min:idx_max]
spec_Mo_50_0_1 /= spec_Mo_50_0_1.sum()

plt.plot(energies, intensities)
plt.plot(spec_Mo_50_energies_1, spec_Mo_50_0_1)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
spekpy_Mo_50_energies = np.copy(energies)
spekpy_Mo_50 = np.copy(intensities)

# %%
GOS_mus_50 = xraydb.material_mu('GOS', spekpy_Mo_50_energies * 1000) / 10
GOS_t_50 = np.exp(-GOS_mus_50 * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV

GOS_n_p_50 = spekpy_Mo_50_energies * qe
# GOS_n_p_50 = np.ones(spekpy_Mo_50_energies.size)

GOS_eff_50 = GOS_n_p_50 * (1 - GOS_t_50)

plt.plot(spekpy_Mo_50_energies, GOS_eff_50)
plt.grid()
plt.show()

# %%
att_SiC_0 = xraydb.material_mu('SiC', spekpy_Mo_50_energies * 1000) / 10
plt.plot(spekpy_Mo_50_energies, att_SiC_0)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

# %%
_, recon_GOS = calc_object_mus_from_spectrum(bim_SiC, gaussian(im_SiC_0), spekpy_Mo_50, att_SiC_0, voxel_size, GOS_eff_50, h_line_SiC)

# %%
plt.figure(figsize=(10, 5))
plt.plot(im_SiC_0[h_line_SiC], label='Эксперимент', linewidth=0.5)
plt.plot(recon_SiC_50_0_eff[h_line_SiC], label='Моделирование 1', linewidth=0.5)
plt.plot(recon_GOS[h_line_SiC], label='Моделирование 2', linewidth=0.5)
plt.xlabel('voxels')
plt.ylabel('Attenuation µ, 1/mm')
plt.legend()
plt.grid()
plt.show()

# %%
default_blue_color = u'#1f77b4'

# plt.figure(figsize=(10, 5))
# plt.plot(im_SiC_0[h_line_SiC], label='Эксперимент', c='black', linewidth=0.5)
# plt.plot(recon_SiC_50_0_eff[h_line_SiC], label='Моделирование', c=default_blue_color, linewidth=0.75, linestyle=(0, (7.5, 5)))
# plt.xlabel('воксели')
# plt.ylabel('Коэффициент ослабления µ, 1/мм')
# plt.legend()
# plt.grid()
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(im_SiC_0[h_line_SiC], label='Эксперимент', c='black', linewidth=0.5)
# plt.plot(recon_SiC_50_0_eff[h_line_SiC], label='Моделирование 1')
plt.plot(recon_GOS[h_line_SiC], label='Моделирование 2', c=default_blue_color, linewidth=0.75, linestyle=(0, (7.5, 5)))
plt.xlabel('воксели')
plt.ylabel('Коэффициент ослабления µ, 1/мм')
plt.legend()
plt.grid()
plt.show()

# %%
en_step = np.mean(spec_Mo_50_energies[1:] - spec_Mo_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
s.filter('Al', 1.8)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.plot(spekpy_Mo_50_energies, spekpy_Mo_50)
plt.plot(spec_Mo_50_energies, spec_Mo_50_18)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
spekpy_Mo_50_18 = np.copy(intensities)

# %%
_ = calc_object_mus_from_spectrum(bim_SiC, gaussian(im_SiC_18), spekpy_Mo_50_18, att_SiC_0, voxel_size, GOS_eff_50, h_line_SiC)

# %%
en_step = np.mean(spec_Mo_50_energies[1:] - spec_Mo_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
s.filter('Al', 3.24)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
plt.plot(spekpy_Mo_50_energies, spekpy_Mo_50)
plt.plot(spekpy_Mo_50_energies, spekpy_Mo_50_18)
plt.plot(spec_Mo_50_energies, spec_Mo_50_324)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

spekpy_Mo_50_324 = np.copy(intensities)

# %%
_ = calc_object_mus_from_spectrum(bim_SiC, gaussian(im_SiC_324), spekpy_Mo_50_324, att_SiC_0, voxel_size, GOS_eff_50, h_line_SiC)

# %%

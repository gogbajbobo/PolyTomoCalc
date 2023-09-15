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
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from skimage.io import imread

from spec_gen import generate_spectrum
import spekpy


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

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_bh_corr_2.57_optimal.npy'
with open(input_path, 'rb') as f:
  bim_SiC = np.load(f)

bim_SiC = gaussian(median(bim_SiC)) > 2

plt.imshow(bim_SiC)
plt.show()

# %%
h_line_SiC = 550

def process_porous_pic(im, bim):
    im = gaussian(median(im))
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

# %%
# Mo_spec_*.npy should be created in spectrum_analysis.ipynb first

input_path = 'Mo_spec_poly_50_energies.npy'
with open(input_path, 'rb') as f:
  spec_Mo_50_energies = np.load(f)

input_path = 'Mo_spec_poly_50.npy'
with open(input_path, 'rb') as f:
  spec_Mo_50_0 = np.load(f)

input_path = 'Mo_spec_poly_50_18.npy'
with open(input_path, 'rb') as f:
  spec_Mo_50_18 = np.load(f)

input_path = 'Mo_spec_poly_50_324.npy'
with open(input_path, 'rb') as f:
  spec_Mo_50_324 = np.load(f)

plt.plot(spec_Mo_50_energies, spec_Mo_50_0, label='no filter')
plt.plot(spec_Mo_50_energies, spec_Mo_50_18, label='Al filter, 1.8 mm')
plt.plot(spec_Mo_50_energies, spec_Mo_50_324, label='Al filter, 3.24 mm')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

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
xraydb.add_material('SiC','SiC', 3.21)
print(f"SiC attenuation at MoKa1: { xraydb.material_mu('SiC', Mo_lines['Ka1'].energy) / 10 }")

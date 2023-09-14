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
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_beta_0)
ax[1].set_title('MoKβ', fontsize=18)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(im_poly_0)
ax[2].set_title('Poly', fontsize=18)
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

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

bim_SiC = gaussian(median(bim_SiC))
thresh = threshold_otsu(bim_SiC)
print(f'thresh: {thresh}')
bim_SiC = (bim_SiC > thresh).astype(int)

# bim_SiC = gaussian(median(bim_SiC)) > 2

plt.imshow(bim_SiC)
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
xraydb.add_material('SiC','SiC', 3.21)
print(f"SiC attenuation at MoKa1: { xraydb.material_mu('SiC', Mo_lines['Ka1'].energy) / 10 }")

# %%
att_SiC = xraydb.material_mu('SiC', spec_Mo_50_energies * 1000) / 10

plt.plot(spec_Mo_50_energies, att_SiC)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

# %%
exp_mu_depth = np.exp(np.outer(-att_SiC, [0, 0.1, 0.3, 1.0])).T

plt.plot(exp_mu_depth.T)
plt.show()

# %%
passed_spectrums = exp_mu_depth * spec_Mo_50_0

plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, passed_spectrums.T)
plt.ylim(1e-5, 1e-1)
plt.yscale('log')
plt.grid()
plt.show()

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm
total_lenght = 1 # cm
length_ticks = np.arange(0, total_lenght, voxel_size)

exp_mu_depth = np.exp(np.outer(-att_SiC, length_ticks)).T
passed_spectrums = exp_mu_depth * spec_Mo_50_0

p_specs_norm = (passed_spectrums.T / np.sum(passed_spectrums, axis=1)).T
plt.plot(length_ticks, np.sum(passed_spectrums, axis=1))
plt.plot(length_ticks, np.sum(p_specs_norm, axis=1))

print(p_specs_norm.shape)

# %%
poly_mu_map = p_specs_norm * att_SiC
poly_mu_depth = np.sum(poly_mu_map, axis=1)
print(poly_mu_map.shape)
print(poly_mu_depth.shape)

plt.plot(length_ticks, poly_mu_depth)
# plt.yscale('log')
plt.grid()
plt.show()


# %%
fill_values = poly_mu_depth
out_im = np.zeros(bim_SiC.shape)

for y, bim in enumerate(bim_SiC):
  for i, x in enumerate(np.where(bim > 0)[0]):
    out_im[y, x] = fill_values[i] * bim[x]

plt.imshow(out_im)
plt.colorbar()

# %%
att_obj_sum = np.zeros(bim_SiC.shape)

angles = np.arange(0, 360, 18)

for angle in angles:
  r_im = rotate(bim_SiC, angle, order=1, preserve_range=True)
  out_im = np.zeros(r_im.shape)
  for y, rim in enumerate(r_im):
    for i, x in enumerate(np.where(rim > 0)[0]):
      out_im[y, x] = fill_values[i] * rim[x]
  att_obj_sum += rotate(out_im, -angle, order=1, preserve_range=True)

att_obj_sum /= angles.size

# %%
att_obj_sum[bim_SiC == 0] = 0

vmax = np.max(np.maximum(att_obj_sum, im_SiC_0))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(att_obj_sum, vmax=vmax)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_0, vmax=vmax)
plt.colorbar(im1, ax=ax[1])
plt.show()


# %%
def show_porous_profiles(images, h_profile_line):

    plt.figure(figsize=(10, 5))
    for im in images:
        plt.plot(im[h_profile_line])
    plt.grid()
    plt.show()


show_porous_profiles([im_SiC_0, att_obj_sum], h_line_SiC)

# %%
angles = np.arange(0, 180, 1)
sinogram = np.zeros((angles.size, bim_SiC.shape[0]))
proj_sums = np.zeros(angles.size)

for j, angle in enumerate(angles):
  r_im = rotate(bim_SiC, angle, order=1, preserve_range=True)
  out_im = np.zeros(r_im.shape)
  for y, rim in enumerate(r_im):
    for i, x in enumerate(np.where(rim > 0)[0]):
      out_im[y, x] = fill_values[i] * rim[x]
  sinogram[j] = np.sum(out_im, axis=1)
  proj_sums[j] = np.sum(sinogram[j])

plt.imshow(sinogram, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(proj_sums)
plt.show()

# %%
recon = gaussian(iradon(sinogram.T, theta=angles).T)

recon[bim_SiC == 0] = 0
recon[recon < 0] = 0

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(recon)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(im_SiC_0)
plt.colorbar(im1, ax=ax[1])
plt.show()

# %%
show_porous_profiles([im_SiC_0, recon], h_line_SiC)

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus = xraydb.material_mu('GOS', spec_Mo_50_energies * 1000) / 10
GOS_t = np.exp(-GOS_mus * 22 * 0.001) # (22 * 0.001)mm == 22µm

# %%
beta = 3
en_gap = 4.6 # eV
GOS_n_p = spec_Mo_50_energies * 1000 / (beta * en_gap)
# GOS_n_p = np.ones(spec_Mo_50_energies.size)

GOS_eff = GOS_n_p * (1 - GOS_t)

plt.plot(spec_Mo_50_energies, GOS_eff)
plt.grid()
plt.show()

# %%
qe = 60 # photon/keV

GOS_n_p = spec_Mo_50_energies * qe
# GOS_n_p = np.ones(spec_Mo_50_energies.size)

GOS_eff = GOS_n_p * (1 - GOS_t)

plt.plot(spec_Mo_50_energies, GOS_eff)
plt.grid()
plt.show()


# %%
def calc_object_mus_from_spectrum(bin_im, exp_im, spectrum, energies, mat_att, voxel_size, GOS_eff, h_line):

    angles = np.arange(0, 180, 1)
    sino_shape = (angles.size, bin_im.shape[0])
    sinogram_wo_GOS = np.zeros(sino_shape)
    sinogram_GOS = np.zeros(sino_shape)
    
    current_spec_GOS = spectrum * GOS_eff
    
    for j, angle in enumerate(angles):
        
        r_im = rotate(bin_im, angle, order=1, preserve_range=True)
        ray_sums = np.sum(r_im, axis=1) * voxel_size
        trans_sums = np.exp(np.outer(-mat_att, ray_sums)).T
        
        passed_specs_wo_GOS = trans_sums * spectrum
        passed_specs_GOS = passed_specs_wo_GOS * GOS_eff
        passed_specs_GOS /= current_spec_GOS.sum()
        
        passed_intensity_wo_GOS = np.sum(passed_specs_wo_GOS, axis=1)
        passed_intensity_GOS = np.sum(passed_specs_GOS, axis=1)
        
        sinogram_wo_GOS[j] = -np.log(passed_intensity_wo_GOS)
        sinogram_GOS[j] = -np.log(passed_intensity_GOS)
        
    recon_wo_GOS = gaussian(iradon(sinogram_wo_GOS.T, theta=angles).T)
    recon_wo_GOS[bin_im == 0] = 0
    recon_wo_GOS[recon_wo_GOS < 0] = 0
    recon_wo_GOS /= voxel_size # convert to 1/mm values
    
    recon_GOS = gaussian(iradon(sinogram_GOS.T, theta=angles).T)
    recon_GOS[bin_im == 0] = 0
    recon_GOS[recon_GOS < 0] = 0
    recon_GOS /= voxel_size # convert to 1/mm values
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(recon_wo_GOS)
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(recon_GOS)
    plt.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(exp_im)
    plt.colorbar(im2, ax=ax[2])
    plt.show()
    
    row = h_line or bin_im.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(recon_wo_GOS[row], linewidth=.5, c='green')
    plt.plot(recon_GOS[row], linewidth=.5)
    
    # exp_row = np.sum(exp_im[row-10:row+10], axis=0) / exp_im[row-10:row+10].shape[0]
    exp_row = exp_im[row]
    
    plt.plot(exp_row, linewidth=1, c='gray')
    plt.ylabel('Коэффициент ослабления, 1/мм')
    plt.grid(color='gray')
    plt.show()
    
    return recon_wo_GOS, recon_GOS



# %%
_ = calc_object_mus_from_spectrum(bim_SiC, im_SiC_0, spec_Mo_50_0, spec_Mo_50_energies, att_SiC, voxel_size, GOS_eff, h_line_SiC)

# %%

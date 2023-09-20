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

# %% [markdown]
# ## **SiC attenuation**

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
poly_mu_SiC = (att_SiC * spec_Mo_50_0).sum()
poly_mu_SiC

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
voxel_size = 0.009 # in mm — 0.01 = 10µm
total_lenght = 10 # 1cm
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

# %% [markdown]
# ## **Naive reconstruction**

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

# %% [markdown]
# ## **µ-filled reconstruction**

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

# %% [markdown]
# ## **Gadolinium oxysulfide**

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


# %% [markdown]
# ## **Reconstruction with spectrum attenuation calculation and GOS**

# %%
def calc_object_mus_from_spectrum(bin_im, exp_im, spectrum, mat_att, voxel_size, GOS_eff, h_line=None):

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
_ = calc_object_mus_from_spectrum(bim_SiC, im_SiC_0, spec_Mo_50_0, att_SiC, voxel_size, GOS_eff, h_line_SiC)

# %% [markdown]
# ## **Iohexol samles**
#
# Mo-anode 45keV spectrum

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



# %%
h_c, o_c, d = calc_Ihx_solution(5, 60) #4.17:60

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_mu = xraydb.material_mu('iohexol', spec_Mo_45_energies*1000) / 10
plt.plot(spec_Mo_45_energies, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])

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

# %% [markdown]
# ## **Ihx samples**

# %% [markdown]
# ## Water/Iohexol 1/12

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
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spec_Mo_45, iohexol_mu, voxel_size, GOS_eff_45)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spec_Mo_45, iohexol_mu, voxel_size, (1 - GOS_t_45))

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
bim_ihx_1_12_mono = (gaussian(im_ihx_1_12_mono, sigma=3) > 0.1).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_ihx_1_12_mono))
ax[1].imshow(bim_ihx_1_12_mono)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12_mono, gaussian(im_ihx_1_12_mono), np.array([1]), 0.1956, voxel_size, np.array([1]))

# %% [markdown]
# ## Water/Iohexol 1/1

# %%
h_c, o_c, d = calc_Ihx_solution(1, 1)

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_mu = xraydb.material_mu('iohexol', spec_Mo_45_energies*1000) / 10
plt.plot(spec_Mo_45_energies, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])

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
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spec_Mo_45, iohexol_mu, voxel_size, GOS_eff_45)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spec_Mo_45, iohexol_mu, voxel_size, (1 - GOS_t_45))

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/82fc7477-dafb-4950-aea5-6e522910181d.npy'
with open(input_path, 'rb') as f:
  im_ihx_1_1_mono = np.load(f)

# print('im_ihx_1_1_mono', im_ihx_1_1_mono.shape)

im_ihx_1_1_mono = im_ihx_1_1_mono[40:1260, 40:1260]
im_ihx_1_1_mono[im_ihx_1_1_mono < 0] = 0

# print('im_ihx_1_1_mono', im_ihx_1_1_mono.shape)

plt.imshow(im_ihx_1_1_mono)
plt.colorbar()
plt.show()

plt.plot(im_ihx_1_1_mono[650])
plt.plot(sp.ndimage.gaussian_filter(im_ihx_1_1_mono[650], sigma=2))
plt.show()

print(np.mean(im_ihx_1_1_mono[650, 200:1000]))

# %%
bim_ihx_1_1_mono = (gaussian(im_ihx_1_1_mono, sigma=3) > 0.15).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_ihx_1_1_mono))
ax[1].imshow(bim_ihx_1_1_mono)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1_mono, gaussian(im_ihx_1_1_mono), np.array([1]), 0.656038, voxel_size, np.array([1]))

# %% [markdown]
# ## **generate SpekPy spectrum**

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
GOS_mus_45 = xraydb.material_mu('GOS', spekpy_Mo_45_energies * 1000) / 10
GOS_t_45 = np.exp(-GOS_mus_45 * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV

GOS_n_p_45 = spekpy_Mo_45_energies * qe
# GOS_n_p_45 = np.ones(spekpy_Mo_45_energies.size)

GOS_eff_45 = GOS_n_p_45 * (1 - GOS_t_45)

plt.plot(spekpy_Mo_45_energies, GOS_eff_45)
plt.grid()
plt.show()

# %%
h_c, o_c, d = calc_Ihx_solution(5, 60) #4.17:60

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_1_12_mu = xraydb.material_mu('iohexol', spekpy_Mo_45_energies*1000) / 10
plt.plot(spekpy_Mo_45_energies, iohexol_1_12_mu)
plt.yscale('log')
plt.show()

spekpy_MoKa_idx = 348 # 348 — index for 17.479 keV
iohexol_MoKa_mu = iohexol_1_12_mu[spekpy_MoKa_idx]
print(iohexol_MoKa_mu)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spekpy_Mo_45, iohexol_1_12_mu, voxel_size, GOS_eff_45)

# %%
h_c, o_c, d = calc_Ihx_solution(1, 1)

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_c} O{o_c}', d)
iohexol_1_1_mu = xraydb.material_mu('iohexol', spekpy_Mo_45_energies*1000) / 10
plt.plot(spekpy_Mo_45_energies, iohexol_1_1_mu)
plt.yscale('log')
plt.show()

spekpy_MoKa_idx = 348 # 348 — index for 17.479 keV
iohexol_MoKa_mu = iohexol_1_1_mu[spekpy_MoKa_idx]
print(iohexol_MoKa_mu)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spekpy_Mo_45, iohexol_1_1_mu, voxel_size, GOS_eff_45)

# %% [markdown]
# ## **Generate SpekPy w/o bremsstrahlung**

# %%
s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo', brem=False)
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
# plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

spekpy_Mo_45_wo_brem_energies = np.copy(energies)
spekpy_Mo_45_wo_brem = np.copy(intensities)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spekpy_Mo_45_wo_brem, iohexol_1_12_mu, voxel_size, GOS_eff_45)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spekpy_Mo_45_wo_brem, iohexol_1_1_mu, voxel_size, GOS_eff_45)

# %% [markdown]
# ## **Generate SpekPy w/o characteristic lines**

# %%
s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo', char=False)
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)
# plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

spekpy_Mo_45_wo_char_energies = np.copy(energies)
spekpy_Mo_45_wo_char = np.copy(intensities)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_12, gaussian(im_ihx_1_12), spekpy_Mo_45_wo_char, iohexol_1_12_mu, voxel_size, GOS_eff_45)

# %%
_ = calc_object_mus_from_spectrum(bim_ihx_1_1, gaussian(im_ihx_1_1), spekpy_Mo_45_wo_char, iohexol_1_1_mu, voxel_size, GOS_eff_45)

# %%

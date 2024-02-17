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

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from _handler_funcs import generate_spectrum

import spekpy


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
en_step = np.mean(spec_Mo_45_energies[1:] - spec_Mo_45_energies[:-1])

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 1440)
model_energies, model_intensities = s.get_spectrum()
model_intensities /= model_intensities.sum()

plt.plot(model_energies, model_intensities)
plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.yscale('log')
plt.ylim(1e-5, 1)
plt.grid()
plt.show()

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/cd130f11-38b7-4de8-ad96-85f30f8a6105.npy'
with open(input_path, 'rb') as f:
  im = np.load(f)

print('im', im.shape)

im = im[40:1260, 40:1260]

print('im', im.shape)

plt.imshow(im)
plt.colorbar()
plt.show()

plot_row = im.shape[0] // 2

plt.plot(im[plot_row])
plt.plot(sp.ndimage.gaussian_filter(im[plot_row], sigma=5))
plt.show()

print(np.mean(im[plot_row, 200:1000]))

# %%
bim = gaussian(im, sigma=3) > 0.15

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im)
ax[1].imshow(bim)


# %%
def calc_object_mus_from_spectrum(bin_im, exp_im, spectrum, energies, mat_att, voxel_size, diffraction_loss=0):

  bin_im = bin_im.astype(int)

  exp_im[bin_im == 0] = 0
  exp_im[exp_im < 0] = 0

  print('bin_im', bin_im.shape)
  print('im', im.shape)

  xraydb.add_material('GOS', 'Gd2O2S', 7.34)
  GOS_mus = xraydb.material_mu('GOS', energies)
  GOS_t = np.exp(-GOS_mus * 22 * 0.0001) # (22 * 0.0001)cm == 22µm
  # GOS_t = np.exp(-GOS_mus * 88 * 0.0001)
  beta = 3
  en_gap = 4.6 # eV

  GOS_n_p = energies / (beta * en_gap)
  # GOS_n_p = np.ones(energies.size)

  GOS_eff = GOS_n_p * (1 - GOS_t)

  angles = np.arange(0, 180, 1)
  sino_shape = (angles.size, bin_im.shape[0])
  sinogram_wo_GOS = np.zeros(sino_shape)
  sinogram_GOS = np.zeros(sino_shape)
  # sinogram_GOS_diff = np.zeros(sino_shape)

  # Be_mus = xraydb.material_mu('beryllium', energies)
  # Be_t = np.exp(-Be_mus * 0.05)

  # Xe_mus = xraydb.material_mu('xenon', energies)
  # Xe_t = np.exp(-Xe_mus * 0.2)

  # current_spec_GOS = spectrum * Be_t
  # current_spec_GOS *= Xe_t
  # current_spec_GOS *= GOS_eff

  current_spec_GOS = spectrum * GOS_eff

  for j, angle in enumerate(angles):

    r_im = rotate(bin_im, angle, order=1, preserve_range=True)
    ray_sums = np.sum(r_im, axis=1) * voxel_size
    trans_sums = np.exp(np.outer(-mat_att, ray_sums)).T
    # diff_t = np.full(ray_sums.shape, 1 - diffraction_loss)
    # diff_t[ray_sums == 0] = 1

    passed_specs_wo_GOS = trans_sums * spectrum
    # passed_specs_wo_GOS *= Be_t
    # passed_specs_wo_GOS *= Xe_t
    passed_specs_GOS = passed_specs_wo_GOS * GOS_eff
    # passed_specs_GOS_diff = (passed_specs_wo_GOS.T * diff_t).T * GOS_eff
    passed_specs_GOS /= current_spec_GOS.sum()
    # passed_specs_GOS_diff /= current_spec_GOS.sum()

    passed_intensity_wo_GOS = np.sum(passed_specs_wo_GOS, axis=1)
    passed_intensity_GOS = np.sum(passed_specs_GOS, axis=1)
    # passed_intensity_GOS_diff = np.sum(passed_specs_GOS_diff, axis=1)

    sinogram_wo_GOS[j] = -np.log(passed_intensity_wo_GOS)
    sinogram_GOS[j] = -np.log(passed_intensity_GOS)
    # sinogram_GOS_diff[j] = -np.log(passed_intensity_GOS_diff)

  recon_wo_GOS = gaussian(iradon(sinogram_wo_GOS.T, theta=angles).T)
  recon_wo_GOS[bin_im == 0] = 0
  recon_wo_GOS[recon_wo_GOS < 0] = 0
  recon_wo_GOS /= voxel_size * 10 # convert to 1/mm values
  # recon_wo_GOS *= 2

  recon_GOS = gaussian(iradon(sinogram_GOS.T, theta=angles).T)
  recon_GOS[bin_im == 0] = 0
  recon_GOS[recon_GOS < 0] = 0
  recon_GOS /= voxel_size * 10 # convert to 1/mm values
  # recon_GOS *= 2

  # recon_GOS_diff = gaussian(iradon(sinogram_GOS_diff.T, theta=angles).T)
  # recon_GOS_diff[bin_im == 0] = 0
  # recon_GOS_diff[recon_GOS_diff < 0] = 0
  # recon_GOS_diff /= voxel_size * 10 # convert to 1/mm values

  fig, ax = plt.subplots(1, 4, figsize=(20, 5))
  im0 = ax[0].imshow(recon_wo_GOS)
  # plt.colorbar(im0, ax=ax[0])
  im1 = ax[1].imshow(recon_GOS)
  plt.colorbar(im1, ax=ax[1])
  # im2 = ax[2].imshow(recon_GOS_diff)
  # plt.colorbar(im2, ax=ax[2])
  im3 = ax[3].imshow(exp_im)
  plt.colorbar(im3, ax=ax[3])
  plt.show()

  row = bin_im.shape[0] // 2
  plt.figure(figsize=(10, 5))
  plt.plot(recon_wo_GOS[row], linewidth=.5, c='green')
  plt.plot(recon_GOS[row], linewidth=.5)
  # plt.plot(recon_GOS_diff[row], linewidth=.5, c='red')

  exp_row = np.sum(exp_im[row-10:row+10], axis=0) / exp_im[row-10:row+10].shape[0]

  plt.plot(exp_row, linewidth=1, c='gray')
  plt.ylabel('Коэффициент ослабления, 1/мм')
  plt.grid(color='gray')
  plt.show()

  return recon_wo_GOS, recon_GOS#, recon_GOS_diff



# %%
ihx_d = 2.2 #g/cm3
wat_d = 0.997
ihx_w = 0.647 #g
ihx_d_s1 = ihx_w + wat_d * (1 - ihx_w/ihx_d)
print(ihx_d_s1)

ihx_p = 1
wat_p = 1
ihx_d_s2 = (ihx_d_s1 * ihx_p + wat_d * wat_p) / (ihx_p + wat_p)

print('solution density:', ihx_d_s2)

ihx_mm = 821.144 #g/mol
wat_mm = 18.01528

ihx_m = ihx_w * ihx_p / ihx_mm # mol
wat_m = (wat_d * (1 - ihx_w/ihx_d) * ihx_p + wat_d * wat_p) / wat_mm
print(ihx_m)
print(wat_m)

m_ratio = wat_m / ihx_m
print(m_ratio)
h_i = int(np.rint(m_ratio * 2))
o_i = int(np.rint(m_ratio))

print('H2O:', h_i, o_i)

# %%
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(spec_Mo_45.shape[0])])

print(en_keV)

xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_i} O{o_i}', ihx_d_s2)
iohexol_mu = xraydb.material_mu('iohexol', en_keV*1000) / 10
plt.plot(en_keV, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

_, m_im = calc_object_mus_from_spectrum(bim, gaussian(im), spec_Mo_45, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[1, 0.8, 1])

im2show = gaussian(im, sigma=3)
im2show[im2show < 0] = 0
im0 = ax[0].imshow(im2show)
ax[0].axhline(im2show.shape[0] // 2, linewidth=4, c='white', alpha=0.5)
plt.colorbar(im0, ax=ax[0])
ax[0].set_title('а', fontsize=24)

im1 = ax[1].imshow(bim)
ax[1].set_title('б', fontsize=24)

im2 = ax[2].imshow(m_im)
plt.colorbar(im2, ax=ax[2])
ax[2].set_title('в', fontsize=24)

plt.show()

# %%
row = bim.shape[0] // 2

plt.figure(figsize=(10, 5))

exp_row = np.sum(im2show[row-10:row+10], axis=0) / im2show[row-10:row+10].shape[0]
plt.plot(exp_row, c='black', linewidth=0.5, label='Эксперимент')

plt.plot(m_im[row], linewidth=0.75, linestyle=(0, (7.5, 5)), label='Моделирование')

plt.xlabel('воксели')
plt.ylabel('Коэффициент ослабления, 1/мм')

plt.grid()
plt.legend()

plt.show()

# %%
iohexol_mu_2 = xraydb.material_mu('iohexol', model_energies*1000) / 10
plt.plot(model_energies, iohexol_mu_2)
plt.yscale('log')
plt.show()

print(iohexol_mu_2[371])

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

_, m_im_2 = calc_object_mus_from_spectrum(bim, gaussian(im), model_intensities, model_energies*1000, iohexol_mu_2*10, voxel_size)

# %%
row = bim.shape[0] // 2

plt.figure(figsize=(10, 5))

exp_row = np.sum(im2show[row-10:row+10], axis=0) / im2show[row-10:row+10].shape[0]
plt.plot(exp_row, c='black', linewidth=0.5, label='Эксперимент')

plt.plot(m_im_2[row], linewidth=0.75, linestyle=(0, (7.5, 5)), label='Моделирование 2')

plt.xlabel('воксели')
plt.ylabel('Коэффициент ослабления, 1/мм')

plt.grid()
plt.legend()

plt.show()

# %%

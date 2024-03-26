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
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian, median

import xraydb
import spekpy

import skimage

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/cd130f11-38b7-4de8-ad96-85f30f8a6105.npy'
with open(input_path, 'rb') as f:
  ihx_1_1 = np.load(f)

# %%
plt.imshow(ihx_1_1)
plt.colorbar()

# %%
plt.plot(ihx_1_1[ihx_1_1.shape[0]//2])

# %%
im2show = gaussian(ihx_1_1, sigma=3)
im2show[im2show < 0] = 0

row = im2show.shape[0]//2

plt.figure(figsize=(10, 5))

exp_row = np.sum(im2show[row-10:row+10], axis=0) / im2show[row-10:row+10].shape[0]
plt.plot(exp_row, c='black', linewidth=0.5, label='Эксперимент')

# plt.plot(im2show[row])

plt.grid()

# %%
bim = im2show > 0.15
plt.imshow(bim)

# %%
radius = 10
shape = (radius * 2, radius * 2)
img = np.zeros(shape, dtype=np.uint8)
rr, cc = skimage.draw.disk((radius, radius), radius, shape=shape)
img[rr, cc] = 1

bim_erroded = skimage.morphology.binary_erosion(bim, footprint=img)
plt.imshow(bim_erroded)


# %%
ihx_im = ihx_1_1.copy()
ihx_im[~bim_erroded] = 0
ihx_im[ihx_im < 0] = 0

plt.imshow(ihx_im)
plt.colorbar()

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm

# SiC_lengths = np.sum(np.fliplr(bim_SiC), axis=0) * voxel_size
# SiC_att_0 = np.sum(np.fliplr(im_SiC_0), axis=0) * voxel_size

ihx_lengths = np.sum(bim_erroded, axis=0) * voxel_size
ihx_att = np.sum(ihx_im, axis=0) * voxel_size

plt.scatter(ihx_lengths, ihx_att, marker='.', s=1, c='gray')
plt.grid()

# %%
# see spec_edit.ipynb to load, handle and save spectrum

input_path = 'Mo_spec_poly_45_0.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45_0 = np.load(f).astype(float)
    spec_Mo_45_0 /= spec_Mo_45_0.sum()

input_path = 'Mo_spec_poly_45_energies_0.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45_energies_0 = np.load(f).astype(float)

input_path = 'Mo_spec_poly_45.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45 = np.load(f).astype(float)
    # spec_Mo_45 /= spec_Mo_45.sum()

input_path = 'Mo_spec_poly_45_energies.npy'
with open(input_path, 'rb') as f:
    spec_Mo_45_energies = np.load(f).astype(float)

# en_step = (19.608 - 17.479) / (416 - 371)
# spec_Mo_45_energies = np.array([17.479 + (i - 371) * en_step for i in np.arange(spec_Mo_45.shape[0])])

plt.plot(spec_Mo_45_energies_0, spec_Mo_45_0)
plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.yscale('log')
plt.grid()
plt.show()

# %%
en_step = np.mean(spec_Mo_45_energies[1:] - spec_Mo_45_energies[:-1])

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 140)
model_energies_0, model_intensities_0 = s.get_spectrum()
model_intensities_0 /= model_intensities_0.sum()

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 1440)
model_energies, model_intensities = s.get_spectrum()
model_intensities /= model_intensities.sum()

plt.plot(model_energies, model_intensities)
plt.plot(model_energies_0, model_intensities_0)
plt.plot(spec_Mo_45_energies, spec_Mo_45)
plt.plot(spec_Mo_45_energies_0, spec_Mo_45_0)
plt.yscale('log')
plt.ylim(1e-5, 1)
plt.grid()
plt.show()

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
iohexol_mu_2 = xraydb.material_mu('iohexol', model_energies*1000) / 10
plt.plot(model_energies, iohexol_mu_2)
plt.yscale('log')
plt.show()

print(iohexol_mu_2[371])

# %%
default_blue_color = u'#1f77b4'

voxel_size = 0.1 # in mm
total_lenght = 11 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_IHX_at_depths = np.exp(np.outer(-iohexol_mu, length_ticks)).T

passed_spectrums_45_0 = transmissions_IHX_at_depths * spec_Mo_45_0
passed_intensity_0 = np.sum(passed_spectrums_45_0, axis=1)
attenuation_0 = -np.log(passed_intensity_0)

passed_spectrums_45 = transmissions_IHX_at_depths * spec_Mo_45
passed_intensity = np.sum(passed_spectrums_45, axis=1)
attenuation = -np.log(passed_intensity)

model_transmissions_IHX_at_depths = np.exp(np.outer(-iohexol_mu_2, length_ticks)).T

model_passed_spectrums_45_0 = model_transmissions_IHX_at_depths * model_intensities_0
model_passed_intensity_0 = np.sum(model_passed_spectrums_45_0, axis=1)
model_attenuation_0 = -np.log(model_passed_intensity_0)

model_passed_spectrums_45 = model_transmissions_IHX_at_depths * model_intensities
model_passed_intensity = np.sum(model_passed_spectrums_45, axis=1)
model_attenuation = -np.log(model_passed_intensity)

plt.scatter(ihx_lengths, ihx_att, marker='.', s=1, c='gray', label='Экспериментальные данные')

# plt.plot(length_ticks, attenuation_0, label='Моделирование: Экспериментальный спектр 0')

plt.plot(length_ticks, attenuation, label='Моделирование: Экспериментальный спектр')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o')

# plt.plot(length_ticks, model_attenuation_0, label='Моделирование: Расчётный спектр 0')

plt.plot(length_ticks, model_attenuation, linestyle=(0, (2, 1)), c=default_blue_color, label='Моделирование: Расчётный спектр')
plt.scatter(length_ticks[::10], model_attenuation[::10], facecolors='none', edgecolors=default_blue_color)

plt.xlabel('Толщина, мм', fontsize=14)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=14)
plt.xlim(-0.5, 9.2)
plt.grid()
plt.legend(framealpha=1)
plt.title(r'Йогексол, $C_{19}H_{26}I_3N_3O_9$')
# plt.savefig('Fig9b.eps', dpi=600)
plt.show()


# %%

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
import spekpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import numpy as np

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from skimage.draw import disk

# %%
s = spekpy.Spek(kvp=50, dk=0.5, targ='Mo')
s.filter('Air', 1440)
spek_energies, spek_intensities = s.get_spectrum()
spek_intensities /= spek_intensities.sum()

s.filter('Al', 10)
_, spek_intensities_Al_filter = s.get_spectrum()
spek_intensities_Al_filter /= spek_intensities_Al_filter.sum()

plt.plot(spek_energies, spek_intensities)
plt.plot(spek_energies, spek_intensities_Al_filter)
plt.yscale('log')
plt.ylim(1e-4, 1e0)
plt.grid()

# %%
Mo_lines = xraydb.xray_lines('Mo')

xraydb.add_material('SiC','SiC', 3.21)
print(f"SiC attenuation at MoKa1: { xraydb.material_mu('SiC', Mo_lines['Ka1'].energy) / 10 } 1/mm")
print(f"Fe attenuation at MoKa1: { xraydb.material_mu('Fe', Mo_lines['Ka1'].energy) / 10 } 1/mm")

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus_50 = xraydb.material_mu('GOS', spek_energies * 1000) / 10
GOS_t_50 = np.exp(-GOS_mus_50 * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV
GOS_n_p_50 = spek_energies * qe

GOS_eff_50 = GOS_n_p_50 * (1 - GOS_t_50)

plt.plot(spek_energies, GOS_eff_50)
plt.grid()
plt.show()

# %%
spek_intensities_eff = spek_intensities * GOS_eff_50
spek_intensities_eff /= spek_intensities_eff.sum()

spek_intensities_eff_Al_filter = spek_intensities_Al_filter * GOS_eff_50
spek_intensities_eff_Al_filter /= spek_intensities_eff_Al_filter.sum()

plt.plot(spek_energies, spek_intensities)
plt.plot(spek_energies, spek_intensities_eff)
plt.plot(spek_energies, spek_intensities_Al_filter)
plt.plot(spek_energies, spek_intensities_eff_Al_filter)
plt.yscale('log')
plt.ylim(1e-4, 1e0)
plt.grid()
plt.show()

# %%
att_SiC = xraydb.material_mu('SiC', spek_energies * 1000) / 10
att_Fe = xraydb.material_mu('Fe', spek_energies * 1000) / 10
att_H2O = xraydb.material_mu('Water', spek_energies * 1000) / 10
att_Air = xraydb.material_mu('Air', spek_energies * 1000) / 10

plt.plot(spek_energies, att_SiC)
plt.plot(spek_energies, att_Fe)
plt.plot(spek_energies, att_H2O)
plt.plot(spek_energies, att_Air)
plt.yscale('log')
plt.grid()
plt.show()


# %%
def calc_object_mus_from_spectrum(im, spectrum, mat_att, voxel_size):

    angles = np.arange(0, 360, 1)
    sino_shape = (angles.size, im.shape[0])
    sinogram = np.zeros(sino_shape)
    
    for j, angle in enumerate(angles):
        
        # r_im = rotate(im, angle, order=1, preserve_range=True)
        r_im = rotate(im, angle, order=0, preserve_range=True)
        ray_sums = np.sum(r_im, axis=1) * voxel_size
        trans_sums = np.exp(np.outer(-mat_att, ray_sums)).T
        
        passed_specs = trans_sums * spectrum
        passed_intensity = np.sum(passed_specs, axis=1)
        sinogram[j] = -np.log(passed_intensity)
        
    recon = gaussian(iradon(sinogram.T, theta=angles).T)
    recon[im == 0] = 0
    recon[recon < 0] = 0
    recon /= voxel_size # convert to 1/mm values
    
    plt.imshow(recon)
    plt.colorbar()
    plt.show()
    
    row = im.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(recon[row])
    plt.ylabel('Коэффициент ослабления, 1/мм')
    plt.grid(color='gray')
    plt.show()
    
    return recon


# %%
size = 257
shape = (size, size)
im1 = np.zeros(shape).astype(int)
rr, cc = disk((size//2, size//2), size//3, shape=shape)
im1[rr, cc] = 1

plt.imshow(im1)

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm

recon_0 = calc_object_mus_from_spectrum(im1, spek_intensities_eff, att_SiC, voxel_size)

# %%
im2 = np.zeros(shape).astype(int)
rr, cc = disk((size//2, size//2), size//3, shape=shape)
im2[rr, cc] = 2

rr, cc = disk((size//3, size//2), size//24, shape=shape)
# rr, cc = disk((size//3, size//2), size//12, shape=shape)
# rr, cc = disk((size//4, size//2), size//6, shape=shape)
im2[rr, cc] = 1

rr, cc = disk((2 * size//3, size//2), size//24, shape=shape)
# rr, cc = disk((2 * size//3, size//2), size//12, shape=shape)
# rr, cc = disk((3 * size//4, size//2), size//6, shape=shape)
im2[rr, cc] = 1

plt.imshow(im2)

# %%
attenuations = np.array([att_Fe, att_SiC])
# attenuations = np.array([att_SiC, att_SiC])
attenuations.shape


# %%
def recon_multiple_matters(im, attenuations, spectrum, voxel_size):

    angles = np.arange(0, 360, 1)
    sino_shape = (angles.size, im.shape[0])
    sinogram = np.zeros(sino_shape)
    
    counts_minlength=np.max(im) + 1
    
    for j, angle in enumerate(angles):
        
        r_im = rotate(im, angle, order=0, preserve_range=True)
        ray_sums = np.apply_along_axis(np.bincount, 1, r_im, minlength=counts_minlength)[:, 1:].T * voxel_size
    
        att_material_map = np.zeros((attenuations.shape[1], ray_sums.shape[1]))
        for idx, _ in enumerate(ray_sums):
            att_material_map += np.outer(attenuations[idx], ray_sums[idx])
            
        trans_sums = np.exp(-att_material_map).T
        
        passed_specs = trans_sums * spectrum
        passed_intensity = np.sum(passed_specs, axis=1)
        sinogram[j] = -np.log(passed_intensity)
        
    recon = gaussian(iradon(sinogram.T, theta=angles).T)
    recon[im == 0] = 0
    recon[recon < 0] = 0
    recon /= voxel_size # convert to 1/mm values

    return recon


# %%
recon_1 = recon_multiple_matters(im2, attenuations, spek_intensities_eff, voxel_size)
recon_2 = recon_multiple_matters(im1, np.array([att_SiC, att_SiC]), spek_intensities_eff_Al_filter, voxel_size)
recon_3 = recon_multiple_matters(im2, attenuations, spek_intensities_eff_Al_filter, voxel_size)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im_0 = ax[0].imshow(recon_1)
plt.colorbar(im_0, ax=ax[0])
im_1 = ax[1].imshow(recon_2)
plt.colorbar(im_1, ax=ax[1])
im_2 = ax[2].imshow(recon_3)
plt.colorbar(im_2, ax=ax[2])
plt.show()

row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
plt.plot(recon_0[row])
plt.plot(recon_1[row])
plt.plot(recon_2[row])
plt.plot(recon_3[row])
plt.ylabel('Коэффициент ослабления, 1/мм')
plt.grid(color='gray')
plt.show()


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im_0 = ax[0].imshow(recon_0)
plt.colorbar(im_0, ax=ax[0])
ax[0].axhline(recon_0.shape[0]//2, linewidth=4, c='white', alpha=0.5)
im_1 = ax[1].imshow(recon_1)
plt.colorbar(im_1, ax=ax[1])
ax[1].axhline(recon_0.shape[0]//2, linewidth=4, c='white', alpha=0.5)
plt.suptitle('а', fontsize=18)
# plt.savefig('Fig11a.eps', dpi=600)
plt.show()

# plt.imshow(recon_1)
# plt.colorbar()
# plt.show()

row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
plt.plot(recon_0[row], label='SiC')
plt.plot(recon_1[row], label='SiC + Fe')
plt.xlabel('воксели')
plt.ylabel('Коэффициент ослабления, 1/мм')
plt.grid(color='gray')
plt.legend(framealpha=1)
plt.suptitle('б', fontsize=18)
# plt.savefig('Fig11b.eps', dpi=600)
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im_0 = ax[0].imshow(recon_0, cmap='gray')
cbar = plt.colorbar(im_0, ax=ax[0])
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Коэффициент\nослабления, 1/мм', fontsize=28)

ax[0].axhline(recon_0.shape[0]//2, linewidth=4, c='white', alpha=0.5)
im_1 = ax[1].imshow(recon_1, cmap='gray')
cbar = plt.colorbar(im_1, ax=ax[1])
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Коэффициент\nослабления, 1/мм', fontsize=28)

ax[1].axhline(recon_0.shape[0]//2, linewidth=4, c='white', alpha=0.5)
plt.suptitle('(а)', fontsize=28, font='Times New Roman')

ax[0].tick_params(labelsize='14')
ax[1].tick_params(labelsize='14')

plt.tight_layout()
# plt.savefig('Fig11a.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

plt.show()

row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot(recon_0[row], label='SiC', c='black')
plt.text(126, 1.23, '1', fontsize=12, fontstyle='italic')

plt.plot(recon_1[row], label='SiC + Fe', c='black', ls='dotted')
plt.text(142, 1.32, '2', fontsize=12, fontstyle='italic')

plt.xlabel('Номер вокселя', fontsize=28)
plt.ylabel('Коэффициент\nослабления, 1/мм', fontsize=28)

plt.suptitle('(б)', fontsize=28, font='Times New Roman')

plt.tick_params(direction='in', labelsize='14')

plt.tight_layout()
# plt.savefig('Fig11b.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

plt.show()

# %%
im2[im2==1] = 0
plt.imshow(im2)

# %%
# attenuations = np.array([att_SiC, att_Air])
attenuations = np.array([np.zeros(att_SiC.shape), att_SiC])
recon_4 = recon_multiple_matters(im2, attenuations, spek_intensities_eff, voxel_size)

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
im_0 = ax[0].imshow(recon_1)
plt.colorbar(im_0, ax=ax[0])
im_1 = ax[1].imshow(recon_4)
plt.colorbar(im_1, ax=ax[1])
plt.show()

plt.imshow(recon_4)
plt.colorbar()
plt.show()

plt.imshow(recon_4)
plt.colorbar()
plt.axhline(recon_4.shape[0]//2, linewidth=4, c='white', alpha=0.5)
plt.axvline(recon_4.shape[0]//2, linewidth=4, c='white', alpha=0.5)
plt.show()


row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
plt.plot(recon_0[row], label='SiC')
# plt.plot(recon_1[row])
plt.plot(recon_4[row], label='SiC + void')
plt.xlabel('Горизонтальное сечение')
plt.ylabel('Коэффициент ослабления, 1/мм')
plt.grid(color='gray')
plt.legend()
plt.show()

# %%
row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
plt.plot(recon_0[:, row], label='SiC')
plt.plot(recon_4[:, row], label='SiC + void')
plt.xlabel('Вертикальное сечение')
plt.ylabel('Коэффициент ослабления, 1/мм')
plt.grid(color='gray')
plt.legend()
plt.show()

# %%
im2[im2==1] = 0
plt.imshow(im2)

# %%
recon_5 = recon_multiple_matters(im2, attenuations, spek_intensities_eff, voxel_size)
recon_6 = recon_multiple_matters(im2, np.array([att_Air, att_Air]), spek_intensities_eff, voxel_size)

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
im_0 = ax[0].imshow(recon_5)
plt.colorbar(im_0, ax=ax[0])
im_1 = ax[1].imshow(recon_6)
plt.colorbar(im_1, ax=ax[1])
plt.show()

row = im1.shape[0] // 2
plt.figure(figsize=(10, 5))
# plt.plot(recon_5[row])
plt.plot(recon_6[row])
plt.ylabel('Коэффициент ослабления, 1/мм')
plt.grid(color='gray')
plt.show()

# %%
plt.imshow(recon_1)

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
ax.plot_surface(np.arange(recon_1.shape[0]), np.arange(recon_1.shape[1]), recon_1, cmap=cm.Blues)

# ax.set(xticklabels=[],
#        yticklabels=[],
#        zticklabels=[])

plt.show()

# %%

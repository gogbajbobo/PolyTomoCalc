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
import spekpy
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from skimage.draw import disk

# %%
s = spekpy.Spek(kvp=50, dk=0.5, targ='Mo')
s.filter('Air', 1440)
spek_energies, spek_intensities = s.get_spectrum()
spek_intensities /= spek_intensities.sum()

plt.plot(spek_energies, spek_intensities)
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

plt.plot(spek_energies, spek_intensities)
plt.plot(spek_energies, spek_intensities_eff)
plt.yscale('log')
plt.ylim(1e-4, 1e0)
plt.grid()
plt.show()

# %%
att_SiC = xraydb.material_mu('SiC', spek_energies * 1000) / 10
att_Fe = xraydb.material_mu('Fe', spek_energies * 1000) / 10

plt.plot(spek_energies, att_SiC)
plt.plot(spek_energies, att_Fe)
plt.yscale('log')
plt.grid()
plt.show()


# %%
def calc_object_mus_from_spectrum(im, spectrum, mat_att, voxel_size):

    angles = np.arange(0, 180, 1)
    sino_shape = (angles.size, im.shape[0])
    sinogram_wo_GOS = np.zeros(sino_shape)
    
    for j, angle in enumerate(angles):
        
        r_im = rotate(im, angle, order=1, preserve_range=True)
        ray_sums = np.sum(r_im, axis=1) * voxel_size
        trans_sums = np.exp(np.outer(-mat_att, ray_sums)).T
        
        passed_specs_wo_GOS = trans_sums * spectrum
        passed_intensity_wo_GOS = np.sum(passed_specs_wo_GOS, axis=1)
        sinogram_wo_GOS[j] = -np.log(passed_intensity_wo_GOS)
        
    recon_wo_GOS = gaussian(iradon(sinogram_wo_GOS.T, theta=angles).T)
    recon_wo_GOS[im == 0] = 0
    recon_wo_GOS[recon_wo_GOS < 0] = 0
    recon_wo_GOS /= voxel_size # convert to 1/mm values
    
    plt.imshow(recon_wo_GOS)
    plt.colorbar()
    plt.show()
    
    row = im.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(recon_wo_GOS[row])
    plt.ylabel('Коэффициент ослабления, 1/мм')
    plt.grid(color='gray')
    plt.show()
    
    return recon_wo_GOS


# %%
size = 256
shape = (size, size)
im1 = np.zeros(shape).astype(int)
rr, cc = disk((size//2, size//2), size//3, shape=shape)
im1[rr, cc] = 1

plt.imshow(im1)

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm

_ = calc_object_mus_from_spectrum(im1, spek_intensities_eff, att_SiC, voxel_size)

# %%
size = 256
shape = (size, size)
im2 = np.zeros(shape).astype(int)
rr, cc = disk((size//2, size//2), size//3, shape=shape)
im2[rr, cc] = 1

rr, cc = disk((size//3, size//2), size//12, shape=shape)
im2[rr, cc] = 2

rr, cc = disk((2 * size//3, size//2), size//12, shape=shape)
im2[rr, cc] = 2

plt.imshow(im2)

# %%
np.unique(im1, return_counts=True)

# %%
np.unique(im2, return_counts=True)

# %%
rim = rotate(im2, 10, order=0, preserve_range=True)
np.unique(rim, return_counts=True)

# %%
counts = np.apply_along_axis(np.bincount, 1, im2, minlength=np.max(im2)+1)
counts.shape

# %%
plt.imshow(counts * voxel_size)

# %%
attenuations = np.array([att_SiC, att_Fe])
attenuations.shape

# %%
counts[:, 1:][106]

# %%
np.outer(-attenuations, counts[:, 1:]).shape

# %%

# %%

# %%
im = np.copy(im2).astype(int)

angles = np.arange(0, 180, 1)
sino_shape = (angles.size, im.shape[0])
sinogram = np.zeros(sino_shape)

counts_minlength=np.max(im) + 1

for j, angle in enumerate(angles):
    
    r_im = rotate(im, angle, order=0, preserve_range=True)
    ray_sums = np.apply_along_axis(np.bincount, 1, r_im, minlength=counts_minlength) * voxel_size
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


# %%

# %%

# %%

# %%

# %%

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
import xraydb
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian, median

# %%
input_path = 'Mo_spec_poly_45.npy'
with open(input_path, 'rb') as f:
    spectrum = np.load(f).astype(float)

input_path = 'Mo_spec_poly_45_energies.npy'
with open(input_path, 'rb') as f:
    en_keV = np.load(f).astype(float)

plt.plot(en_keV, spectrum)
plt.yscale('log')

# %%
att_air = np.exp(-xraydb.material_mu('air', en_keV*1000) * 144)
plt.plot(en_keV, att_air)
plt.show()

spectrum_filtered = sp.ndimage.gaussian_filter(spectrum, sigma=1) * att_air
spectrum_filtered[:140] = 0
spectrum_filtered[980:] = 0
spectrum_filtered /= spectrum_filtered.sum()

plt.plot(en_keV, spectrum_filtered)
plt.yscale('log')

# %%
ihx_d = 2.2 #g/cm3
wat_d = 0.997
ihx_w = 0.647 #g
ihx_d_s1 = ihx_w + wat_d * (1 - ihx_w/ihx_d)
print(ihx_d_s1)

ihx_p = 5 #4.17
wat_p = 60
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
xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_i} O{o_i}', ihx_d_s2)
iohexol_mu = xraydb.material_mu('iohexol', en_keV*1000) / 10
plt.plot(en_keV, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/dbace4ca-3ba6-4a8a-b191-d52fe70c8a4f.npy'
with open(input_path, 'rb') as f:
  im = np.load(f)

print('im', im.shape)

im = im[40:1260, 40:1260]

print('im', im.shape)

plt.imshow(im)
plt.colorbar()
plt.show()

line_y = im.shape[0]//2

plt.plot(im[line_y])
plt.plot(sp.ndimage.gaussian_filter(im[line_y], sigma=2))
plt.show()

print(np.mean(im[line_y, 200:1000]))

# %%
bim = gaussian(im, sigma=3) > 0.1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im))
ax[1].imshow(bim)

# %%
plt.plot(bim[line_y])
plt.plot(im[line_y])

recon_point_sum = im[line_y][bim[line_y] > 0].sum()

print(bim[line_y].sum())
print(recon_point_sum)

# %%
voxel_size = 0.009 # in mm — 0.01 = 10µm

ray_sum = bim[line_y].sum() * voxel_size
trans_sum = np.exp(-iohexol_mu * ray_sum)

print(ray_sum)

plt.plot(en_keV, trans_sum)
plt.grid()

# %%
spectrum_passed = spectrum_filtered * trans_sum

plt.plot(en_keV, spectrum_filtered)
plt.plot(en_keV, spectrum_passed)
plt.yscale('log')
plt.grid()
plt.show()

xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus = xraydb.material_mu('GOS', en_keV*1000) / 10
GOS_t = np.exp(-GOS_mus * 22 * 0.001) # (22 * 0.001)mm == 22µm
# GOS_t = np.exp(-GOS_mus * 11 * 0.001)
beta = 3
en_gap = 4.6 # eV

GOS_n_p = en_keV*1000 / (beta * en_gap)
GOS_eff = GOS_n_p * (1 - GOS_t)

plt.plot(en_keV, GOS_t)
plt.grid()
plt.show()
plt.plot(en_keV, GOS_n_p)
plt.grid()
plt.show()
plt.plot(en_keV, GOS_eff)
plt.grid()
plt.show()

sf_GOS = spectrum_filtered * GOS_eff
sp_GOS = spectrum_passed * GOS_eff

intensity = spectrum_passed.sum() / spectrum_filtered.sum()
intensity_GOS = sp_GOS.sum() / sf_GOS.sum()

print(f'intensity flat: {spectrum_filtered.sum()}')
print(f'intensity passed: {spectrum_passed.sum()}')
print(f'intensity: {intensity}')
print(f'intensity_GOS: {intensity_GOS}')
print(f'model point sum: {-np.log(intensity) / voxel_size}')
print(f'GOS model point sum: {-np.log(intensity_GOS) / voxel_size}')
print(f'recon point sum: {recon_point_sum}')


# %%
def point_sum_calc(spectrum_filtered, ihx_mu):

    voxel_size = 0.009 # in mm — 0.01 = 10µm

    ray_sum = bim[line_y].sum() * voxel_size
    trans_sum = np.exp(-ihx_mu * ray_sum)
    spectrum_passed = spectrum_filtered * trans_sum

    sf_GOS = spectrum_filtered * GOS_eff
    sp_GOS = spectrum_passed * GOS_eff

    intensity = spectrum_passed.sum() / spectrum_filtered.sum()
    intensity_GOS = sp_GOS.sum() / sf_GOS.sum()

    print(f'intensity: {intensity}')
    print(f'intensity_GOS: {intensity_GOS}')
    print(f'model point sum: {-np.log(intensity) / voxel_size}')
    print(f'GOS model point sum: {-np.log(intensity_GOS) / voxel_size}')
    print(f'recon point sum: {recon_point_sum}')



# %%
point_sum_calc(spectrum_filtered, iohexol_mu)

# %%
from _handler_funcs import generate_spectrum

# %%
_, s = generate_spectrum(40, 45, 'Mo', energies=en_keV)
s /= s.sum()
sf = s * att_air
sf /= sf.sum()

# %%
plt.plot(en_keV, s)
plt.plot(en_keV, sf)
plt.plot(en_keV, spectrum_filtered)
plt.grid()
plt.ylim([1e-6, 5e-1])
plt.yscale('log')

# %%
point_sum_calc(spectrum_filtered, iohexol_mu)

# %%
point_sum_calc(sf, iohexol_mu)

# %%
en_step = np.mean(en_keV[1:] - en_keV[:-1])

s = spekpy.Spek(kvp=45, dk=en_step, targ='Mo')
s.filter('Air', 1440)
energies, intensities = s.get_spectrum()
intensities /= intensities.sum()

plt.plot(energies, intensities)

plt.ylim([2e-5, 4e-1])
plt.yscale('log')
plt.grid()

# %%
GOS_mus = xraydb.material_mu('GOS', energies * 1000) / 10
GOS_t = np.exp(-GOS_mus * 22 * 0.001) # (22 * 0.001)mm == 22µm
# GOS_t = np.exp(-GOS_mus * 11 * 0.001)
beta = 3
en_gap = 4.6 # eV

GOS_n_p = energies * 1000 / (beta * en_gap)
GOS_eff = GOS_n_p * (1 - GOS_t)

plt.plot(energies, GOS_eff)
plt.grid()
plt.show()


# %%
iohexol_mu = xraydb.material_mu('iohexol', energies * 1000) / 10
plt.plot(energies, iohexol_mu)
plt.yscale('log')
plt.show()


# %%
point_sum_calc(intensities, iohexol_mu)

# %%
a = np.array([[2, 2, 1]])
b = np.array([[45, 4, 1]])
t = np.array([[1, 2, 3], [3, 4, 5], [6, 3, 1]])
np.matmul(a.T, b)

# %%
np.linalg.inv(b.T)

# %%

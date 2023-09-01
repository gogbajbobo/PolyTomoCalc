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


# %%
def extract_data(file_name):
    res = []
    with open(file_name,'rb') as f:
        find_data = False
        for l in f.readlines():
            s = l.decode('ascii').replace('\r\n','')
            if s == '<<END>>'  and find_data:
                break
            if find_data:
                res.append(s)
            if s == '<<DATA>>':
                find_data =True

        return np.array(res).astype(int)


# %%
spectrum = extract_data('/Users/grimax/Desktop/tmp/Iohexol_samples/20230829 -- Mo-tube-poly-45kV .mca')
plt.plot(spectrum)
plt.yscale('log')

# %%
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(spectrum.shape[0])])

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
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/dbace4ca-3ba6-4a8a-b191-d52fe70c8a4f.npy'
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
voxel_size = 0.0009 # in cm — 0.001 = 10µm

ray_sums = bim[line_y].sum() * voxel_size
trans_sums = np.exp(-iohexol_mu * ray_sums)

plt.plot(en_keV, trans_sums)
plt.grid()

# %%
spectrum_passed = spectrum_filtered * trans_sums

plt.plot(en_keV, spectrum_filtered)
plt.plot(en_keV, spectrum_passed)
plt.yscale('log')

print(f'intensity flat: {spectrum_filtered.sum()}')
print(f'intensity passed: {spectrum_passed.sum()}')

intensity = spectrum_passed.sum() / spectrum_filtered.sum()
print(f'intensity: {intensity}')
print(f'model point sum: {-np.log(intensity) / voxel_size}')
print(f'recon point sum: {recon_point_sum}')


# %%
def point_sum_calc(spectrum_filtered):

    voxel_size = 0.0009 # in cm — 0.001 = 10µm

    ray_sums = bim[line_y].sum() * voxel_size
    trans_sums = np.exp(-iohexol_mu * ray_sums)
    spectrum_passed = spectrum_filtered * trans_sums
    
    intensity = spectrum_passed.sum() / spectrum_filtered.sum()
    print(f'intensity: {intensity}')
    print(f'model point sum: {-np.log(intensity) / voxel_size}')
    print(f'recon point sum: {recon_point_sum}')



# %%
point_sum_calc(spectrum_filtered)

# %%
test = np.ones(spectrum_filtered.shape)
test /= test.sum()

point_sum_calc(test)

# %%

# %%
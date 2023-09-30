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

# %% id="rP2oSvmykXIF"
import xraydb
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, gaussian, median
from skimage.transform import rotate, iradon, iradon_sart
from _handler_funcs import generate_spectrum


# %% id="LhzCFhee27MO"
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


# %% colab={"base_uri": "https://localhost:8080/"} id="_A8FQwxNkhg8" outputId="4188dbc1-2d82-488e-ede8-32ae8348348c"
spectrum = extract_data('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/20230829 -- Mo-tube-poly-45kV .mca')
spectrum

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="NjXlyrXB3MPF" outputId="6f8409ee-9693-409a-9532-4ad2d7d7db43"
plt.plot(spectrum)
plt.yscale('log')

# %% colab={"base_uri": "https://localhost:8080/"} id="BGh_MV6g3qMm" outputId="57b8502d-bd59-48f5-fe26-d8335bedf92e"
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(spectrum.shape[0])])

en_keV

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="1po8xLrn5dv1" outputId="d657237b-e03b-490a-f5af-5c9239751a62"
plt.plot(en_keV, spectrum)
plt.yscale('log')

# %% colab={"base_uri": "https://localhost:8080/"} id="NbWcPZiKaz7R" outputId="aa550f52-fb59-4275-c206-9a06dd460f2f"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 447} id="xQMT_IXB_gAa" outputId="d28e970d-6268-4032-b182-1c2ab548e4ff"
xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_i} O{o_i}', ihx_d_s2)
iohexol_mu = xraydb.material_mu('iohexol', en_keV*1000) / 10
plt.plot(en_keV, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])


# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="fJPXFdu3lOHN" outputId="c54a9ca9-da5b-4929-a4ee-f502a179e714"
# input_path = '/content/drive/MyDrive/Colab Data/Iohexol_samples/e82c1068-5c0f-40c3-9dba-4e811b566344.npy'
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/Iohexol_samples/dbace4ca-3ba6-4a8a-b191-d52fe70c8a4f.npy'
with open(input_path, 'rb') as f:
  im = np.load(f)

print('im', im.shape)

im = im[40:1260, 40:1260]

print('im', im.shape)

plt.imshow(im)
plt.colorbar()
plt.show()

plt.plot(im[650])
plt.plot(sp.ndimage.gaussian_filter(im[650], sigma=2))
plt.show()

print(np.mean(im[650, 200:1000]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 448} id="ekjVVd3cUVzK" outputId="da69431d-076a-4a56-9a81-4ee005f5a187"
att_air = np.exp(-xraydb.material_mu('air', en_keV*1000) * 144)
plt.plot(en_keV, att_air)
# plt.yscale('log')

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="4KCZR_iOU8KB" outputId="d8253c1e-19c6-44a5-fc70-3862db0cdd51"
spectrum_filtered = sp.ndimage.gaussian_filter(spectrum, sigma=1) * att_air
spectrum_filtered[:140] = 0
spectrum_filtered[980:] = 0
spectrum_filtered /= spectrum_filtered.sum()

plt.plot(en_keV, spectrum_filtered)
plt.yscale('log')

# %% colab={"base_uri": "https://localhost:8080/"} id="o3_S-vFnWfNW" outputId="fd6e247f-df72-492e-8a9c-059ee0a94d93"
np.sum(iohexol_mu * spectrum_filtered)

# spec = np.copy(spectrum).astype('float')
# spec[:140] = 0
# spec[980:] = 0
# spec /= spec.sum()
# np.sum(iohexol_mu * spec)

# %% colab={"base_uri": "https://localhost:8080/", "height": 435} id="E5zXmvOupDd-" outputId="b0f8d8cc-086c-40ed-df35-35161a45f031"
bim = gaussian(im, sigma=3) > 0.1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im))
ax[1].imshow(bim)


# %% id="Cpwn4UuRktyM"
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

  # return recon_wo_GOS, recon_GOS, recon_GOS_diff



# %% colab={"base_uri": "https://localhost:8080/", "height": 840} id="E1n8EgDbrGeb" outputId="30a3e4f9-7d14-458f-f703-8653490c60a9"
voxel_size = 0.0009 # in cm — 0.001 = 10µm

# wo_GOS, w_GOS, diff_GOS = calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)
calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="QKSMj2DJrkEb" outputId="ff1568af-620c-4c60-e443-e34a2b999300"
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/e82c1068-5c0f-40c3-9dba-4e811b566344.npy'
with open(input_path, 'rb') as f:
  im_mono = np.load(f)

print('im_mono', im_mono.shape)

im_mono = im_mono[40:1260, 40:1260]

print('im_mono', im_mono.shape)

plt.imshow(im_mono)
plt.colorbar()
plt.show()

plt.plot(im_mono[650])
plt.plot(sp.ndimage.gaussian_filter(im_mono[650], sigma=2))
plt.show()

print(np.mean(im_mono[650, 200:1000]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 435} id="5X2EHcjGsRFl" outputId="2be66bfe-5bbf-4b7d-b790-df7a2f332da5"
bim_mono = gaussian(im_mono, sigma=3) > 0.1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gaussian(im_mono))
ax[1].imshow(bim_mono)


# %% colab={"base_uri": "https://localhost:8080/", "height": 840} id="9zwKLjeR5_jk" outputId="674f3dc9-7a8d-4989-b194-9084c60eac3f"
voxel_size = 0.0009 # in cm — 0.001 = 10µm

# wo_GOS, w_GOS, diff_GOS = calc_object_mus_from_spectrum(bim, gaussian(im), np.array([1]), np.array([17480]), 0.1956*10, voxel_size)
calc_object_mus_from_spectrum(bim_mono, gaussian(im_mono), np.array([1]), np.array([17480]), 0.1956*10, voxel_size)

# %% colab={"base_uri": "https://localhost:8080/", "height": 843} id="-_jMRWZf6cPl" outputId="2c134e9d-2116-4f55-c3ac-1ff956a5e1b1"
Be_mus = xraydb.material_mu('beryllium', en_keV*1000)
Xe_mus = xraydb.material_mu('xenon', en_keV*1000)
GOS_mus = xraydb.material_mu('GOS', en_keV*1000)

GOS_t = np.exp(-GOS_mus * 22 * 0.0001) # (22 * 0.0001)cm == 22µm
Be_t = np.exp(-Be_mus * 0.05)
Xe_t = np.exp(-Xe_mus * 0.2)

plt.plot(en_keV, Be_mus)
plt.plot(en_keV, Xe_mus)
plt.plot(en_keV, GOS_mus)
plt.yscale('log')
plt.show()

plt.plot(en_keV, Be_t)
plt.plot(en_keV, Xe_t)
plt.plot(en_keV, GOS_t)
plt.show()

# %% id="0n41L95qz6Ce"
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

# %% id="TISaAE07AbbB"
voxel_size = 0.0009 # in cm — 0.001 = 10µm

# wo_GOS, w_GOS, diff_GOS = calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)
calc_object_mus_from_spectrum(bim, gaussian(im), sf, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
_, s20 = generate_spectrum(20, 45, 'Mo', energies=en_keV)
s20 /= s20.sum()
s20 = s20 * att_air
s20 /= s20.sum()

# %%
plt.plot(en_keV, sf)
plt.plot(en_keV, s20)
plt.grid()
plt.ylim([1e-6, 5e-1])
plt.yscale('log')

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
xraydb.add_material('iohexol', f'C19H26I3N3O9 H{h_i} O{o_i}', ihx_d_s2)
iohexol_mu = xraydb.material_mu('iohexol', en_keV*1000) / 10
plt.plot(en_keV, iohexol_mu)
plt.yscale('log')
plt.show()

print(iohexol_mu[371])

# %%
with open('Mo_spec_mono_45.npy', 'rb') as f:
    spec_mono = np.load(f)

plt.plot(en_keV, spec_mono)

# %%
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/82fc7477-dafb-4950-aea5-6e522910181d.npy'
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
bim = gaussian(im, sigma=3) > 0.1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im)
ax[1].imshow(bim)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), np.array([1]), np.array([17480]), 0.6560*10, voxel_size)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spec_mono, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
with open('Mo_spec_mono_45_0.npy', 'rb') as f:
    spec_mono_0 = np.load(f)

plt.plot(en_keV, spec_mono_0)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spec_mono_0, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
plt.plot(en_keV, sf)
plt.plot(en_keV, spectrum_filtered)
plt.grid()
plt.ylim([0, 0.1])
plt.xlim([17, 20])
# plt.yscale('log')

# %%
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/cd130f11-38b7-4de8-ad96-85f30f8a6105.npy'
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
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

# wo_GOS, w_GOS, diff_GOS = calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)
calc_object_mus_from_spectrum(bim, gaussian(im), sf, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
ihx_d = 2.2 #g/cm3
wat_d = 0.997
ihx_w = 0.647 #g
ihx_d_s1 = ihx_w + wat_d * (1 - ihx_w/ihx_d)
print(ihx_d_s1)

ihx_p = 1
wat_p = 3
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
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/f244cee7-79da-4190-8442-560e3f3a4622.npy'
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
bim = gaussian(im, sigma=3) > 0.2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im)
ax[1].imshow(bim)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spectrum_filtered, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), sf, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
input_path = '/Users/grimax/Desktop/tmp/Iohexol_samples/246198bb-212e-43a1-92b2-30bc2c46a351.npy'
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
bim = gaussian(im, sigma=3) > 0.2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im)
ax[1].imshow(bim)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), np.array([1]), np.array([17480]), 0.38438*10, voxel_size)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spec_mono, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spec_mono_0, en_keV*1000, iohexol_mu*10, voxel_size)

# %%
with open('Mo_spec_mono_45_2.npy', 'rb') as f:
    spec_mono_2 = np.load(f)

plt.plot(en_keV, spec_mono_2)

# %%
voxel_size = 0.0009 # in cm — 0.001 = 10µm

calc_object_mus_from_spectrum(bim, gaussian(im), spec_mono_2, en_keV*1000, iohexol_mu*10, voxel_size)

# %%

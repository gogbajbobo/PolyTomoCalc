# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from skimage.transform import rotate, iradon, iradon_sart
from skimage.filters import threshold_otsu, gaussian, median


# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_sino.npy'
with open(input_path, 'rb') as f:
  sino = np.load(f)

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_darks.npy'
with open(input_path, 'rb') as f:
  darks = np.load(f)

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_empties.npy'
with open(input_path, 'rb') as f:
  empties = np.load(f)

# %%
x_min = 500
x_max = 2500

# %%
sino = sino[:, x_min:x_max]
darks = darks[:, x_min:x_max]
empties = empties[:, x_min:x_max]

# %%
plt.imshow(sino, aspect='auto')
plt.colorbar()
plt.show()

# %%
plt.imshow(darks, aspect='auto')
plt.colorbar()
plt.show()

# %%
plt.imshow(empties, aspect='auto')
plt.colorbar()
plt.show()

# %%
print(f'darks.shape {darks.shape}')
dark = np.sum(darks, axis=0) / darks.shape[0]

# %%
plt.plot(darks[0])
plt.plot(dark)
plt.show()

# %%
print(f'empties.shape {empties.shape}')
empty = np.sum(empties, axis=0) / empties.shape[0]

# %%
plt.plot(empties[0])
plt.plot(empty)
plt.show()

# %%
sino -= dark
empty -= dark

# %%
plt.imshow(sino, aspect='auto')
plt.colorbar()
plt.show()

# %%
plt.plot(empty)
plt.show()

# %%
att_sino = - np.log(sino / empty)

# %%
plt.imshow(att_sino)
plt.colorbar()
plt.show()

# %%
att_sino = att_sino[:, 250:1750]

# %%
plt.imshow(att_sino)
plt.colorbar()
plt.show()

# %%
step = .2
voxel_size = 0.009 # in mm — 0.01 = 10µm

# %%
angles = np.arange(0, step * att_sino.shape[0], step)
recon = iradon(att_sino.T, theta=angles)
recon /= voxel_size # convert to 1/mm values

# %%
plt.imshow(recon)
plt.colorbar()
plt.show()

# %%
recon[recon < 0] = 0
plt.imshow(recon[250:1250, 400:1400])
plt.colorbar()
plt.show()

# %%
plt.plot(recon[800])
plt.show()

# %%
plt.plot((sino / empty)[0])
plt.show()

# %%
plt.plot(att_sino[0])
plt.plot(np.sum(recon, axis=0) * voxel_size)
plt.show()


# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_bh_corr_1.0.npy'
with open(input_path, 'rb') as f:
  im_SiC_0 = np.load(f)


# %%
plt.imshow(im_SiC_0)
plt.colorbar()

im_SiC_0.shape

# %%
plt.plot(att_sino[0, 300:1380])
plt.plot(np.sum(np.fliplr(im_SiC_0), axis=0) * voxel_size)
plt.show()

# %%
input_path = '/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/uvarov_samples/a99114d8-4d13-4350-9886-a437aa7bf22e_bh_corr_2.57_optimal.npy'
with open(input_path, 'rb') as f:
  bim_SiC = np.load(f)

bim_SiC = gaussian(median(bim_SiC))
thresh = threshold_otsu(bim_SiC)
print(f'thresh: {thresh}')
bim_SiC = (bim_SiC > thresh).astype(int)

plt.imshow(bim_SiC)
plt.show()

bim_SiC.shape

# %%
SiC_lengths = np.sum(np.fliplr(bim_SiC), axis=0) * voxel_size
SiC_att_0 = np.sum(np.fliplr(im_SiC_0), axis=0) * voxel_size
SiC_att_1 = att_sino[0, 300:1380]

x_min = 80
x_max = 1044

SiC_lengths = SiC_lengths[x_min:x_max]
SiC_att_0 = SiC_att_0[x_min:x_max]
SiC_att_1 = SiC_att_1[x_min:x_max]

plt.plot(SiC_att_0)
# plt.plot(SiC_att_1)
plt.plot(SiC_lengths)
plt.yscale('log')
plt.show()

# %%
np.random.seed(1)

rnd_indx = np.random.choice(range(SiC_lengths.size), size=SiC_lengths.size//20, replace=False)

plt.scatter(SiC_lengths, SiC_att_0, marker='.', c='lightgray')
plt.scatter(SiC_lengths[rnd_indx], SiC_att_0[rnd_indx], marker='.')
# plt.scatter(SiC_lengths, SiC_att_1, marker='.')
plt.show()

# %%
bin_edges = np.histogram_bin_edges(SiC_lengths, bins=32)
bin_width = bin_edges[1] - bin_edges[0]
bin_centers = bin_edges[:-1] + bin_width/2

# print(bin_edges)
# print(bin_centers)

# %%
mean_att_SiC_values = np.zeros(bin_centers.shape)

for idx, _ in enumerate(bin_centers):
    indicies = np.where((bin_edges[idx] < SiC_lengths) & (SiC_lengths < bin_edges[idx+1]))
    mean_att_SiC_values[idx] = np.mean(SiC_att_0[indicies])

plt.scatter(bin_centers, mean_att_SiC_values, marker='.')
plt.grid()
plt.show()


# %%
def equations(p):
    a, b, alpha = p
    result = list()
    middle_point = bin_centers.size // 4
    indicies = [0, middle_point, -1]
    xx = bin_centers.take([indicies])[0]
    yy = mean_att_SiC_values.take([indicies])[0]
    for x, y in zip(xx, yy):
        result.append(a + b / x**alpha - y)
    return result


# %%
a, b, alpha =  fsolve(equations, (1, 1, 1))

print(a, b, alpha)

# %%
num_solution_values = [a + b / v**alpha for v in bin_centers]

plt.scatter(SiC_lengths, SiC_att_0, marker='.', c='lightgray')
# plt.scatter(bin_centers, mean_att_SiC_values, marker='.')
plt.plot(bin_centers, num_solution_values)
plt.grid()
plt.show()


# %%
def calc_slope(a, b, alpha, x):
    x_delta = 0.001
    y0 = a + b / x**alpha
    y1 = a + b / (x - x_delta / 2)**alpha
    y2 = a + b / (x + x_delta / 2)**alpha
    y_slope = (y2 - y1) / x_delta
    return y0, y_slope


plt.scatter(SiC_lengths, SiC_att_0, marker='.', c='lightgray')
plt.plot(bin_centers, num_solution_values, linewidth=2)
for x in [1, 2, 4]:
    y0, y_slope = calc_slope(a, b, alpha, x)
    plt.axline((x, y0), slope=y_slope, c='green', linewidth=1)
plt.grid()
plt.show()

# %%
xx = np.arange(0.5, 7.5, 0.5)
yy_slope = np.array([])

for x in xx:
    y0, y_slope = calc_slope(a, b, alpha, x)
    yy_slope = np.append(yy_slope, y_slope)

# plt.plot(xx, yy_slope, c='lightgray')
plt.scatter(xx, yy_slope, marker='.')
plt.grid()
plt.show()

# %%
np.save('SiC_lengths', SiC_lengths)
np.save('SiC_att_0', SiC_att_0)

# %%

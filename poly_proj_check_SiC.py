# -*- coding: utf-8 -*-
"""Poly_proj_check_SiC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yaEl2LZ5qT1XXQsz07V-qEbyL0iQH-eG
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, iradon, iradon_sart

input_path = '/content/drive/MyDrive/Colab Data/uvarov_samples/projections/a99114d8-4d13-4350-9886-a437aa7bf22e_sino.npy'
with open(input_path, 'rb') as f:
  sino = np.load(f)

input_path = '/content/drive/MyDrive/Colab Data/uvarov_samples/projections/a99114d8-4d13-4350-9886-a437aa7bf22e_darks.npy'
with open(input_path, 'rb') as f:
  darks = np.load(f)

input_path = '/content/drive/MyDrive/Colab Data/uvarov_samples/projections/a99114d8-4d13-4350-9886-a437aa7bf22e_empties.npy'
with open(input_path, 'rb') as f:
  empties = np.load(f)

x_min = 500
x_max = 2500

sino = sino[:, x_min:x_max]
darks = darks[:, x_min:x_max]
empties = empties[:, x_min:x_max]

plt.imshow(sino, aspect='auto')
plt.colorbar()
plt.show()

plt.imshow(darks, aspect='auto')
plt.colorbar()
plt.show()

plt.imshow(empties, aspect='auto')
plt.colorbar()
plt.show()

print(f'darks.shape {darks.shape}')
dark = np.sum(darks, axis=0) / darks.shape[0]

plt.plot(darks[0])
plt.plot(dark)
plt.show()

print(f'empties.shape {empties.shape}')
empty = np.sum(empties, axis=0) / empties.shape[0]

plt.plot(empties[0])
plt.plot(empty)
plt.show()

sino -= dark
empty -= dark

plt.imshow(sino, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(empty)
plt.show()

att_sino = - np.log(sino / empty)

plt.imshow(att_sino)
plt.colorbar()
plt.show()

att_sino = att_sino[:, 250:1750]

plt.imshow(att_sino)
plt.colorbar()
plt.show()

step = .2
voxel_size = 0.0009 # in cm — 0.001 = 10µm

angles = np.arange(0, step * att_sino.shape[0], step)
recon = iradon(att_sino.T, theta=angles)
recon /= voxel_size * 10 # convert to 1/mm values

plt.imshow(recon)
plt.colorbar()
plt.show()

recon[recon < 0] = 0
plt.imshow(recon[250:1250, 400:1400])
plt.colorbar()
plt.show()

plt.plot(recon[800])
plt.show()

plt.plot((sino / empty)[0])
plt.show()

plt.plot(att_sino[0])
plt.show()

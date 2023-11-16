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

# %%
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(1024)])
en_keV

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus = xraydb.material_mu('GOS', en_keV*1000) / 10
GOS_t = np.exp(-GOS_mus * 22 * 0.001) # (22 * 0.001)mm == 22µm
GOS_t2 = np.exp(-GOS_mus * 44 * 0.001)
GOS_t3 = np.exp(-GOS_mus * 88 * 0.001)
GOS_t4 = np.exp(-GOS_mus * 176 * 0.001)
GOS_t5 = np.exp(-GOS_mus * 11 * 0.001)
GOS_t6 = np.exp(-GOS_mus * 5.5 * 0.001)

beta = 3
en_gap = 4.6 # eV
GOS_n_p = en_keV*1000 / (beta * en_gap)

GOS_eff = GOS_n_p * (1 - GOS_t)
GOS_eff2 = GOS_n_p * (1 - GOS_t2)
GOS_eff3 = GOS_n_p * (1 - GOS_t3)
GOS_eff4 = GOS_n_p * (1 - GOS_t4)
GOS_eff5 = GOS_n_p * (1 - GOS_t5)
GOS_eff6 = GOS_n_p * (1 - GOS_t6)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].plot(en_keV, GOS_t6, label='5.5µm')
ax[0].plot(en_keV, GOS_t5, label='11µm')
ax[0].plot(en_keV, GOS_t, label='22µm')
ax[0].plot(en_keV, GOS_t2, label='44µm')
ax[0].plot(en_keV, GOS_t3, label='88µm')
ax[0].plot(en_keV, GOS_t4, label='176µm')
ax[0].grid()
ax[0].legend()

ax[1].plot(en_keV, GOS_eff6, label='5.5µm')
ax[1].plot(en_keV, GOS_eff5, label='11µm')
ax[1].plot(en_keV, GOS_eff, label='22µm')
ax[1].plot(en_keV, GOS_eff2, label='44µm')
ax[1].plot(en_keV, GOS_eff3, label='88µm')
ax[1].plot(en_keV, GOS_eff4, label='176µm')
ax[1].grid()
ax[1].legend()

plt.show()


# %%
xraydb.add_material('CsI', 'CsI', 4.51)
CsI_mus = xraydb.material_mu('CsI', en_keV*1000) / 10
CsI_t = np.exp(-CsI_mus * 18.75 * 0.001)
CsI_t2 = np.exp(-CsI_mus * 37.5 * 0.001)
CsI_t3 = np.exp(-CsI_mus * 75 * 0.001)
CsI_t4 = np.exp(-CsI_mus * 150 * 0.001)
CsI_t5 = np.exp(-CsI_mus * 300 * 0.001)
CsI_t6 = np.exp(-CsI_mus * 600 * 0.001)

beta = 3
en_gap = 6.135 # eV # doi: 10.3866/PKU.WHXB201707031
CsI_n_p = en_keV*1000 / (beta * en_gap)

CsI_eff = CsI_n_p * (1 - CsI_t)
CsI_eff2 = CsI_n_p * (1 - CsI_t2)
CsI_eff3 = CsI_n_p * (1 - CsI_t3)
CsI_eff4 = CsI_n_p * (1 - CsI_t4)
CsI_eff5 = CsI_n_p * (1 - CsI_t5)
CsI_eff6 = CsI_n_p * (1 - CsI_t6)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].plot(en_keV, CsI_t, label='18.75µm')
ax[0].plot(en_keV, CsI_t2, label='37.5µm')
ax[0].plot(en_keV, CsI_t3, label='75µm')
ax[0].plot(en_keV, CsI_t4, label='150µm')
ax[0].plot(en_keV, CsI_t5, label='300µm')
ax[0].plot(en_keV, CsI_t6, label='600µm')
ax[0].grid()
ax[0].legend()

ax[1].plot(en_keV, CsI_eff, label='18.75µm')
ax[1].plot(en_keV, CsI_eff2, label='37.5µm')
ax[1].plot(en_keV, CsI_eff3, label='75µm')
ax[1].plot(en_keV, CsI_eff4, label='150µm')
ax[1].plot(en_keV, CsI_eff5, label='300µm')
ax[1].plot(en_keV, CsI_eff6, label='600µm')
ax[1].grid()
ax[1].legend()

plt.show()


# %%
xraydb.add_material('LuAG', 'Lu3Al5O12', 6.71)
LuAG_mus = xraydb.material_mu('LuAG', en_keV*1000) / 10
LuAG_t = np.exp(-LuAG_mus * 18.75 * 0.001)
LuAG_t2 = np.exp(-LuAG_mus * 37.5 * 0.001)
LuAG_t3 = np.exp(-LuAG_mus * 75 * 0.001)
LuAG_t4 = np.exp(-LuAG_mus * 150 * 0.001)
LuAG_t5 = np.exp(-LuAG_mus * 300 * 0.001)
LuAG_t6 = np.exp(-LuAG_mus * 600 * 0.001)

beta = 3
en_gap = 7 # eV # https://pubs.acs.org/doi/10.1021/acs.jpcc.2c04523
LuAG_n_p = en_keV*1000 / (beta * en_gap)

LuAG_eff = LuAG_n_p * (1 - LuAG_t)
LuAG_eff2 = LuAG_n_p * (1 - LuAG_t2)
LuAG_eff3 = LuAG_n_p * (1 - LuAG_t3)
LuAG_eff4 = LuAG_n_p * (1 - LuAG_t4)
LuAG_eff5 = LuAG_n_p * (1 - LuAG_t5)
LuAG_eff6 = LuAG_n_p * (1 - LuAG_t6)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].plot(en_keV, LuAG_t, label='18.75µm')
ax[0].plot(en_keV, LuAG_t2, label='37.5µm')
ax[0].plot(en_keV, LuAG_t3, label='75µm')
ax[0].plot(en_keV, LuAG_t4, label='150µm')
ax[0].plot(en_keV, LuAG_t5, label='300µm')
ax[0].plot(en_keV, LuAG_t6, label='600µm')
ax[0].grid()
ax[0].legend()

ax[1].plot(en_keV, LuAG_eff, label='18.75µm')
ax[1].plot(en_keV, LuAG_eff2, label='37.5µm')
ax[1].plot(en_keV, LuAG_eff3, label='75µm')
ax[1].plot(en_keV, LuAG_eff4, label='150µm')
ax[1].plot(en_keV, LuAG_eff5, label='300µm')
ax[1].plot(en_keV, LuAG_eff6, label='600µm')
ax[1].grid()
ax[1].legend()

plt.show()


# %%
en_keV = np.arange(140) + 1
en_keV

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus = xraydb.material_mu('GOS', en_keV*1000) / 10
GOS_t_140 = np.exp(-GOS_mus * 140 * 0.001)
GOS_t_210 = np.exp(-GOS_mus * 210 * 0.001)

qe = 60 # photon/keV
GOS_n_p = en_keV * qe

GOS_eff_140 = GOS_n_p * (1 - GOS_t_140)
GOS_eff_210 = GOS_n_p * (1 - GOS_t_210)

plt.plot(en_keV, GOS_eff_140, label='140µm')
plt.plot(en_keV, GOS_eff_210, label='210µm')
plt.grid()
plt.legend()

plt.show()

# %%

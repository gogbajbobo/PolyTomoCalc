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
plt.grid()

# %%
en_step = (19.608 - 17.479) / (416 - 371)
en_keV = np.array([17.479 + (i - 371) * en_step for i in np.arange(spectrum.shape[0])])

plt.plot(en_keV, spectrum)
plt.yscale('log')
plt.grid()

# %%
att_air = np.exp(-xraydb.material_mu('air', en_keV*1000) * 144)
# plt.plot(en_keV, att_air)
# plt.yscale('log')

spectrum_filtered = sp.ndimage.gaussian_filter(spectrum, sigma=1) * att_air
spectrum_filtered[:140] = 0
spectrum_filtered[980:] = 0
spectrum_filtered /= spectrum_filtered.sum()

plt.plot(en_keV, spectrum_filtered)
plt.yscale('log')


# %%
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


# %%
sf_edit = np.copy(spectrum_filtered)

en_range = (10, 11)
idx_min = np.where(en_keV > en_range[0])[0][0]
idx_max = np.where(en_keV < en_range[1])[0][-1]
idx_range = slice(idx_min, idx_max)
print(idx_range)

x = en_keV[idx_range]
y = sf_edit[idx_range]

k = (y[-1] - y[0]) / (len(y) - 1)
y = [v - (y[0] + k * idx) for idx, v in enumerate(y)]

mean = np.average(x, weights=y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

print(mean, sigma)

popt, pcov = sp.optimize.curve_fit(gauss, x, y, p0=[1, mean, sigma])

print(popt)

gauss_fit = gauss(x, *popt)

sf_edit[idx_range] = sf_edit[idx_range] - gauss_fit

plt.plot(en_keV[idx_range], spectrum_filtered[idx_range])
plt.plot(en_keV[idx_range], sf_edit[idx_range])
plt.plot(x, gauss_fit)
plt.grid()
plt.show()

plt.plot(en_keV, spectrum_filtered)
plt.plot(en_keV, sf_edit)
plt.grid()
plt.yscale('log')

# %%
en_range = (12, 13.2)
idx_min = np.where(en_keV > en_range[0])[0][0]
idx_max = np.where(en_keV < en_range[1])[0][-1]
idx_range = slice(idx_min, idx_max)
print(idx_range)

x = en_keV[idx_range]
y = sf_edit[idx_range]

k = (y[-1] - y[0]) / (len(y) - 1)
y = [v - (y[0] + k * idx) for idx, v in enumerate(y)]

mean = np.average(x, weights=y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

print(mean, sigma)

popt, pcov = sp.optimize.curve_fit(gauss, x, y, p0=[1, mean, sigma])

print(popt)

gauss_fit = gauss(x, *popt)

sf_edit[idx_range] = sf_edit[idx_range] - gauss_fit

plt.plot(en_keV[idx_range], spectrum_filtered[idx_range])
# plt.plot(en_keV[idx_range], sf_edit[idx_range] - gauss_fit)
plt.plot(en_keV[idx_range], sf_edit[idx_range])
plt.plot(x, gauss_fit)
plt.grid()
plt.show()

plt.plot(en_keV, spectrum_filtered)
plt.plot(en_keV, sf_edit)
plt.grid()
plt.yscale('log')

# %%
plt.plot(en_keV, sf_edit)
plt.grid()
plt.yscale('log')

# %%
np.save('spectrum_edited', sf_edit)

# %%

# %%

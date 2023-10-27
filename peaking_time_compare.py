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
spectrum_1_2 = extract_data('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/peaking time spectrums/20231013 -- Mo-tube-poly-45kV (ptime 1.2us) .mca')
spectrum_11_2 = extract_data('/Users/grimax/Documents/Science/xtomo/poly_tomo_calc/peaking time spectrums/20231013 -- Mo-tube-poly-45kV (ptime 11.2us) .mca')

spectrum_1_2 = sp.ndimage.gaussian_filter(spectrum_1_2, sigma=1)
spectrum_11_2 = sp.ndimage.gaussian_filter(spectrum_11_2, sigma=1)

en_step = (19.608 - 17.479) / (348 - 309)
en_keV = np.array([17.479 + (i - 309) * en_step for i in np.arange(spectrum_1_2.shape[0])])

plt.figure(figsize=(15, 5))
plt.plot(en_keV, spectrum_1_2)
plt.plot(en_keV, spectrum_11_2)
plt.yscale('log')

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="NjXlyrXB3MPF" outputId="6f8409ee-9693-409a-9532-4ad2d7d7db43"
plt.plot(en_keV, spectrum_1_2 - spectrum_11_2)
# plt.yscale('log')

# %%

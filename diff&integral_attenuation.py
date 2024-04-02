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
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import spekpy



# %%
input_path = 'Mo_spec_poly_50_energies.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_energies = np.load(f)

# spec_Mo_50_energies = spec_Mo_50_energies[spec_Mo_50_energies > 0]
spec_Mo_50_energies = spec_Mo_50_energies[27:]

input_path = 'Mo_spec_poly_50.npy'
with open(input_path, 'rb') as f:
    spec_Mo_50_0 = np.load(f)[-spec_Mo_50_energies.size:].astype(float)
    spec_Mo_50_0 /= spec_Mo_50_0.sum()


plt.figure(figsize=(10, 5))
plt.plot(spec_Mo_50_energies, spec_Mo_50_0, label='no filter')
plt.ylim(1e-5, 1e-1)
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

# %%
xraydb.add_material('SiC','SiC', 3.21)

# %%
att_SiC = xraydb.material_mu('SiC', spec_Mo_50_energies * 1000) / 10

plt.plot(spec_Mo_50_energies, att_SiC)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

# %%
poly_mu_SiC = (att_SiC * spec_Mo_50_0).sum()
print(poly_mu_SiC)


# %%
def calc_slope(a, b, alpha, x):
    x_delta = 0.001
    y0 = a + b / x**alpha
    y1 = a + b / (x - x_delta / 2)**alpha
    y2 = a + b / (x + x_delta / 2)**alpha
    y_slope = (y2 - y1) / x_delta
    return y0, y_slope



# %%
voxel_size = 0.1 # in mm
total_lenght = 11 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_SiC_at_depths = np.exp(np.outer(-att_SiC, length_ticks)).T

passed_spectrums_50_0 = transmissions_SiC_at_depths * spec_Mo_50_0
p_specs_norm_50_0 = (passed_spectrums_50_0.T / np.sum(passed_spectrums_50_0, axis=1)).T
passed_intensity = np.sum(passed_spectrums_50_0, axis=1)
attenuation = -np.log(passed_intensity)

# plt.plot(length_ticks, np.sum(passed_spectrums_50_0, axis=1), label='50_0')
# plt.scatter(length_ticks[::10], np.sum(passed_spectrums_50_0, axis=1)[::10], marker='o')
# plt.grid()

# plt.figure(figsize=(10, 5))

plt.plot((0, length_ticks[31]), (0, attenuation[31]), c='green', linewidth=0.5)
plt.plot((0, 3), (0, 0), c='green', linewidth=0.5)
plt.plot((3, 3), (0, attenuation[31]), c='green', linewidth=0.5)
theta2 = 9 * 2 * np.pi * np.arctan(attenuation[31] / length_ticks[31])
plt.gca().add_patch(Arc((0, 0), 1.5, 0.56, theta1=0, theta2=theta2, color='green', linewidth=0.5))
plt.text(1, 0.1, r'$\mu_{eff}$', fontsize=16, color='green')

plt.plot((length_ticks[30], length_ticks[55]), (attenuation[30], attenuation[30]), c='red', linewidth=0.5)
plt.axline((length_ticks[30], attenuation[30]), (length_ticks[31], attenuation[31]), c='red', linewidth=0.5)
theta2 = 9 * 2 * np.pi * np.arctan((attenuation[31]-attenuation[30]) / (length_ticks[31]-length_ticks[30]))
plt.gca().add_patch(Arc((length_ticks[30], attenuation[30]), 1.5, 0.56, theta1=0, theta2=theta2, color='red', linewidth=0.5))
plt.text(4.1, 1.34, r'$\bar\mu$', fontsize=16, color='red')

plt.plot(length_ticks, attenuation)
plt.scatter(length_ticks[::10], attenuation[::10], marker='o')

plt.xlabel('Толщина, мм', fontsize=14)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=14)
plt.xlim(-0.5, 10.5)
plt.grid()
# plt.savefig('Fig3a.eps', dpi=600)


# %%
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot((0, length_ticks[31]), (0, attenuation[31]), c='black', lw=1, ls='dotted')
plt.plot((0, 3), (0, 0), c='black', lw=1, ls='dotted')
plt.plot((3, 3), (0, attenuation[31]), c='black', lw=1, ls='dotted')
theta2 = 9 * 2 * np.pi * np.arctan(attenuation[31] / length_ticks[31])
plt.gca().add_patch(Arc((0, 0), 1.5, 0.56, theta1=0, theta2=theta2, color='black', lw=0.5, ls='dotted'))
plt.text(1, 0.1, r'$\mu_{eff}$', fontsize=16, color='black')

plt.plot((length_ticks[30], length_ticks[55]), (attenuation[30], attenuation[30]), c='black', lw=1, ls='dashed')
plt.axline((length_ticks[30], attenuation[30]), (length_ticks[31], attenuation[31]), c='black', lw=1, ls='dashed')
theta2 = 9 * 2 * np.pi * np.arctan((attenuation[31]-attenuation[30]) / (length_ticks[31]-length_ticks[30]))
plt.gca().add_patch(Arc((length_ticks[30], attenuation[30]), 1.5, 0.56, theta1=0, theta2=theta2, color='black', lw=1, ls='dashed'))
plt.text(4.1, 1.34, r'$\bar\mu$', fontsize=16, color='black')

plt.plot(length_ticks, attenuation, c='black', lw=2)
plt.scatter(length_ticks[::10], attenuation[::10], marker='o', c='black')

plt.xlabel('Толщина, мм', fontsize=28)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=28)

plt.tick_params(direction='in', labelsize='14')

plt.xlim(-0.5, 10.5)

plt.tight_layout()
# plt.savefig('Fig3a.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

# %%
xraydb.add_material('GOS', 'Gd2O2S', 7.34)
GOS_mus_50 = xraydb.material_mu('GOS', spec_Mo_50_energies * 1000) / 10
GOS_t_50 = np.exp(-GOS_mus_50 * 22 * 0.001) # (22 * 0.001)mm == 22µm

qe = 60 # photon/keV
GOS_n_p_50 = spec_Mo_50_energies * qe
GOS_eff_50 = GOS_n_p_50 * (1 - GOS_t_50)

voxel_size = 0.1 # in mm
total_lenght = 11 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_SiC_at_depths = np.exp(np.outer(-att_SiC, length_ticks)).T

passed_spectrums_50_0 = transmissions_SiC_at_depths * spec_Mo_50_0
passed_intensity = np.sum(passed_spectrums_50_0, axis=1)
attenuation = -np.log(passed_intensity)

spec_Mo_GOS = spec_Mo_50_0 * GOS_eff_50
spec_Mo_GOS /= spec_Mo_GOS.sum()

passed_spectrums_GOS = transmissions_SiC_at_depths * spec_Mo_GOS
passed_intensity_GOS = np.sum(passed_spectrums_GOS, axis=1)
attenuation_GOS = -np.log(passed_intensity_GOS)

GOS_t_50_1 = np.exp(-GOS_mus_50 * 10 * 0.001)
GOS_eff_50_1 = GOS_n_p_50 * (1 - GOS_t_50_1)

spec_Mo_GOS_1 = spec_Mo_50_0 * GOS_eff_50_1
spec_Mo_GOS_1 /= spec_Mo_GOS_1.sum()

passed_spectrums_GOS_1 = transmissions_SiC_at_depths * spec_Mo_GOS_1
passed_intensity_GOS_1 = np.sum(passed_spectrums_GOS_1, axis=1)
attenuation_GOS_1 = -np.log(passed_intensity_GOS_1)

xraydb.add_material('CsI', 'CsI', 4.51)
CsI_mus = xraydb.material_mu('CsI', spec_Mo_50_energies * 1000) / 10
CsI_t = np.exp(-CsI_mus * 150 * 0.001)

CsI_qe = 65
CsI_n_p = spec_Mo_50_energies * CsI_qe
CsI_eff = CsI_n_p * (1 - CsI_t)

spec_Mo_CsI = spec_Mo_50_0 * CsI_eff
spec_Mo_CsI /= spec_Mo_CsI.sum()

passed_spectrums_CsI = transmissions_SiC_at_depths * spec_Mo_CsI
passed_intensity_CsI = np.sum(passed_spectrums_CsI, axis=1)
attenuation_CsI = -np.log(passed_intensity_CsI)


plt.plot(length_ticks, attenuation, label='flat')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o')

plt.plot(length_ticks, attenuation_GOS, label='GadOx 22µm')
plt.scatter(length_ticks[::10], attenuation_GOS[::10], marker='o')

plt.plot(length_ticks, attenuation_GOS_1, label='GadOx 10µm')
plt.scatter(length_ticks[::10], attenuation_GOS_1[::10], marker='o')

plt.plot(length_ticks, attenuation_CsI, label='CsI 150µm')
plt.scatter(length_ticks[::10], attenuation_CsI[::10], marker='o')

plt.xlabel('Толщина, мм', fontsize=14)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=14)
plt.xlim(-0.5, 10.5)
plt.grid()
plt.legend(framealpha=1)
# plt.savefig('Fig5b.eps', dpi=600)
plt.show()

# plt.axhline(100)
plt.plot([spec_Mo_50_energies[0], 50], [700, 700], label='flat')
plt.plot(spec_Mo_50_energies, GOS_eff_50, label='GadOx 22µm')
plt.plot(spec_Mo_50_energies, GOS_eff_50_1, label='GadOx 10µm')
plt.plot(spec_Mo_50_energies, CsI_eff, label='CsI 150µm')
plt.xlabel('Энергия, кэВ', fontsize=14)
plt.ylabel('Число фотонов', fontsize=14)
plt.xlim(0, 50)
plt.legend(framealpha=1)
plt.grid()
# plt.savefig('Fig5a.eps', dpi=600)


# %%
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot(length_ticks, attenuation, label='flat', c='black')
# plt.scatter(length_ticks[::10], attenuation[::10], marker='o', c='black')
plt.text(9.5, 2.65, '4', fontstyle='italic', fontsize='12')

plt.plot(length_ticks, attenuation_GOS, label='GadOx 22µm', ls='dashed', lw=1, c='black')
plt.scatter(length_ticks[::10], attenuation_GOS[::10], marker='o', c='black')
plt.text(8.5, 2.75, '1', fontstyle='italic', fontsize='12')

plt.plot(length_ticks, attenuation_GOS_1, label='GadOx 10µm', ls='dotted', lw=1, c='black')
plt.scatter(length_ticks[::10], attenuation_GOS_1[::10], marker='s', c='black')
plt.text(7.2, 3.1, '2', fontstyle='italic', fontsize='12')

plt.plot(length_ticks, attenuation_CsI, label='CsI 150µm', ls='dashdot', lw=1, c='black')
plt.scatter(length_ticks[::10], attenuation_CsI[::10], marker='^', c='black')
plt.text(9.5, 1.9, '3', fontstyle='italic', fontsize='12')

plt.xlabel('Толщина, мм', fontsize=28)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=28)
plt.xlim(-0.5, 10.5)

plt.tick_params(direction='in', labelsize='14')
plt.tight_layout()
# plt.savefig('Fig5b.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

plt.show()

ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot([spec_Mo_50_energies[0], 50], [700, 700], label='flat', c='black')
plt.text(45, 775, '4', fontstyle='italic', fontsize='12')

plt.plot(spec_Mo_50_energies, GOS_eff_50, label='GadOx 22µm', ls='dashed', lw=1, c='black')
plt.text(35, 350, '1', fontstyle='italic', fontsize='12')

plt.plot(spec_Mo_50_energies, GOS_eff_50_1, label='GadOx 10µm', ls='dotted', lw=1, c='black')
plt.text(15, 200, '2', fontstyle='italic', fontsize='12')

plt.plot(spec_Mo_50_energies, CsI_eff, label='CsI 150µm', ls='dashdot', lw=1, c='black')
plt.text(25, 1100, '3', fontstyle='italic', fontsize='12')

plt.xlabel('Энергия, кэВ', fontsize=28)
plt.ylabel('Число фотонов', fontsize=28)
plt.xlim(0, 50)

plt.tick_params(direction='in', labelsize='14')
plt.tight_layout()
# plt.savefig('Fig5a.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

plt.show()


# %%
att_k_eff = np.append([np.inf], attenuation[1:] / length_ticks[1:])
att_k_diff = np.append([np.inf], attenuation[1:] - attenuation[:-1]) / voxel_size

fig, ax = plt.subplots(2, 1, figsize=(5, 5))

ax[0].plot(length_ticks, att_k_eff, c='green')
ax[0].scatter(length_ticks[::10], att_k_eff[::10], marker='o', c='green')
ax[0].grid()
ax[0].tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
ax[0].set_ylabel(r'$µ_{eff}, mm^{-1}$')
ax[0].set_xlim(-0.5, 10.5)
ax[0].set_ylim(0, 1.2)

ax[1].plot(length_ticks, att_k_diff, c='red')
ax[1].scatter(length_ticks[::10], att_k_diff[::10], marker='o', c='red')
ax[1].set_ylabel(r'$\bar µ, mm^{-1}$')
ax[1].set_xlabel('Толщина, мм')
ax[1].set_xlim(-0.5, 10.5)
ax[1].set_ylim(0, 1.1)
ax[1].grid()

plt.show()

plt.plot(length_ticks, att_k_eff, c='green', label=r'$µ_{eff}$')
plt.scatter(length_ticks[::10], att_k_eff[::10], marker='o', c='green')
plt.plot(length_ticks, att_k_diff, c='red', label=r'$\bar µ$')
plt.scatter(length_ticks[::10], att_k_diff[::10], marker='o', c='red')
plt.axline((0, att_SiC[844]), slope=0, c='darkviolet', label=r'$µ_{SiC} @ 50 KeV$')
plt.xlim(-0.5, 10.5)
plt.ylim(0, 1.1)
plt.ylabel(r'$mm^{-1}$')
plt.xlabel('Толщина, мм')
plt.legend(framealpha=1)
plt.grid()
# plt.savefig('Fig3b.eps', dpi=600)
plt.show()

# %%
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot(length_ticks, att_k_eff, c='black', lw=1, ls='dotted', label=r'$µ_{eff}$')
plt.scatter(length_ticks[::10], att_k_eff[::10], marker='o', c='black')
plt.plot(length_ticks, att_k_diff, c='black', lw=1, ls='dashed', label=r'$\bar µ$')
plt.scatter(length_ticks[::10], att_k_diff[::10], marker='s', c='black')
plt.axline((0, att_SiC[844]), slope=0, c='black', lw=1, label=r'$µ_{SiC} @ 50 KeV$')
plt.text(2.5, 0.5, '1', fontsize=12, fontstyle='italic')
plt.text(1.75, 0.37, '2', fontsize=12, fontstyle='italic')
plt.text(1, 0.15, '3', fontsize=12, fontstyle='italic')
plt.xlim(-0.5, 10.5)
plt.ylim(0, 1.1)
plt.ylabel(r'$мм^{-1}$', fontsize=28)
plt.xlabel('Толщина, мм', fontsize=28)
plt.tick_params(direction='in', labelsize='14')
plt.tight_layout()

# plt.savefig('Fig3b.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()

# %%
plt.plot(spec_Mo_50_energies, att_SiC)
plt.yscale('log')
# plt.xscale('log')
plt.grid()
plt.show()

# %%
print(spec_Mo_50_energies[844])

print(att_SiC[844])

# %%
default_blue_color = u'#1f77b4'

voxel_size = 0.1 # in mm
total_lenght = 11 # 1cm
length_ticks = np.arange(0, total_lenght, voxel_size)

transmissions_SiC_at_depths = np.exp(np.outer(-att_SiC, length_ticks)).T

passed_spectrums_50_0 = transmissions_SiC_at_depths * spec_Mo_50_0
passed_intensity = np.sum(passed_spectrums_50_0, axis=1)
attenuation = -np.log(passed_intensity)

en_step = np.mean(spec_Mo_50_energies[1:] - spec_Mo_50_energies[:-1])

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
# s.filter('Air', 140)
s.filter('Air', 250)

model_energies_0, model_intensities_0 = s.get_spectrum()
model_intensities_0 /= model_intensities_0.sum()
model_att_SiC_0 = xraydb.material_mu('SiC', model_energies_0 * 1000) / 10
model_transmissions_SiC_at_depths_0 = np.exp(np.outer(-model_att_SiC_0, length_ticks)).T

model_passed_spectrums_50_0_0 = model_transmissions_SiC_at_depths_0 * model_intensities_0
model_passed_intensity_0 = np.sum(model_passed_spectrums_50_0_0, axis=1)
model_attenuation_0 = -np.log(model_passed_intensity_0)

s = spekpy.Spek(kvp=50, dk=en_step, targ='Mo')
s.filter('Air', 1440)
model_energies, model_intensities = s.get_spectrum()
model_intensities /= model_intensities.sum()
model_att_SiC = xraydb.material_mu('SiC', model_energies * 1000) / 10
model_transmissions_SiC_at_depths = np.exp(np.outer(-model_att_SiC, length_ticks)).T

model_passed_spectrums_50_0 = model_transmissions_SiC_at_depths * model_intensities
model_passed_intensity = np.sum(model_passed_spectrums_50_0, axis=1)
model_attenuation = -np.log(model_passed_intensity)

plt.plot(length_ticks, attenuation, label='Экспериментальный спектр')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o')

# plt.plot(length_ticks, model_attenuation_0, label='Расчётный спектр 0')

plt.plot(length_ticks, model_attenuation, linestyle=(0, (2, 1)), c=default_blue_color, label='Расчётный спектр')
plt.scatter(length_ticks[::10], model_attenuation[::10], facecolors='none', edgecolors=default_blue_color)

plt.xlabel('Толщина, мм', fontsize=14)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=14)
plt.xlim(-0.5, 10.5)
plt.grid()
plt.legend(framealpha=1)
# plt.savefig('Fig4b.eps', dpi=600)
plt.show()


# %%
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.plot(length_ticks, attenuation, label='Экспериментальный спектр', c='black')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o', c='black')
plt.text(9.5, 2.2, '1', fontstyle='italic', fontsize=12)

plt.plot(length_ticks, model_attenuation, linestyle=(0, (2, 1)), label='Расчётный спектр', c='black')
plt.scatter(length_ticks[::10], model_attenuation[::10], facecolors='none', edgecolors='black')
plt.text(9.5, 3.9, '2', fontstyle='italic', fontsize=12)

plt.xlabel('Толщина, мм', fontsize=28)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=28)
plt.xlim(-0.5, 10.5)

plt.tick_params(direction='in', labelsize='14')
plt.tight_layout()

# plt.savefig('Fig4b.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()


# %%
plt.plot(model_energies, model_intensities)
plt.plot(model_energies_0, model_intensities_0)
plt.plot(spec_Mo_50_energies, spec_Mo_50_0)
plt.yscale('log')
plt.ylim(1e-5, 1)

# %%
with open('SiC_lengths.npy', 'rb') as f:
    SiC_lengths = np.load(f)

with open('SiC_att_0.npy', 'rb') as f:
    SiC_att_0 = np.load(f)

# %%
plt.scatter(SiC_lengths, SiC_att_0, s=1, marker='.', c='gray', label='Экспериментальные данные')

plt.plot(length_ticks, attenuation, label='Моделирование: Экспериментальный спектр')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o')

# plt.plot(length_ticks, model_attenuation_0, label='Расчётный спектр 0')

plt.plot(length_ticks, model_attenuation, linestyle=(0, (2, 1)), c=default_blue_color, label='Моделирование: Расчётный спектр')
plt.scatter(length_ticks[::10], model_attenuation[::10], facecolors='none', edgecolors=default_blue_color)

plt.xlabel('Толщина, мм', fontsize=14)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=14)
plt.xlim(-0.5, 7.5)
plt.grid()
plt.legend(framealpha=1)
plt.title('Карбид кремния, SiC')
# plt.savefig('Fig9a.eps', dpi=600)


# %%
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)

plt.scatter(SiC_lengths, SiC_att_0, s=1, marker='.', c='gray', label='Экспериментальные данные')
plt.text(1, 2, '1', fontstyle='italic', fontsize=12)

plt.plot(length_ticks, attenuation, label='Моделирование: Экспериментальный спектр', c='black')
plt.scatter(length_ticks[::10], attenuation[::10], marker='o', c='black')
plt.text(1.3, 1.15, '3', fontstyle='italic', fontsize=12)

plt.plot(length_ticks, model_attenuation, linestyle='dashed', c='black', label='Моделирование: Расчётный спектр')
plt.scatter(length_ticks[::10], model_attenuation[::10], facecolors='none', edgecolors='black')
plt.text(1.5, 0.5, '2', fontstyle='italic', fontsize=12)

plt.xlabel('Толщина, мм', fontsize=28)
plt.ylabel(r'$–ln\frac{\Phi (x)}{\Phi _0}$', fontsize=28)
plt.xlim(-0.5, 7.5)

plt.title('Карбид кремния, SiC', fontsize=28)

plt.tick_params(direction='in', labelsize='14')
plt.tight_layout()
# plt.savefig('Fig9a.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})


# %%

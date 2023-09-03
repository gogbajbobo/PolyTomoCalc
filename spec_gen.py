import numpy as np
import xraydb

# following code based on https://github.com/ingacheva/XRayUtil


def generate_spectrum(i, u, M_a, num_points=100, energies=None):
    # :param i: (float): Current in [mA]
    # :param u: (float): Voltage [kV]

    if energies is None and num_points is None:
        print('incomplete parameters')
        return []

    info_tube = {'Cr': (24, 1.97, 2.39, 1.0, 0.50, 0.15, 0.15, 1100),
                 'Cu': (29, 2.40, 2.98, 1.0, 0.51, 0.17, 0.17, 1300),
                 'Mo': (42, 6.42, 6.66, 1.0, 0.52, 0.15, 0.03, 1700),
                 'Ag': (47, 8.60, 8.90, 1.0, 0.53, 0.16, 0.04, 2000),
                 'W': (74, 8.60, 8.90, 1.0, 0.58, 0.22, 0.08, 2500)}

    z, w1, w2, k_a1, k_a2, k_b1, k_b2, const = info_tube[M_a]
    if energies is None:
        energies = np.linspace(u * 1e-3, u, num_points)
    spectrum = np.zeros_like(energies)

    c = 2.8
    lambda_0 = 12.398 / u  # Critical wavelength [A]
    lambdas = 12.398 / energies
    spectrum += c * c / lambda_0 * i * 0.001 * z * (lambdas - lambda_0) / lambdas ** 3

    w1 *= 1e-3
    w2 *= 1e-3

    energy = xraydb.xray_lines(M_a, 'K')['Ka1'].energy / 1000
    if energy < u:
        intensity = k_a1 * const * i * 0.001 * np.power(u - energy, 1.5)
        spectrum += intensity * (w1 / 2) ** 2 / ((12.398 / energies - 12.398 / energy) ** 2 + (w1 / 2) ** 2)

    energy = xraydb.xray_lines(M_a, 'K')['Ka2'].energy / 1000
    if energy < u:
        intensity = k_a2 * const * i * 0.001 * np.power(u - energy, 1.5)
        spectrum += intensity * (w1 / 2) ** 2 / ((12.398 / energies - 12.398 / energy) ** 2 + (w1 / 2) ** 2)

    energy = xraydb.xray_lines(M_a, 'K')['Kb1'].energy / 1000
    if energy < u:
        intensity = k_b1 * const * i * 0.001 * np.power(u - energy, 1.5)
        spectrum += intensity * (w2 / 2) ** 2 / ((12.398 / energies - 12.398 / energy) ** 2 + (w1 / 2) ** 2)

    if z != 24 and z != 29:
        energy = xraydb.xray_lines(M_a, 'K')['Kb2'].energy / 1000
        if energy < u:
            intensity = k_b2 * const * i * 0.001 * np.power(u - energy, 1.5)
            spectrum += intensity * (w2 / 2) ** 2 / ((12.398 / energies - 12.398 / energy) ** 2 + (w1 / 2) ** 2)

    return energies, spectrum

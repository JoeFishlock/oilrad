import matplotlib.pyplot as plt
import numpy as np
from gulf.black_body import solar_irradiance


plt.figure()
wavelengths = np.linspace(350, 700, 1000)
plt.plot(
    wavelengths,
    solar_irradiance(wavelengths, environment_conditions=0),
    label="atm top",
)
plt.plot(
    wavelengths,
    solar_irradiance(wavelengths, environment_conditions=3),
    label="hazy 60 deg zenith",
)
plt.legend()
plt.grid(True)
plt.ylim([0, 2])
plt.xlabel("Wavelength nm")
plt.ylabel("Plane downwelling irradiance W/m2 nm")
plt.show()

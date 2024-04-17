from gulf.infinite_layer import InfiniteLayerModel
import numpy as np
import matplotlib.pyplot as plt

ICE_TYPE = "FYI"
H = 2
uniform_conc = 1000


plt.figure()
z = np.linspace(-H, 0, 1000)

oil_func = lambda z: 2 * uniform_conc * ((z / H) + 1)
model = InfiniteLayerModel(oil_func, ice_thickness=H, ice_type=ICE_TYPE)
upwelling, downwelling = model.get_upwelling_and_downwelling
plt.plot(downwelling(z, 400), z, "b")
plt.plot(upwelling(z, 400), z, "b--")
plt.plot(downwelling(z, 700), z, "r")
plt.plot(upwelling(z, 700), z, "r--")

# oil_func = lambda _: uniform_conc
oil_func = lambda _: uniform_conc
model = InfiniteLayerModel(oil_func, ice_thickness=H, ice_type=ICE_TYPE)
upwelling, downwelling = model.get_upwelling_and_downwelling
plt.plot(downwelling(z, 400), z, "b:")
plt.plot(upwelling(z, 400), z, "b-.")
plt.plot(downwelling(z, 700), z, "r:")
plt.plot(upwelling(z, 700), z, "r-.")

plt.show()

plt.figure()
oil_func = lambda z: 2 * uniform_conc * ((z / H) + 1)
model = InfiniteLayerModel(oil_func, ice_thickness=H, ice_type=ICE_TYPE)
wavelengths = np.linspace(350, 700, 1000)
albedo = np.array([model.albedo(wavelength) for wavelength in wavelengths])
plt.plot(wavelengths, albedo, label="linear")

oil_func = lambda _: uniform_conc
model = InfiniteLayerModel(oil_func, ice_thickness=H, ice_type=ICE_TYPE)
upwelling, downwelling = model.get_upwelling_and_downwelling
albedo = np.array([model.albedo(wavelength) for wavelength in wavelengths])
plt.plot(wavelengths, albedo, label="uniform")
plt.show()

plt.figure()
oil_func = lambda z: 2 * uniform_conc * ((z / H) + 1)
plt.plot(oil_func(z), z, "k")
plt.show()

from gulf.infinite_layer import InfiniteLayerModel
import numpy as np
import matplotlib.pyplot as plt

ICE_TYPE = "FYI"
H = 2
uniform_conc = 1000


z = np.linspace(-H, 0, 1000)
wavelengths = np.linspace(350, 700, 1000)

linear_oil_func = lambda z: 2 * uniform_conc * ((z / H) + 1)
constant_oil_func = lambda _: uniform_conc

linear_oil_model = InfiniteLayerModel(
    linear_oil_func, ice_thickness=H, ice_type=ICE_TYPE
)
uniform_oil_model = InfiniteLayerModel(
    constant_oil_func, ice_thickness=H, ice_type=ICE_TYPE
)


plt.figure()
plt.title(f"planar irradiances for {ICE_TYPE} at {H}m thick, 400nm (blue), 700nm (red)")

upwelling, downwelling = linear_oil_model.get_upwelling_and_downwelling
plt.plot(downwelling(z, 400), z, "b", label=f"linear oil conc")
plt.plot(upwelling(z, 400), z, "b", alpha=0.2)
plt.plot(downwelling(z, 700), z, "r")
plt.plot(upwelling(z, 700), z, "r", alpha=0.2)

upwelling, downwelling = uniform_oil_model.get_upwelling_and_downwelling
plt.plot(downwelling(z, 400), z, "b--", label=f"uniform {uniform_conc}ng oil/g ice")
plt.plot(upwelling(z, 400), z, "b--", alpha=0.2)
plt.plot(downwelling(z, 700), z, "r--")
plt.plot(upwelling(z, 700), z, "r--", alpha=0.2)
plt.legend()
plt.grid(True)
plt.xlabel("non dimensional irradiance")
plt.ylabel("depth (m)")
plt.savefig("figures/infinite_layer/upwelling_downwelling_irradiance.pdf")

plt.figure()
plt.title("spectral albedo")
for model, label in zip(
    [linear_oil_model, uniform_oil_model], ["linear oil conc", "uniform oil conc"]
):
    plt.plot(wavelengths, model.albedo(wavelengths), label=label)
plt.xlabel("wavelength (nm)")
plt.ylabel("spectral albedo")
plt.legend()
plt.grid(True)
plt.savefig("figures/infinite_layer/spectral_albedo.pdf")

"""This script creates a noisy vertical oil mass ratio profile interpolated on a 
fairly coarse grid as may be the case in a sea ice model. It then constructs a smooth
version analytically and plots the solution to the infinite layer two stream model 
for these two cases and a case with no oil.

The conclusion is that the radiative heating term will be jagged like the input oil
mass ratio profile bu the upwelling and downwelling irrradiances are smoothed.
This shouldn't matter as this heating will undergo thermal diffusion and so temperature
variations will be smoothed"""

import numpy as np
import matplotlib.pyplot as plt
from oilrad.infinite_layer import InfiniteLayerModel

H = 2
ICE_TYPE = "FYI"
DROPLET_RADIUS = 0.5

sample_depths = np.linspace(-H, 0, 25)
sample_oil = np.maximum(
    0,
    (1000 * (sample_depths + H) ** 2 / 4)
    * (1 + 2 * np.sin(np.pi * 500 * sample_depths) * np.sin(1000 * sample_depths)),
)
plt.figure()
plt.plot(sample_oil, sample_depths)
plt.show()
plt.close()
interpolated_oil_func = lambda z: np.interp(z, sample_depths, sample_oil)
interpolated_oil_model = InfiniteLayerModel(
    interpolated_oil_func, H, ICE_TYPE, DROPLET_RADIUS
)

smooth_oil_func = lambda z: 1000 * (z + H) ** 2 / 4
smooth_oil_model = InfiniteLayerModel(smooth_oil_func, H, ICE_TYPE, DROPLET_RADIUS)

base = InfiniteLayerModel(lambda z: 0, H, ICE_TYPE, DROPLET_RADIUS)

z = np.linspace(-H, 0, 1000)

plt.figure()
for model in [interpolated_oil_model, base, smooth_oil_model]:
    plt.plot(model.downwelling(z, 350), z)
    plt.plot(model.upwelling(z, 350), z, ls="--")
plt.show()
plt.close()

plt.figure()
for model in [interpolated_oil_model, base, smooth_oil_model]:
    plt.plot(model.heating(z, 350), z)

plt.show()
plt.close()

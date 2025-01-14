# oilrad #

Oilrad is a python package for solving a two-stream shortwave radiation transfer model in a vertical domain containing sea ice and ocean water.
It solves the spectral two-stream equations for ice and seawater polluted with a mass concentration of oil droplets, where oil absorption is calculated from [1].
Below is an example of spectral albedo computed for a range of oil mass ratios in a layer of sea ice.

![Sea ice spectral albedo for ice containing droplets of Romashkino oil](https://github.com/JoeFishlock/oilrad/blob/40f616a5fce75f0dd3bebc9f55508cf4c827ae0f/docs/example_spectral_albedo.svg)

## Installation ##
Oilrad is available on PyPI and can be installed with pip via
```bash
pip install oilrad
```

## Usage ##
The following python code illustrates how to set up and solve a simulation.
```python
import numpy as np
import oilrad as oi

# set up model parameters
ICE_DEPTH = 0.8
depth = np.linspace(-1.5, 0, 1000)
model = oi.CtsWavelengthModel(
    z=depth,
    wavelengths=np.geomspace(350, 3000, 100),
    oil_mass_ratio=np.full_like(depth, 1000),
    ice_scattering_coefficient=1.5,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(depth > -ICE_DEPTH, 0, 1),
)

# solve the model
spectral_solution = oi.solve_two_stream_model(model)

# plot the spectral albedo
import matplotlib.pyplot as plt

plt.figure()
plt.plot(spectral_solution.wavelengths, spectral_solution.albedo)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.show()

# integrate the solution over the black body spectrum to find total albedo
integrated_solution = oi.integrate_over_SW(
    spectral_solution, oi.BlackBodySpectrum(350, 3000)
)
print(f"total albedo={integrated_solution.albedo}")
```

## Tests ##
The tests directory contains tests of simulations for a variety of input parameters.
Run the tests using `pytest` from the root directory of the repository.

## Docs ##
API reference documentation built using `mkdocs gh-deploy` is available at
[documentation](https://JoeFishlock.github.io/oilrad).

## License ##
[MIT](https://choosealicense.com/licenses/mit/)

## References ##
[1] B. H. Redmond Roche and M. D. King, ‘Quantifying the effects of background concentrations of crude oil pollution on sea ice albedo’, The Cryosphere, vol. 16, no. 10, pp. 3949–3970, Oct. 2022, doi: 10.5194/tc-16-3949-2022.

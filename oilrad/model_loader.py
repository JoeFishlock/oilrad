"""Define an interface here that all solutions of the two stream radiation model
implement. Specifically once a model is initialised from its required parameters it
will provide methods to determine the upwelling radiation, downwelling radiation and
the radiative heating as functions of depth and wavelength. It will also provide
methods for the spectral albedo and transmission."""

from oilrad.single_layer import SingleLayerModel
from oilrad.two_layer import TwoLayerModel
from oilrad.infinite_layer import InfiniteLayerModel


def two_stream_model(model_choice: str, **kwargs):
    MODELS = {"1L": SingleLayerModel, "2L": TwoLayerModel, "IL": InfiniteLayerModel}
    return MODELS[model_choice](**kwargs)

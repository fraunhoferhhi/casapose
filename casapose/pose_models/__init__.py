# import keras_applications as ka
# from .__version__ import __version__


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get("backend")
    layers = kwargs.get("layers")
    models = kwargs.get("models")
    utils = kwargs.get("utils")
    return backend, layers, models, utils


import numpy


__all__ = ['initialize_network']


def initialize_network(n_inputs, n_hidden_neurons, n_outputs):
    r"""Initializes a neural network.

    Generates a neural network that takes `n_inputs`, contains `n_hidden_neurons` layers, and generates `n_outputs`.

    Parameters
    ----------
    n_inputs : int
        The number of inputs of the neural network.
    n_hidden_neurons : int
        The number of hidden neurons of the neural network with one hidden layer.
    n_outputs : int
        The number of outputs of the neural network.

    Returns
    -------
    neural_network : list of dict of numpy.ndarray
        Contains the definition of the neural network, with
    """
    print('Initializing network...')

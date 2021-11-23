
import numpy

__all__ = ['initialize_network']


def initialize_network(n_inputs, n_hidden_neurons, n_outputs):
    r"""Initializes a neural network.

    Generates a neural network that takes `n_inputs`, contains `n_hidden_neurons` layers, and generates `n_outputs`.

    Parameters
    ----------
    n_inputs : int
        The number of inputs of the neural network.
    n_hidden_neurons : numpy.ndarray of int
        The number of hidden neurons per hidden layer of the neural network.
        n_hidden_neurons[i] contains the number of neuron of the ith hidden layer.
        (size: [n_layers,])
    n_outputs : int
        The number of outputs of the neural network.

    Returns
    -------
    neural_network : list of dict of numpy.ndarray of float
        Contains the definition of the neural network, with
    """

    # Preallocate space for the list that contains weights and biases for all layers
    n_hidden_layers = n_hidden_neurons.shape[0]  # the number of hidden layers
    n_sets_of_weights_and_biases = n_hidden_layers + 1  # data for the hidden layers and for the output
    network_weights = [None] * n_sets_of_weights_and_biases
    network_biases = [None] * n_sets_of_weights_and_biases

    # Initialize the weights for the layer 0 which takes the inputs
    network_weights[0], network_biases[0] = initialize_layer(n_inputs, n_hidden_neurons[0])

    # Loop over the remaining layers and initialize them
    for layer_idx in range(1, n_hidden_layers):
        network_weights[layer_idx], network_biases[layer_idx] = \
            initialize_layer(n_hidden_neurons[layer_idx - 1], n_hidden_neurons[layer_idx])

    # Initialize the output layer
    network_weights[-1], network_biases[-1] = initialize_layer(n_hidden_neurons[-1], n_outputs)

    # Return the network
    return [network_weights, network_biases]


def initialize_layer(n_inputs, n_neurons):
    r"""Initializes a layer of a neural network.

    For a layer with n_inputs and n_neurons it generates the n_neurons x n_inputs array of weights
    and the n_neurons vector of biases. Weights and biases are all initialized to random values.

    Parameters
    ----------
    n_inputs : int
        The number of inputs to the layer.
    n_neurons : int
        The number of neurons in the layer.

    Returns
    -------
    layer_data : list of numpy.ndarray
        Contains weights and the offsets for the layer. layer_data[0] contains the
        n_neurons x n_inputs array of weights and layer_data[1] contains the vector
        of biases of dimension n_neurons.
    """

    # Initialize the weights and biases as random values
    weights = numpy.random.rand(n_neurons, n_inputs)
    biases = numpy.random.rand(n_neurons, 1)

    # Package them as a list
    return [weights, biases]

import pytest

import numpy

import nnfromscratch.__nnfromscratch


def test_initialize_layer():
    r"""Tests nnfromscratch.initialize_layer function.
    """

    # Test to check layer with different number of inputs and neurons
    n_inputs = 10
    n_neurons = 5

    layer_data = nnfromscratch.__nnfromscratch.initialize_layer(n_inputs, n_neurons)

    # Check neuron size
    if layer_data[0].shape[0] != n_neurons:
        pytest.fail(f'Number of neurons in layer not as expected, is {layer_data[0].shape[0]}, expected {n_neurons}')

    # Check input size
    if layer_data[0].shape[1] != n_inputs:
        pytest.fail(f'Number of inputs in layer not as expected, is {layer_data[0].shape[1]}, expected {n_inputs}')


def test_initialize_network():
    r"""Tests nnfromscratch.initialize_network function.
    """

    # Test to check network with one single layer of 10 inputs, 3 neurons, and 4 outputs
    n_inputs = 10
    n_hidden_neurons = numpy.array([3])
    n_outputs = 4

    nn_data = nnfromscratch.__nnfromscratch.initialize_network(n_inputs, n_hidden_neurons, n_outputs)

    # Pre-computation of expected parameters
    n_hidden_layers = n_hidden_neurons.shape[0]  # the number of hidden layers
    n_sets_of_weights_and_biases = n_hidden_layers + 1  # data for the hidden layers and for the output

    # Check number of sets of weights (number of hidden layers plus output layer
    if len(nn_data[0]) != n_sets_of_weights_and_biases:
        pytest.fail(f'Number of weights of hidden layers in neural network not as expected, is {len(nn_data[0])}, '
                    f'expected {n_sets_of_weights_and_biases}')

    # Check the number of sets of biases (number of hidden layers plus output layer
    if len(nn_data[1]) != n_sets_of_weights_and_biases:
        pytest.fail(f'Number of biases of hidden layers in neural network not as expected, is {len(nn_data[1])}, '
                    f'expected {n_sets_of_weights_and_biases}')

    # Check input size is correct for number of columns of weight matrix of layer 0
    if nn_data[0][0].shape[1] != n_inputs:
        pytest.fail(f'Number of inputs in neural network not as expected, is {nn_data[0].shape[1]}, '
                    f'expected {n_inputs}')

    # Check output size is correct for number of rows of weight matrix of last layer
    if nn_data[0][-1].shape[0] != n_outputs:
        pytest.fail(f'Number of outputs in neural network not as expected, is {nn_data[-1].shape[0]}, '
                    f'expected {n_outputs}')

    # Check weight matrix sizes of hidden layers

    # First check the rows of first layer
    layer_idx = 0
    if nn_data[0][layer_idx].shape[0] != n_hidden_neurons[layer_idx]:
        pytest.fail(f'Number of rows of weights of hidden layer {layer_idx} of neural network not as expected, '
                    f'is {nn_data[0][layer_idx].shape[0]}, expected {n_hidden_neurons[layer_idx]}')

    # Check the size of the weight matrix of other layers
    for layer_idx in range(1, n_hidden_layers):
        # First the number of rows
        if nn_data[0][layer_idx].shape[0] != n_hidden_neurons[layer_idx]:
            pytest.fail(f'Number of rows of weights of hidden layer {layer_idx} of neural network not as expected, '
                        f'is {nn_data[0][layer_idx].shape[0]}, expected {n_hidden_neurons[layer_idx]}')
        # Then the number of columns (neurons of previous layer)
        if nn_data[0][layer_idx].shape[1] != n_hidden_neurons[layer_idx - 1]:
            pytest.fail(f'Number of columns of weights of hidden layer {layer_idx} of neural network not as expected, '
                        f'is {nn_data[0][layer_idx].shape[1]}, expected {n_hidden_neurons[layer_idx - 1]}')

    # Check the size of number of columns of weight matrix of output layer
    layer_idx = -1
    if nn_data[0][-1].shape[1] != n_hidden_neurons[layer_idx]:
        pytest.fail(f'Number of columns of weights of hidden layer {layer_idx} of neural network not as expected, '
                    f'is {nn_data[0][-1].shape[1]}, expected {n_hidden_neurons[layer_idx]}')

    # Check number of biases in each layer
    for layer_idx in range(0, n_hidden_layers):
        if nn_data[1][layer_idx].shape[0] != n_hidden_neurons[layer_idx]:
            pytest.fail(f'Number of biases of hidden layer {layer_idx} of neural network not as expected, '
                        f'is {nn_data[1][layer_idx].shape[0]}, expected {n_hidden_neurons[layer_idx]}')

    # Check number of biases for output layer
    layer_idx = -1
    if nn_data[1][layer_idx].shape[0] != n_outputs:
        pytest.fail(f'Number of biases for output of neural network not as expected, '
                    f'is {nn_data[1][layer_idx].shape[0]}, expected {n_outputs}')

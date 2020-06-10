import numpy
import math


class Utils:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


class Layer:
    def __init__(self, number_of_neuron):
        self.neuronList = []

        if number_of_neuron:
            for neuronNumber in range(number_of_neuron):
                self.neuronList.append(Neuron())

        self.layer_link = None

    def link_to_layer(self, layer):
        self.layer_link = LayerLink(self, layer)

    def propagate(self, inputs):
        if self.layer_link is None:
            return inputs
        if len(inputs) != len(self.neuronList):
            raise ValueError("incorrect number of inputs")

        transferred_inputs = self.apply_transfert_to_inputs(inputs)

        return self.layer_link.combine_and_propagate(transferred_inputs)

    def apply_transfert_to_inputs(self, inputs):
        transferred_inputs = []
        input_neuron_index = 0
        for input_neuron in self.neuronList:
            input_value = inputs[input_neuron_index]
            transferred_inputs.append(input_neuron.transfert(input_value))

        return transferred_inputs


class LayerLink:
    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        # save output_layer for back propagation
        self.output_layer = output_layer

        self.weight_mat = self.init_weight_mat()

    def init_weight_mat(self):
        weight_mat = []

        for output_layer_neuron in self.output_layer.neuronList:
            horizontal_array = []
            for input_layer_neuron in self.input_layer.neuronList:
                horizontal_array.append(numpy.random.uniform(-0.25, 0.25))

            weight_mat.append(horizontal_array)

        return weight_mat

    def combine_and_propagate(self, inputs):

        outputs = []
        output_neuron_index = 0
        for output_layer_neuron in self.output_layer.neuronList:

            output_sum = 0
            input_neuron_index = 0
            for input_layer_neuron in self.input_layer.neuronList:
                transferred_input_value = inputs[input_neuron_index]
                weight_mat_value = self.weight_mat[output_neuron_index][input_neuron_index]

                sum_line_value = transferred_input_value * weight_mat_value

                output_sum = output_sum + sum_line_value
                input_neuron_index = input_neuron_index + 1

            outputs.append(output_sum)
            output_neuron_index = output_neuron_index + 1

        return self.output_layer.propagate(outputs)


class Neuron:
    def __init__(self):
        pass

    def transfert(self, input_value):
        return Utils.sigmoid(input_value)


inputs = [0.01, 0.25, 0.03]
print("Inputs:", inputs)

""" Level 1 """
print("=== Level 1 ===")
# create layers
layer_1 = Layer(3)
layer_2 = Layer(2)

# link to layer2
layer_1.link_to_layer(layer_2)

print("Outputs:", layer_1.propagate(inputs))

""" Level 2 """
print("=== Level 2 ===")
# create layers
layer_1 = Layer(3)
layer_2 = Layer(3)
layer_3 = Layer(2)

# link to layer2
layer_1.link_to_layer(layer_2)
layer_2.link_to_layer(layer_3)

print("Outputs:", layer_1.propagate(inputs))

# -*- coding: utf-8 -*-
import sys
from neural_networks import create_network
from helper import read_def_network, read_initial_weights, read_dataset
# importando bibliotecas auxiliares
import math
import numpy as np

def dot(weights, inputs):
    return sum(float(weight_i) * float(input_i) for weight_i, input_i in zip(weights, inputs))

def sigmoid(activation):
    '''função responsável por calcular a aproximação da função'''
    return 1 / (1 + math.exp(-activation))

def forward_propagate(network, row):
    '''função responsável em propagar os valores da rede para frente'''
    inputs = row
    
    index = 0
    for layer in network:
        new_inputs = list()
        if index != len(network)-1:
            new_inputs.append(1.00000) # adicionando a entrada do vies
        for neuron in layer:
            activation = dot(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
        index = index + 1
    return inputs

def cost(fx, y):
    """ fx: Valores preditos; y: Saída original """
    j = -((y * np.log(fx) + (1 - y) * np.log(1 - fx)))
    return j.sum()

def total_cost(regulation, n_dataset, weights):
    return (regulation/(2*n_dataset)) * np.sum(np.square(w[:, 1:]).sum() for w in weights)

def transfer_derivative(output):
    return output * (1.0 - output)

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (float(neuron['weights'][j]) * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - float(expected[j]))
            print('deltas: ', errors)
        for j in range(len(layer)): # TODO: é gradiente?
            neuron = layer[j]
            neuron['delta'] = errors[j] / transfer_derivative(neuron['output'])

def to_float(values):
    result = list()
    for value in values:
        result.append(float(value))
    return result

def print_network(network):
    for layer in network:
        for neuron in layer:
            print(neuron)
        print('\n')

def predict(network, input):
    return feed_forward(network, input)[-1]

# print(predict([0.42000]))

def main(args):
    print('[INFO] lendo o arquivo de definição da rede')
    def_network = read_def_network(args[1])
    print('[INFO] lendo o arquivo de definição de pesos')
    initial_weights = read_initial_weights(args[2])
    print('[INFO] lendo o arquivo de dataset')
    dataset = read_dataset(args[3])
    
    print('[INFO] criado a rede neural')
    network = create_network(def_network, initial_weights)

    print('\n')

    total_cost = 0 # TODO:
    total_lines_dataset = 0

    for row in dataset:
        result_split = row[0].rstrip().split('; ')
        
        # preparando os dados
        other_attributes = result_split[0].rstrip().split(', ')
        attributes = list()
        attributes.append('1.00000')
        attributes = attributes + other_attributes
        
        expected_values = result_split[1].rstrip().split(', ')

        # realizando o forward_propagate
        output = forward_propagate(network, attributes)

        # realizando o calculo do custo
        last_neuron = network[-1][0]
        print('custo: ', cost(np.array(output), np.array(to_float(expected_values))))

        # realizando o backpropagation
        backward_propagate_error(network, expected_values)


        # preparando o dataset para atualizar os pesos
        row_without_bies = attributes
        row_without_bies.pop(0)
        row_to_update = row_without_bies + expected_values

        # realizando a atualização dos pesos
        update_weights(network, to_float(row_to_update), 0.001)

        #print_network(network)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
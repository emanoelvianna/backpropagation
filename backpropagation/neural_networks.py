# -*- coding: utf-8 -*- 

def create_network(def_network, initial_weights):
    '''função responsável em criar a rede neural a partir dos arquivos de definição'''
    '''def_network definição da estrutura da rede'''
    '''initial_weights pesos inicias da rede'''
    network = list()
    
    # trabalhando sobre o arquivo que define a rede
    regulation = def_network[0]
    def_network.pop(0) # remove a regulação já armazenada
    
    input_length = def_network[0]
    def_network.pop(0)
    
    output_length = def_network[len(def_network)-1]
    def_network.pop(len(def_network)-1)
    
    # imprimindo informações capturadas da definição da rede
    #print('[INFO] Regulação: ', regulation)
    #print('[INFO] Quantidade de entradas: ', input_length)
    #print('[INFO] Quantidade de saídas: ', output_length)
    # imprimindo informações das camadas ocultas
    #print('[INFO] Numero de neurônios em cada camada oculta: ', def_network, '\n')
    
    # trabalhando sobre o arquivo de quefine os pesos
    for weights in initial_weights:
        layers = weights.rstrip().split('; ')
        network_layer = list()
        for neuron in layers:
            if ', ' in neuron:
                network_layer.append({'weights': [float(value) for value in neuron.rstrip().split(', ')]})
            else:
                network_layer.append({'weights': float(neuron)})
        network.append(network_layer)
    #print('[INFO] Rede neural criada: ', network, '\n')
    return network
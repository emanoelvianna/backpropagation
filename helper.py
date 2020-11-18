# -*- coding: utf-8 -*- 

def read_initial_weights(path_file):
    '''função responsavel em ler o arquivo de pesos inicias'''
    return open(path_file,'r+').readlines()

def read_dataset(path_file):
    '''função responsável em ler o arquivo de dataset'''
    dataset = list()
    file = open(path_file, 'r+')
    for lines in file:
        values_line = list()
        dataset.append([lines.rstrip()])
    return dataset
    
def read_def_network(path_file):
    '''função responsável em ler o arquivo de definição da rede neural'''
    file = open(path_file, "r+")
    
    network = list()
    for line in file:
        network.append(line.rstrip())
    return network
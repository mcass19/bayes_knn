from data_set import DataSet
from generate_attributes import Generator

print('*****************************************')
print('CLASIFICADOR BAYESIANO SENCILLO Y K-NN')
print('*****************************************')

option_algorithm = input('Ingrese 1 si desea ejecutar el clasificador bayesiano simple \no 2 si desea ejecutar el k-nn: ')
k = 0
if (int(option_algorithm) == 2):
    option_k = input('Ingrese valor de k (1, 3 o 7): ')
    k = int(option_k)

option_data_set = input('Ingrese 1 para generar un árbol a partir del data set Iris \no 2 para generar un árbol a partir del data set Covtype: ')
option_cant = input('Ingrese la cantidad de puntos de corte para los atributos continuos: ')

print('\n')

data_set = DataSet()
data_set_test = data_set.load_data_set(int(option_data_set))

print('Generó dataset')

generator = Generator()
attributes = generator.generate_attributes(data_set, int(option_cant), data_set.continue_attributes)

print('Generó atributos')

# if 1 bayes else k

# evaluacion

from data_set import DataSet
from generate_attributes import Generator
from bayes import Bayes
from k_nn import Knn

print('*****************************************')
print('CLASIFICADOR BAYESIANO SENCILLO Y K-NN')
print('*****************************************')

option_algorithm = int(input('Ingrese 1 si desea ejecutar el clasificador bayesiano simple \no 2 si desea ejecutar el k-nn: '))

k = 0
if (option_algorithm == 2):
    option_k = input('Ingrese valor de k (1, 3 o 7): ')
    k = int(option_k)

option_data_set = input('Ingrese 1 para ejecutar el clasificador con el data set Iris \no 2 para ejecutar el clasificador con el data set Covertype: ')
option_cant = input('Ingrese la cantidad de puntos de corte para los atributos continuos: ')

print('\n')

data_set = DataSet()
data_set_test = data_set.load_data_set(int(option_data_set))

generator = Generator()
attributes = generator.generate_attributes(data_set, int(option_cant), data_set.continue_attributes)

# Cambio de valores continuos a sus correspondientes intervalos dentro de los posibles valores del atributo
data_set.replace_continue_attributes(attributes)
data_set_test.replace_continue_attributes(attributes)

if option_algorithm == 1:
    bayes = Bayes(data_set)

    results = []
    well_classified = 0
    for instance in data_set_test.data:
        classified = bayes.simple_classifier(data_set, instance, attributes)
        results.append((classified, instance[-1]))
        if (classified[0] == instance[-1]):
            well_classified += 1
    
    print('Clasificador Bayesiano:')
    
    # # Impresión de probabilidad por instancia
    # for result in results:
    #     print('Con una probabilidad de {}, se puede afirmar que esta instancia \nse clasifica -> {}, siendo la etiqueta de la misma -> {}'.format(result[0][1], result[0][0], result[1]))
    #     print('\n')

    print('\t -> De {} instancias, {} clasificaron correctamente.'.format(len(data_set_test.data), well_classified))
else:
    # recorta la lista de los atributos 
    attributes_aux = attributes.copy()
    best_attributes = data_set.generate_best_attributes(attributes, attributes_aux)
    
    k_nn = Knn(k)

    results = []
    well_classified = 0
    for instance in data_set_test.data:
        classified = k_nn.classify(data_set, instance, attributes)
        results.append((classified, instance[-1]))
        if (classified == instance[-1]):
            well_classified += 1

    print('Clasificador K-NN con k={}:'.format(k))
    
    # # Impresión por instancia
    # for result in results:
    #     print('Esta instancia se clasifica -> {}, siendo la etiqueta de la misma -> {}'.format(result[0], result[1]))
    #     print('\n')

    print('\t -> De {} instancias, {} clasificaron correctamente.'.format(len(data_set_test.data), well_classified))

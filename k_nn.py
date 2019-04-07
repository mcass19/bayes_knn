import math
from operator import itemgetter

class Knn(object):

	def __init__(self, k):
		self.k = k

	# clasifica una nueva instancia según el clasificador K-NN con el k ingresado al momento
	# de construir la clase. Los atributos con los que se calcula la distancia son best_attributes.
	# Retorna la etiqueta con la que clasficó
	def classify(self, data_set, instance, best_attributes):
		k_nn = [''] * self.k
		
		distances = []
		for data_instance in data_set.data:
			distances.append((self.distance(instance, data_instance, best_attributes), data_instance[-1]))

		itt_k = 0
		while itt_k < self.k:
			distance_min = min(distances, key=itemgetter(0))
			neighbors_min = []

			distances_aux = distances.copy()
			for d in distances_aux:
				if (d[0] == distance_min[0]):
					neighbors_min.append(d[1])
					distances.remove(d)

			target_max = self.most_common_target_in_neighbors(data_set, neighbors_min)		

			if ((itt_k == 0) and (len(neighbors_min) > self.k)):
				return target_max
			else:
				it = 0
				while ((itt_k < self.k) and (it < len(neighbors_min))):
					k_nn[itt_k] = target_max
					itt_k += 1
					it += 1
		
		target = self.most_common_target_in_neighbors(data_set, k_nn)	
		
		return target

	# calcula la distancia entre la instancia que se está clasificando (instance) y una 
	# instancia del data set (data_instance)
	def distance(self, instance, data_instance, best_attributes):
		distance = 0
		for att in range(len(best_attributes)):
			distance += (instance[att] - data_instance[att]) ** 2
				
		return math.sqrt(distance)

	# devuelve la etiqueta con mayor cantidad de ocurrencias en neighbors
	def most_common_target_in_neighbors(self, data_set, neighbors):
		target_values = data_set.target_values()
		max_tar_val = 0
		target_max = ''
		
		for t_val in target_values:
			cant_t_val = neighbors.count(t_val)
			if (cant_t_val > max_tar_val):
				max_tar_val = cant_t_val
				target_max = t_val

		return target_max

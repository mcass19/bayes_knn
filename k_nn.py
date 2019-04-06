import math
from operator import itemgetter

class Knn(object):

	def __init__(self, k):
		self.k = k

	def classify(self, data_set, instance, best_attributes):
		k_nn = [''] * self.k
		
		distances = []
		for data_instance in data_set.data:
			distances.append((self.distance(instance, data_instance, best_attributes), data_instance[-1]))

		for itt_k in range(self.k):
			distance_min = min(distances, key=itemgetter(0))
			aux = []

			distances_aux = distances.copy()
			for d in distances_aux:
				if (d[0] == distance_min[0]):
					aux.append(d[1])
					distances.remove(d)

			target_values = data_set.target_values()
			max_tar_val = 0
			target_max = ''
			for t_val in target_values:
				cant_t_val = aux.count(t_val)
				if (cant_t_val > max_tar_val):
					max_tar_val = cant_t_val
					target_max = t_val

			if (len(aux) > self.k):
				return target_max
			else:
				# CAMBIAR EL MANEJO DE K ACA, METER LOS QUE HAYA Y SEGUIR BUSCANDO
				k_nn[itt_k] = target_max
		
		target_values = data_set.target_values()
		max_tar_val = 0
		target_max = ''
		for t_val in target_values:
			cant_t_val = k_nn.count(t_val)
			if (cant_t_val > max_tar_val):
				max_tar_val = cant_t_val
				target_max = t_val
		
		return target_max

	def distance(self, instance, data_instance, best_attributes):
		distance = 0

		for att in range(len(best_attributes)):
			distance += (instance[att] - data_instance[att]) ** 2
		
		# print('distancia ' + str(distance) + ' etiqueta ' + str(data_instance[-1]))
				
		return math.sqrt(distance)
from data_set import DataSet

class Bayes(object):

    def __init__(self, data_set):
        self.m = len(data_set.data)
        
        # probabilidad de cada etiqueta
        self.p_targets = []
        for target in data_set.target_values():
            data_set_target = data_set.data_set_class(target)
            probability = len(data_set_target.data) / len(data_set.data)
            # lista de tuplas (etiqueta, probabilidad, dataset_etiqueta)
            self.p_targets.append((target, probability, data_set_target))

    def simple_classifier(self, data_set, instance, attributes):
        norma = 0
        attmax = 0

        for target in self.p_targets:
            p_attrs_target = target[1]
            data_set_target = target[2]
            for att in range(len(instance) - 1):
                value = instance[att]
                
                len_data_value = len(data_set_target.subset_of_value(att, value).data)
                len_data_target = len(data_set_target.data) 
                p_value_target = len_data_value / len_data_target
                
                # m-estimador
                if (p_value_target == 0):
                    p_value_target = (len_data_value + (self.m * (1 / len(attributes[att])))) / (len_data_target + self.m)

                p_attrs_target *= p_value_target

                # print('att ' + str(att) + ' value ' + str(value) + ' probabilidad ' + str(p_attrs_target))
            
            norma += p_attrs_target
            if p_attrs_target > attmax:
                attmax = p_attrs_target
                target_max = target[0]

        probability = (attmax / norma) * 100

        return target_max, probability

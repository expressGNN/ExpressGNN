from copy import deepcopy

# dictionary of all constant types with the form of {'Person':0, 'Course':1, ...}
TYPE_SET = set()


class ConstantDict:
    
    def __init__(self):
        self.constants = {}
        
    def add_const(self, const_type, const):
        """
        
        :param const_type:
            string 
        :param const:
            string          
        """
        
        # if const_type not in TYPE_DICT:
        #     TYPE_DICT[const_type] = len(TYPE_DICT)
        
        if const_type in self.constants:
            self.constants[const_type].add(const)
        else:
            self.constants[const_type] = {const}

    def __getitem__(self, key):
        return self.constants[key]

    def has_const(self, key, const):
        if key in self.constants:
            return const in self[key]
        else:
            return False


class Fact:
    def __init__(self, pred_name, const_ls, val):
        self.pred_name = pred_name
        self.const_ls = deepcopy(const_ls)
        self.val = val

    def __repr__(self):
        return self.pred_name + '(%s)' % ','.join(self.const_ls)


const_dict = ConstantDict()

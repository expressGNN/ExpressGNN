class Atom:
    def __init__(self, neg, pred_name, var_name_ls, var_type_ls):
        self.neg = neg
        self.pred_name = pred_name
        self.var_name_ls = var_name_ls
        self.var_type_ls = var_type_ls

    def __repr__(self):
        return ('!' if self.neg else '') + self.pred_name + '(%s)' % ','.join(self.var_name_ls)


class Formula:
    """
        only support clause form with disjunction, e.g. !
    """

    def __init__(self, atom_ls, weight):
        self.weight = weight
        self.atom_ls = atom_ls
        self.rule_vars = dict()

        for atom in self.atom_ls:
            self.rule_vars.update(zip(atom.var_name_ls, atom.var_type_ls))
        self.key2ind = dict(zip(self.rule_vars.keys(), range(len(self.rule_vars.keys()))))

    def evaluate(self):
        pass

    def __repr__(self):
        return ' v '.join(list(map(repr, self.atom_ls)))

import re
from common.predicate import Predicate, PRED_DICT
from common.constants import TYPE_SET, const_dict, Fact
from common.formula import Atom, Formula
from os.path import join as joinpath
from os.path import isfile
from common.utils import iterline
from common.cmd_args import cmd_args


def preprocess_large(dataroot):
    """
        Preprocessing for FB and WN. Assuming:

            * all relations are of artiy of 2
            * only one constant type
            * all facts are positive facts

        :param dataroot:
            data root path
        :return:

    """

    fact_path_ls = [joinpath(dataroot, 'facts.txt'),
                    joinpath(dataroot, 'train.txt')]
    query_path = joinpath(dataroot, 'test.txt')
    pred_path = joinpath(dataroot, 'relations.txt')
    const_path = joinpath(dataroot, 'entities.txt')
    valid_path = joinpath(dataroot, 'valid.txt')

    rule_path = joinpath(dataroot, cmd_args.rule_filename)

    assert all(map(isfile, fact_path_ls+[query_path, pred_path, const_path, valid_path, rule_path]))

    # assuming only one type
    TYPE_SET.update(['type'])

    # add all const
    for line in iterline(const_path):
        const_dict.add_const('type', line)

    # add all pred
    for line in iterline(pred_path):
        PRED_DICT[line] = Predicate(line, ['type', 'type'])

    # add all facts
    fact_ls = []
    for fact_path in fact_path_ls:
        for line in iterline(fact_path):
            parts = line.split('\t')

            assert len(parts) == 3, print(parts)

            e1, pred_name, e2 = parts

            assert const_dict.has_const('type', e1) and const_dict.has_const('type', e2)
            assert pred_name in PRED_DICT

            fact_ls.append(Fact(pred_name, [e1, e2], 1))

    # add all validations
    valid_ls = []
    for line in iterline(valid_path):
        parts = line.split('\t')

        assert len(parts) == 3, print(parts)

        e1, pred_name, e2 = parts

        assert const_dict.has_const('type', e1) and const_dict.has_const('type', e2)
        assert pred_name in PRED_DICT

        valid_ls.append(Fact(pred_name, [e1, e2], 1))

    # add all queries
    query_ls = []
    for line in iterline(query_path):
        parts = line.split('\t')

        assert len(parts) == 3, print(parts)

        e1, pred_name, e2 = parts

        assert const_dict.has_const('type', e1) and const_dict.has_const('type', e2)
        assert pred_name in PRED_DICT

        query_ls.append(Fact(pred_name, [e1, e2], 1))

    # add all rules
    rule_ls = []
    strip_items = lambda ls: list(map(lambda x: x.strip(), ls))
    first_atom_reg = re.compile(r'([\d.]+) (!?)([^(]+)\((.*)\)')
    atom_reg = re.compile(r'(!?)([^(]+)\((.*)\)')
    for line in iterline(rule_path):

        atom_str_ls = strip_items(line.split(' v '))
        assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

        atom_ls = []
        rule_weight = 0.0
        for i, atom_str in enumerate(atom_str_ls):
            if i == 0:
                m = first_atom_reg.match(atom_str)
                assert m is not None, 'matching atom failed for %s' % atom_str
                rule_weight = float(m.group(1))
                neg = m.group(2) == '!'
                pred_name = m.group(3).strip()
                var_name_ls = strip_items(m.group(4).split(','))
            else:
                m = atom_reg.match(atom_str)
                assert m is not None, 'matching atom failed for %s' % atom_str
                neg = m.group(1) == '!'
                pred_name = m.group(2).strip()
                var_name_ls = strip_items(m.group(3).split(','))

            atom = Atom(neg, pred_name, var_name_ls, PRED_DICT[pred_name].var_types)
            atom_ls.append(atom)

        rule = Formula(atom_ls, rule_weight)
        rule_ls.append(rule)

    return fact_ls, rule_ls, valid_ls, query_ls


def preprocess_kinship(ppath, fpath, rpath, qpath):
    """

    :param ppath:
        predicate file path
    :param fpath:
        facts file path
    :param rpath:
        rule file path
    :param qpath:
        query file path

    :return:

    """

    assert all(map(isfile, [ppath, fpath, rpath, qpath]))

    strip_items = lambda ls: list(map(lambda x: x.strip(), ls))

    pred_reg = re.compile(r'(.*)\((.*)\)')

    with open(ppath) as f:
        for line in f:

            # skip empty lines
            if line.strip() == '':
                continue

            m = pred_reg.match(line.strip())
            assert m is not None, 'matching predicate failed for %s' % line

            name, var_types = m.group(1), m.group(2)
            var_types = list(map(lambda x: x.strip(), var_types.split(',')))

            PRED_DICT[name] = Predicate(name, var_types)
            TYPE_SET.update(var_types)

    fact_ls = []
    fact_reg = re.compile(r'(!?)(.*)\((.*)\)')
    with open(fpath) as f:
        for line in f:

            # skip empty lines
            if line.strip() == '':
                continue

            m = fact_reg.match(line.strip())
            assert m is not None, 'matching fact failed for %s' % line

            val = 0 if m.group(1) == '!' else 1
            name, consts = m.group(2), m.group(3)
            consts = strip_items(consts.split(','))

            fact_ls.append(Fact(name, consts, val))

            for var_type in PRED_DICT[name].var_types:
                const_dict.add_const(var_type, consts.pop(0))

    rule_ls = []
    first_atom_reg = re.compile(r'([\d.]+) (!?)([\w\d]+)\((.*)\)')
    atom_reg = re.compile(r'(!?)([\w\d]+)\((.*)\)')
    with open(rpath) as f:
        for line in f:

            # skip empty lines
            if line.strip() == '':
                continue

            atom_str_ls = strip_items(line.strip().split(' v '))
            assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

            atom_ls = []
            rule_weight = 0.0
            for i, atom_str in enumerate(atom_str_ls):
                if i == 0:
                    m = first_atom_reg.match(atom_str)
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    rule_weight = float(m.group(1))
                    neg = m.group(2) == '!'
                    pred_name = m.group(3).strip()
                    var_name_ls = strip_items(m.group(4).split(','))
                else:
                    m = atom_reg.match(atom_str)
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    neg = m.group(1) == '!'
                    pred_name = m.group(2).strip()
                    var_name_ls = strip_items(m.group(3).split(','))

                atom = Atom(neg, pred_name, var_name_ls, PRED_DICT[pred_name].var_types)
                atom_ls.append(atom)

            rule = Formula(atom_ls, rule_weight)
            rule_ls.append(rule)

    query_ls = []
    with open(qpath) as f:
        for line in f:

            # skip empty lines
            if line.strip() == '':
                continue

            m = fact_reg.match(line.strip())
            assert m is not None, 'matching fact failed for %s' % line

            val = 0 if m.group(1) == '!' else 1
            name, consts = m.group(2), m.group(3)
            consts = strip_items(consts.split(','))

            query_ls.append(Fact(name, consts, val))

            for var_type in PRED_DICT[name].var_types:
                const_dict.add_const(var_type, consts.pop(0))

    return fact_ls, rule_ls, query_ls

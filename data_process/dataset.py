from data_process.preprocess import preprocess_kinship, preprocess_large
from os.path import join as joinpath
from common.constants import const_dict
from common.predicate import PRED_DICT
import itertools
import random
from random import shuffle, choice
from collections import Counter
import numpy as np

# grounded rule stats code
BAD = 0  # sample not valid
FULL_OBSERVERED = 1  # sample valid, but rule contains only observed vars and does not have negation for all atoms
GOOD = 2  # sample valid


class Dataset:
    def __init__(self, data_root, batchsize, shuffle_sampling=False, ext_rule_path=None, load_method=0):

        guss_fb = 'fb15k' in data_root
        if guss_fb != (load_method == 1):
            print("WARNING: set load_method to 1 if you load Freebase dataset, otherwise 0")


        if load_method == 1:
            fact_ls, rule_ls, valid_ls, query_ls = preprocess_large(data_root)
        else:
            rpath = joinpath(data_root, 'rules') if ext_rule_path is None else ext_rule_path
            fact_ls, rule_ls, query_ls = preprocess_kinship(joinpath(data_root, 'predicates'),
                                                            joinpath(data_root, 'facts'),
                                                            rpath,
                                                            joinpath(data_root, 'queries'))
            valid_ls = []

        self.const_sort_dict = dict(
            [(type_name, sorted(list(const_dict[type_name]))) for type_name in const_dict.constants.keys()])

        if load_method == 1:
            self.const2ind = dict([(const, i) for i, const in enumerate(self.const_sort_dict['type'])])

        # linear in size of facts
        self.fact_dict = dict((pred_name, set()) for pred_name in PRED_DICT)
        self.test_fact_dict = dict((pred_name, set()) for pred_name in PRED_DICT)
        self.valid_dict = dict((pred_name, set()) for pred_name in PRED_DICT)

        self.ht_dict = dict((pred_name, [dict(), dict()]) for pred_name in PRED_DICT)
        self.ht_dict_train = dict((pred_name, [dict(), dict()]) for pred_name in PRED_DICT)

        def add_ht(pn, c_ls, ht_dict):
            if load_method == 0:
                if c_ls[0] in ht_dict[pn][0]:
                    ht_dict[pn][0][c_ls[0]].add(c_ls[0])
                else:
                    ht_dict[pn][0][c_ls[0]] = set([c_ls[0]])
            elif load_method == 1:
                if c_ls[0] in ht_dict[pn][0]:
                    ht_dict[pn][0][c_ls[0]].add(c_ls[1])
                else:
                    ht_dict[pn][0][c_ls[0]] = set([c_ls[1]])

                if c_ls[1] in ht_dict[pn][1]:
                    ht_dict[pn][1][c_ls[1]].add(c_ls[0])
                else:
                    ht_dict[pn][1][c_ls[1]] = set([c_ls[0]])

        const_cnter = Counter()
        for fact in fact_ls:
            self.fact_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict_train)
            const_cnter.update(fact.const_ls)

        for fact in valid_ls:
            self.valid_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)

        # the sorted list version
        self.fact_dict_2 = dict((pred_name, sorted(list(self.fact_dict[pred_name])))
                                for pred_name in self.fact_dict.keys())
        self.valid_dict_2 = dict((pred_name, sorted(list(self.valid_dict[pred_name])))
                                 for pred_name in self.valid_dict.keys())

        self.rule_ls = rule_ls

        # pred_atom-key dict
        self.atom_key_dict_ls = []
        for rule in self.rule_ls:
            atom_key_dict = dict()

            for atom in rule.atom_ls:
                atom_dict = dict((var_name, dict()) for var_name in atom.var_name_ls)

                for i, var_name in enumerate(atom.var_name_ls):

                    if atom.pred_name not in self.fact_dict:
                        continue

                    for v in self.fact_dict[atom.pred_name]:
                        if v[1][i] not in atom_dict[var_name]:
                            atom_dict[var_name][v[1][i]] = [v]
                        else:
                            atom_dict[var_name][v[1][i]] += [v]

                # happens if predicate occurs more than once in one rule then we merge the set
                if atom.pred_name in atom_key_dict:
                    for k, v in atom_dict.items():
                        if k not in atom_key_dict[atom.pred_name]:
                            atom_key_dict[atom.pred_name][k] = v
                else:
                    atom_key_dict[atom.pred_name] = atom_dict

            self.atom_key_dict_ls.append(atom_key_dict)

        self.test_fact_ls = []
        self.valid_fact_ls = []

        for fact in query_ls:
            self.test_fact_ls.append((fact.val, fact.pred_name, tuple(fact.const_ls)))
            self.test_fact_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)

        for fact in valid_ls:
            self.valid_fact_ls.append((fact.val, fact.pred_name, tuple(fact.const_ls)))

        self.shuffle_sampling = shuffle_sampling
        self.batchsize = batchsize
        self.num_rules = len(rule_ls)

        self.rule_gens = None
        self.reset()

    def generate_gnd_pred(self, pred_name):
        """
            return a list of all instantiations of a predicate function, this can be extremely large
        :param pred_name:
            string
        :return:
        """

        assert pred_name in PRED_DICT

        pred = PRED_DICT[pred_name]
        subs = itertools.product(*[self.const_sort_dict[var_type] for var_type in pred.var_types])

        return [(pred_name, sub) for sub in subs]

    def generate_gnd_rule(self, rule):

        subs = itertools.product(*[self.const_sort_dict[rule.rule_vars[k]] for k in rule.rule_vars.keys()])
        sub = next(subs, None)

        while sub is not None:

            latent_vars = []
            latent_neg_mask = []
            observed_neg_mask = []

            for atom in rule.atom_ls:
                grounding = tuple(sub[rule.key2ind[var_name]] for var_name in atom.var_name_ls)
                pos_gnding, neg_gnding = (1, grounding), (0, grounding)

                if pos_gnding in self.fact_dict[atom.pred_name]:
                    observed_neg_mask.append(0 if atom.neg else 1)
                elif neg_gnding in self.fact_dict[atom.pred_name]:
                    observed_neg_mask.append(1 if atom.neg else 0)
                else:
                    latent_vars.append((atom.pred_name, grounding))
                    latent_neg_mask.append(1 if atom.neg else 0)

            isfullneg = (sum(latent_neg_mask) == len(latent_neg_mask)) and \
                        (sum(observed_neg_mask) > 0)

            yield latent_vars, [latent_neg_mask, observed_neg_mask], isfullneg

            sub = next(subs, None)

    def get_batch(self, epoch_mode=False, filter_latent=True):
        """
            return the ind-th batch of ground formula and latent variable indicators
        :param ind:
            index of the batch
        :return:
        """

        batch_neg_mask = [[] for _ in range(len(self.rule_ls))]
        batch_latent_var_inds = [[] for _ in range(len(self.rule_ls))]
        observed_rule_cnts = [0.0 for _ in range(len(self.rule_ls))]
        flat_latent_vars = dict()

        cnt = 0

        inds = list(range(len(self.rule_ls)))

        while cnt < self.batchsize:

            if self.shuffle_sampling:
                shuffle(inds)

            hasdata = False
            for ind in inds:
                latent_vars, neg_mask, isfullneg = next(self.rule_gens[ind], (None, None, None))

                if latent_vars is None:
                    if epoch_mode:
                        continue
                    else:
                        self.rule_gens[ind] = self.generate_gnd_rule(self.rule_ls[ind])
                        latent_vars, neg_mask, isfullneg = next(self.rule_gens[ind])

                if epoch_mode:
                    hasdata = True

                # if rule is fully latent
                if (len(neg_mask[1]) == 0) and filter_latent:
                    continue

                # if rule fully observed
                if len(latent_vars) == 0:
                    observed_rule_cnts[ind] += 0 if isfullneg else 1
                    cnt += 1
                    if cnt >= self.batchsize:
                        break
                    else:
                        continue

                batch_neg_mask[ind].append(neg_mask)

                for latent_var in latent_vars:
                    if latent_var not in flat_latent_vars:
                        flat_latent_vars[latent_var] = len(flat_latent_vars)

                batch_latent_var_inds[ind].append([flat_latent_vars[e] for e in latent_vars])

                cnt += 1

                if cnt >= self.batchsize:
                    break

            if epoch_mode and (hasdata is False):
                break

        flat_list = sorted([(k, v) for k, v in flat_latent_vars.items()], key=lambda x: x[1])
        flat_list = [e[0] for e in flat_list]

        return batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts

    def _instantiate_pred(self, atom, atom_dict, sub, rule, observed_prob):

        key2ind = rule.key2ind
        rule_vars = rule.rule_vars

        # substitute with observed fact
        if np.random.rand() < observed_prob:

            fact_choice_set = None
            for var_name in atom.var_name_ls:
                const = sub[key2ind[var_name]]
                if const is None:
                    choice_set = itertools.chain.from_iterable([v for k, v in atom_dict[var_name].items()])
                else:
                    if const in atom_dict[var_name]:
                        choice_set = atom_dict[var_name][const]
                    else:
                        choice_set = []

                if fact_choice_set is None:
                    fact_choice_set = set(choice_set)
                else:
                    fact_choice_set = fact_choice_set.intersection(set(choice_set))

                if len(fact_choice_set) == 0:
                    break

            if len(fact_choice_set) == 0:
                for var_name in atom.var_name_ls:
                    if sub[key2ind[var_name]] is None:
                        sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])
            else:
                val, const_ls = choice(sorted(list(fact_choice_set)))
                for var_name, const in zip(atom.var_name_ls, const_ls):
                    sub[key2ind[var_name]] = const

        # substitute with random facts
        else:
            for var_name in atom.var_name_ls:
                if sub[key2ind[var_name]] is None:
                    sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])

    def _gen_mask(self, rule, sub, closed_world):

        latent_vars = []
        observed_vars = []
        latent_neg_mask = []
        observed_neg_mask = []

        for atom in rule.atom_ls:
            grounding = tuple(sub[rule.key2ind[var_name]] for var_name in atom.var_name_ls)
            pos_gnding, neg_gnding = (1, grounding), (0, grounding)

            if pos_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((1, atom.pred_name))
                observed_neg_mask.append(0 if atom.neg else 1)
            elif neg_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((0, atom.pred_name))
                observed_neg_mask.append(1 if atom.neg else 0)
            else:
                if closed_world and (len(self.test_fact_dict[atom.pred_name]) == 0):
                    observed_vars.append((0, atom.pred_name))
                    observed_neg_mask.append(1 if atom.neg else 0)
                else:
                    latent_vars.append((atom.pred_name, grounding))
                    latent_neg_mask.append(1 if atom.neg else 0)

        return latent_vars, observed_vars, latent_neg_mask, observed_neg_mask

    def _get_rule_stat(self, observed_vars, latent_vars, observed_neg_mask, filter_latent, filter_observed):

        is_full_latent = len(observed_vars) == 0
        is_full_observed = len(latent_vars) == 0

        if is_full_latent and filter_latent:
            return BAD

        if is_full_observed:

            if filter_observed:
                return BAD

            is_full_neg = sum(observed_neg_mask) == 0

            if is_full_neg:
                return BAD

            else:
                return FULL_OBSERVERED

        # if observed var already yields 1
        if sum(observed_neg_mask) > 0:
            return BAD

        return GOOD

    # TODO only binary | only positive fact!!
    def _inst_var(self, sub, var2ind, var2type, at, ht_dict, gen_latent):

        if len(at.var_name_ls) != 2:
            raise KeyError

        must_latent = gen_latent

        if must_latent:

            tmp = [sub[var2ind[vn]] for vn in at.var_name_ls]

            for i, subi in enumerate(tmp):
                if subi is None:
                    tmp[i] = random.choice(self.const_sort_dict[var2type[at.var_name_ls[i]]])

            islatent = (tmp[0] not in ht_dict[0]) or (tmp[1] not in ht_dict[0][tmp[0]])
            for i, vn in enumerate(at.var_name_ls):
                sub[var2ind[vn]] = tmp[i]
            return [self.const2ind[subi] for subi in tmp], islatent, islatent or at.neg

        vn0 = at.var_name_ls[0]
        sub0 = sub[var2ind[vn0]]
        vn1 = at.var_name_ls[1]
        sub1 = sub[var2ind[vn1]]

        if sub0 is None:

            if sub1 is None:
                if len(ht_dict[0]) > 0:
                    sub0 = random.choice(tuple(ht_dict[0].keys()))
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn0]] = sub0
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg

            else:
                if sub1 in ht_dict[1]:
                    sub0 = random.choice(tuple(ht_dict[1][sub1]))
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg
                else:
                    sub0 = random.choice(self.const_sort_dict[var2type[vn0]])
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True

        else:

            if sub1 is None:
                if sub0 in ht_dict[0]:
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg
                else:
                    sub1 = random.choice(self.const_sort_dict[var2type[vn1]])
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True

            else:
                islatent = (sub0 not in ht_dict[0]) or (sub1 not in ht_dict[0][sub0])
                return [self.const2ind[sub0], self.const2ind[sub1]], islatent, islatent or at.neg

    # TODO use it only for binary rel and positive fact only
    def get_batch_fast(self, batchsize, observed_prob=0.9):

        prob_decay = 0.5

        for rule in self.rule_ls:

            var2ind = rule.key2ind
            var2type = rule.rule_vars
            samples = [[atom.pred_name, []] for atom in rule.atom_ls]
            neg_mask = [[atom.pred_name, []] for atom in rule.atom_ls]
            latent_mask = [[atom.pred_name, []] for atom in rule.atom_ls]
            obs_var = [[atom.pred_name, []] for atom in rule.atom_ls]

            cnt = 0
            while cnt <= batchsize:

                sub = [None] * len(rule.rule_vars)  # substitutions

                sample_buff = [[] for _ in rule.atom_ls]
                neg_mask_buff = [[] for _ in rule.atom_ls]
                latent_mask_buff = [[] for _ in rule.atom_ls]

                atom_inds = list(range(len(rule.atom_ls)))
                shuffle(atom_inds)
                succ = True
                cur_threshold = observed_prob
                obs_list = []

                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    pred_ht_dict = self.ht_dict_train[atom.pred_name]

                    gen_latent = np.random.rand() > cur_threshold
                    c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                               atom, pred_ht_dict, gen_latent)

                    if not islatent:
                        obs_var[atom_ind][1].append(c_ls)

                    cur_threshold *= prob_decay
                    succ = succ and atom_succ
                    obs_list.append(not islatent)

                    if succ:
                        sample_buff[atom_ind].append(c_ls)
                        latent_mask_buff[atom_ind].append(1 if islatent else 0)
                        neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                if succ and any(obs_list):
                    for i in range(len(rule.atom_ls)):
                        samples[i][1].extend(sample_buff[i])
                        latent_mask[i][1].extend(latent_mask_buff[i])
                        neg_mask[i][1].extend(neg_mask_buff[i])

                cnt += 1

            yield samples, neg_mask, latent_mask, obs_var

    def get_batch_by_q(self, batchsize, observed_prob=1.0, validation=False):

        samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        cnt = 0

        num_ents = len(self.const2ind)
        ind2const = self.const_sort_dict['type']

        def gen_fake(c1, c2, pn):
            for _ in range(10):
                c1_fake = random.randint(0, num_ents - 1)
                c2_fake = random.randint(0, num_ents - 1)
                if np.random.rand() > 0.5:
                    if ind2const[c1_fake] not in self.ht_dict_train[pn][1][ind2const[c2]]:
                        return c1_fake, c2
                else:
                    if ind2const[c2_fake] not in self.ht_dict_train[pn][0][ind2const[c1]]:
                        return c1, c2_fake
            return None, None

        if validation:
            fact_ls = self.valid_fact_ls
        else:
            fact_ls = self.test_fact_ls

        for val, pred_name, consts in fact_ls:

            for rule_i, rule in enumerate(self.rule_ls):

                # find rule with pred_name as head
                if rule.atom_ls[-1].pred_name != pred_name:
                    continue

                samples = samples_by_r[rule_i]
                neg_mask = neg_mask_by_r[rule_i]
                latent_mask = latent_mask_by_r[rule_i]
                obs_var = obs_var_by_r[rule_i]
                neg_var = neg_var_by_r[rule_i]

                var2ind = rule.key2ind
                var2type = rule.rule_vars

                sub = [None] * len(rule.rule_vars)  # substitutions
                vn0, vn1 = rule.atom_ls[-1].var_name_ls
                sub[var2ind[vn0]] = consts[0]
                sub[var2ind[vn1]] = consts[1]

                sample_buff = [[] for _ in rule.atom_ls]
                neg_mask_buff = [[] for _ in rule.atom_ls]
                latent_mask_buff = [[] for _ in rule.atom_ls]

                atom_inds = list(range(len(rule.atom_ls) - 1))
                shuffle(atom_inds)
                succ = True
                obs_list = []

                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    pred_ht_dict = self.ht_dict_train[atom.pred_name]

                    gen_latent = np.random.rand() > observed_prob
                    c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                               atom, pred_ht_dict, gen_latent)

                    assert atom_succ

                    if not islatent:
                        obs_var[atom_ind][1].append(c_ls)
                        c1, c2 = gen_fake(c_ls[0], c_ls[1], atom.pred_name)
                        if c1 is not None:
                            neg_var[atom_ind][1].append([c1, c2])

                    succ = succ and atom_succ
                    obs_list.append(not islatent)

                    sample_buff[atom_ind].append(c_ls)
                    latent_mask_buff[atom_ind].append(1 if islatent else 0)
                    neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                if succ and any(obs_list):
                    for i in range(len(rule.atom_ls)):
                        samples[i][1].extend(sample_buff[i])
                        latent_mask[i][1].extend(latent_mask_buff[i])
                        neg_mask[i][1].extend(neg_mask_buff[i])

                    samples[-1][1].append([self.const2ind[consts[0]], self.const2ind[consts[1]]])
                    latent_mask[-1][1].append(1)
                    neg_mask[-1][1].append(1)

                    cnt += 1

            if cnt >= batchsize:
                yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

                samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                cnt = 0

        yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r


    def get_batch_by_q_v2(self, batchsize, observed_prob=1.0):

        samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        cnt = 0

        num_ents = len(self.const2ind)
        ind2const = self.const_sort_dict['type']

        def gen_fake(c1, c2, pn):
            for _ in range(10):
                c1_fake = random.randint(0, num_ents - 1)
                c2_fake = random.randint(0, num_ents - 1)
                if np.random.rand() > 0.5:
                    if ind2const[c1_fake] not in self.ht_dict_train[pn][1][ind2const[c2]]:
                        return c1_fake, c2
                else:
                    if ind2const[c2_fake] not in self.ht_dict_train[pn][0][ind2const[c1]]:
                        return c1, c2_fake
            return None, None

        for val, pred_name, consts in self.test_fact_ls:

            for rule_i, rule in enumerate(self.rule_ls):

                # find rule with pred_name as head
                if rule.atom_ls[-1].pred_name != pred_name:
                    continue

                samples = samples_by_r[rule_i]
                neg_mask = neg_mask_by_r[rule_i]
                latent_mask = latent_mask_by_r[rule_i]

                var2ind = rule.key2ind
                var2type = rule.rule_vars

                sub_ls = [[None for _ in range(len(rule.rule_vars))] for _ in range(2)]  # substitutions

                vn0, vn1 = rule.atom_ls[-1].var_name_ls
                sub_ls[0][var2ind[vn0]] = consts[0]
                sub_ls[0][var2ind[vn1]] = consts[1]

                c1, c2 = gen_fake(self.const2ind[consts[0]], self.const2ind[consts[1]], pred_name)
                if c1 is not None:
                    sub_ls[1][var2ind[vn0]] = ind2const[c1]
                    sub_ls[1][var2ind[vn1]] = ind2const[c2]
                else:
                    sub_ls.pop(1)

                pos_query_succ = False

                for sub_ind, sub in enumerate(sub_ls):

                    sample_buff = [[] for _ in rule.atom_ls]
                    neg_mask_buff = [[] for _ in rule.atom_ls]
                    latent_mask_buff = [[] for _ in rule.atom_ls]

                    atom_inds = list(range(len(rule.atom_ls)-1))
                    shuffle(atom_inds)
                    succ = True
                    obs_list = []

                    for atom_ind in atom_inds:
                        atom = rule.atom_ls[atom_ind]
                        pred_ht_dict = self.ht_dict_train[atom.pred_name]

                        gen_latent = np.random.rand() > observed_prob
                        if sub_ind == 1:
                            gen_latent = np.random.rand() > 0.5
                        c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                                   atom, pred_ht_dict, gen_latent)

                        assert atom_succ

                        succ = succ and atom_succ
                        obs_list.append(not islatent)

                        sample_buff[atom_ind].append(c_ls)
                        latent_mask_buff[atom_ind].append(1 if islatent else 0)
                        neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                    if succ:
                        if any(obs_list) or ((sub_ind == 1) and pos_query_succ):

                            for i in range(len(rule.atom_ls)):
                                samples[i][1].extend(sample_buff[i])
                                latent_mask[i][1].extend(latent_mask_buff[i])
                                neg_mask[i][1].extend(neg_mask_buff[i])

                            if sub_ind == 0:
                                samples[-1][1].append([self.const2ind[consts[0]], self.const2ind[consts[1]]])
                                latent_mask[-1][1].append(1)
                                neg_mask[-1][1].append(1)
                                pos_query_succ = True
                                cnt += 1
                            else:
                                samples[-1][1].append([c1, c2])
                                latent_mask[-1][1].append(0) # sample a negative fact at head
                                neg_mask[-1][1].append(1)

            if cnt >= batchsize:

                yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

                samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                cnt = 0

        yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r



    def get_batch_rnd(self, observed_prob=0.7, filter_latent=True, closed_world=False, filter_observed=False):
        """
            return a batch of gnd formulae by random sampling with controllable bias towards those containing
            observed variables. The overall sampling logic is that:
                1) rnd sample a rule from rule_ls
                2) shuffle the predicates contained in the rule
                3) for each of these predicates, with (observed_prob) it will be instantiated as observed variable, and
                   for (1-observed_prob) if will be simply uniformly instantiated.
                3.1) if observed var, then sample from the knowledge base, which is self.fact_dict, if failed for any
                     reason, go to 3.2)
                3.2) if uniformly sample, then for each logic variable in the predicate, instantiate it with a uniform
                     sample from the corresponding constant dict

        :param observed_prob:
            probability of instantiating a predicate as observed variable
        :param filter_latent:
            filter out ground formula containing only latent vars
        :param closed_world:
            if set True, reduce the sampling space of all predicates not in the test_dict to the set specified in
            fact_dict
        :param filter_observed:
            filter out ground formula containing only observed vars
        :return:

        """

        batch_neg_mask = [[] for _ in range(len(self.rule_ls))]
        batch_latent_var_inds = [[] for _ in range(len(self.rule_ls))]
        batch_observed_vars = [[] for _ in range(len(self.rule_ls))]
        observed_rule_cnts = [0.0 for _ in range(len(self.rule_ls))]
        flat_latent_vars = dict()

        cnt = 0

        inds = list(range(len(self.rule_ls)))

        while cnt < self.batchsize:

            # randomly sample a formula
            if self.shuffle_sampling:
                shuffle(inds)

            for ind in inds:

                rule = self.rule_ls[ind]
                atom_key_dict = self.atom_key_dict_ls[ind]
                sub = [None] * len(rule.rule_vars)  # substitutions

                # randomly sample an atom from the formula
                atom_inds = list(range(len(rule.atom_ls)))
                shuffle(atom_inds)
                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    atom_dict = atom_key_dict[atom.pred_name]

                    # instantiate the predicate
                    self._instantiate_pred(atom, atom_dict, sub, rule, observed_prob)

                    # if variable substitution is complete already then exit
                    if not (None in sub):
                        break

                # generate latent and observed var labels and their negation masks
                latent_vars, observed_vars, \
                latent_neg_mask, observed_neg_mask = self._gen_mask(rule, sub, closed_world)

                # check sampled ground rule status
                stat_code = self._get_rule_stat(observed_vars, latent_vars, observed_neg_mask,
                                                filter_latent, filter_observed)

                # is a valid sample with only observed vars and does not have negation on all of them
                if stat_code == FULL_OBSERVERED:
                    observed_rule_cnts[ind] += 1

                    cnt += 1

                # is a valid sample
                elif stat_code == GOOD:
                    batch_neg_mask[ind].append([latent_neg_mask, observed_neg_mask])

                    for latent_var in latent_vars:
                        if latent_var not in flat_latent_vars:
                            flat_latent_vars[latent_var] = len(flat_latent_vars)

                    batch_latent_var_inds[ind].append([flat_latent_vars[e] for e in latent_vars])
                    batch_observed_vars[ind].append(observed_vars)

                    cnt += 1

                # not a valid sample
                else:
                    continue

                if cnt >= self.batchsize:
                    break

        flat_list = sorted([(k, v) for k, v in flat_latent_vars.items()], key=lambda x: x[1])
        flat_list = [e[0] for e in flat_list]

        return batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars

    def reset(self):
        self.rule_gens = [self.generate_gnd_rule(rule) for rule in self.rule_ls]

    def get_stats(self):

        num_ents = sum([len(v) for k, v in self.const_sort_dict.items()])
        num_rels = len(PRED_DICT)
        num_facts = sum([len(v) for k, v in self.fact_dict.items()])
        num_queries = len(self.test_fact_ls)

        num_gnd_atom = 0
        for pred_name, pred in PRED_DICT.items():
            cnt = 1
            for var_type in pred.var_types:
                cnt *= len(self.const_sort_dict[var_type])
            num_gnd_atom += cnt

        num_gnd_rule = 0
        for rule in self.rule_ls:
            cnt = 1
            for var_type in rule.rule_vars.values():
                cnt *= len(self.const_sort_dict[var_type])
            num_gnd_rule += cnt

        return num_ents, num_rels, num_facts, num_queries, num_gnd_atom, num_gnd_rule

import pickle

import sys, os

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from collections import defaultdict
from common.cmd_args import cmd_args
from os.path import join as joinpath
from data_process.dataset import Dataset
from common.utils import iterline


def get_hits_mrr(convert_ind=False):
    truth_path = joinpath(cmd_args.data_root, 'truths.pckl')
    rank_path = joinpath(cmd_args.data_root, 'rank_list.txt')

    top_k = 10
    raw = True

    if not raw:
        truths = pickle.load(open(truth_path, 'rb'))
        tail_query, head_query = truths['query_head'], truths['query_tail']  # this is correct

    hits = 0
    hits_by_q = defaultdict(list)
    ranks = 0
    ranks_by_q = defaultdict(list)
    rranks = 0.

    if convert_ind:
        dataset = Dataset(cmd_args.data_root, 1, 1, load_method=1)
        ind2const = dict([(i, const) for i, const in enumerate(dataset.const_sort_dict['type'])])

    line_cnt = 0

    for line in iterline(rank_path):

        l = line.split(',')
        if convert_ind:
            l = [l[0]] + [ind2const[int(e)] for e in l[1:]]

        assert (len(l) > 3)
        q, h, t = l[0:3]
        this_preds = l[3:]
        assert (h == this_preds[-1])
        hitted = 0.

        if not raw:
            if q.startswith('inv_'):
                q_ = q[len('inv_'):]
                also_correct = tail_query[(q_, t)]
            else:
                also_correct = head_query[(q, t)]
            also_correct = set(also_correct)
            assert (h in also_correct)
            this_preds_filtered = set(this_preds[:-1]) - also_correct
            this_preds_filtered.add(this_preds[-1])
            if len(this_preds_filtered) <= top_k:
                hitted = 1.
            rank = len(this_preds_filtered)
        else:
            if len(this_preds) <= top_k:
                hitted = 1.
            rank = len(this_preds)

        hits += hitted
        ranks += rank
        rranks += 1. / rank
        hits_by_q[q].append(hitted)
        ranks_by_q[q].append(rank)
        line_cnt += 1

    with open(joinpath(cmd_args.data_root, 'evaluation.txt'), 'w') as f:
        f.write('Hits at %d is %0.4f\n' % (top_k, hits / line_cnt))
        f.write('Mean rank %0.2f\n' % (1. * ranks / line_cnt))
        f.write('Mean Reciprocal Rank %0.4f\n' % (1. * rranks / line_cnt))


def gen_eval_query(dataset, const2ind=None, pickone=None):
    const_ls = dataset.const_sort_dict['type']

    toindex = lambda x: x
    if const2ind is not None:
        toindex = lambda x: const2ind[x]

    for val, pred_name, consts in dataset.test_fact_ls:
        c1, c2 = toindex(consts[0]), toindex(consts[1])

        if pickone is not None:
            if pred_name != pickone:
                continue

        X, invX = [], []
        for const in const_ls:

            if const not in dataset.ht_dict[pred_name][0][consts[0]]:
                X.append([c1, toindex(const)])
            if const not in dataset.ht_dict[pred_name][1][consts[1]]:
                invX.append([toindex(const), c2])

        yield pred_name, X, invX, [[c1, c2]]


if __name__ == '__main__':
    get_hits_mrr()

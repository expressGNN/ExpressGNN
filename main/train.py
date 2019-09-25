import torch
from model.mean_field_posterior import FactorizedPosterior
from model.gcn import GCN, TrainableEmbedding
from model.mln import ConditionalMLN
from data_process.dataset import Dataset
from common.cmd_args import cmd_args
from tqdm import tqdm
import torch.optim as optim
from model.graph import KnowledgeGraph
from common.predicate import PRED_DICT
from common.utils import EarlyStopMonitor, get_lr, count_parameters
from common.evaluate import gen_eval_query
from itertools import chain
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from os.path import join as joinpath
import os
import math
from collections import Counter


def train(cmd_args):
  if not os.path.exists(cmd_args.exp_path):
    os.makedirs(cmd_args.exp_path)

  with open(joinpath(cmd_args.exp_path, 'options.txt'), 'w') as f:
    param_dict = vars(cmd_args)
    for param in param_dict:
      f.write(param + ' = ' + str(param_dict[param]) + '\n')

  logpath = joinpath(cmd_args.exp_path, 'eval.result')
  param_cnt_path = joinpath(cmd_args.exp_path, 'param_count.txt')

  # dataset and KG
  dataset = Dataset(cmd_args.data_root, cmd_args.batchsize,
                    cmd_args.shuffle_sampling, load_method=cmd_args.load_method)
  kg = KnowledgeGraph(dataset.fact_dict, PRED_DICT, dataset)

  # model
  if cmd_args.use_gcn == 1:
    gcn = GCN(kg, cmd_args.embedding_size - cmd_args.gcn_free_size, cmd_args.gcn_free_size,
              num_hops=cmd_args.num_hops, num_layers=cmd_args.num_mlp_layers,
              transductive=cmd_args.trans == 1).to(cmd_args.device)
  else:
    gcn = TrainableEmbedding(kg, cmd_args.embedding_size).to(cmd_args.device)
  posterior_model = FactorizedPosterior(kg, cmd_args.embedding_size, cmd_args.slice_dim).to(cmd_args.device)
  mln = ConditionalMLN(cmd_args, dataset.rule_ls)

  if cmd_args.model_load_path is not None:
    gcn.load_state_dict(torch.load(joinpath(cmd_args.model_load_path, 'gcn.model')))
    posterior_model.load_state_dict(torch.load(joinpath(cmd_args.model_load_path, 'posterior.model')))

  # optimizers
  monitor = EarlyStopMonitor(cmd_args.patience)
  all_params = chain.from_iterable([posterior_model.parameters(), gcn.parameters()])
  optimizer = optim.Adam(all_params, lr=cmd_args.learning_rate, weight_decay=cmd_args.l2_coef)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cmd_args.lr_decay_factor,
                                                   patience=cmd_args.lr_decay_patience, min_lr=cmd_args.lr_decay_min)

  with open(param_cnt_path, 'w') as f:
    cnt_gcn_params = count_parameters(gcn)
    cnt_posterior_params = count_parameters(posterior_model)
    if cmd_args.use_gcn == 1:
      f.write('GCN params count: %d\n' % cnt_gcn_params)
    elif cmd_args.use_gcn == 0:
      f.write('plain params count: %d\n' % cnt_gcn_params)
    f.write('posterior params count: %d\n' % cnt_posterior_params)
    f.write('Total params count: %d\n' % (cnt_gcn_params + cnt_posterior_params))

  if cmd_args.no_train == 1:
    cmd_args.num_epochs = 0

  # for Freebase data
  if cmd_args.load_method == 1:

    # prepare data for M-step
    tqdm.write('preparing data for M-step...')
    pred_arg1_set_arg2 = dict()
    pred_arg2_set_arg1 = dict()
    pred_fact_set = dict()
    for pred in dataset.fact_dict_2:
      pred_arg1_set_arg2[pred] = dict()
      pred_arg2_set_arg1[pred] = dict()
      pred_fact_set[pred] = set()
      for _, args in dataset.fact_dict_2[pred]:
        if args[0] not in pred_arg1_set_arg2[pred]:
          pred_arg1_set_arg2[pred][args[0]] = set()
        if args[1] not in pred_arg2_set_arg1[pred]:
          pred_arg2_set_arg1[pred][args[1]] = set()
        pred_arg1_set_arg2[pred][args[0]].add(args[1])
        pred_arg2_set_arg1[pred][args[1]].add(args[0])
        pred_fact_set[pred].add(args)

    grounded_rules = []
    for rule_idx, rule in enumerate(dataset.rule_ls):
      grounded_rules.append(set())
      body_atoms = []
      head_atom = None
      for atom in rule.atom_ls:
        if atom.neg:
          body_atoms.append(atom)
        elif head_atom is None:
          head_atom = atom
      # atom in body must be observed
      assert len(body_atoms) <= 2
      if len(body_atoms) > 0:
        body1 = body_atoms[0]
        for _, body1_args in dataset.fact_dict_2[body1.pred_name]:
          var2arg = dict()
          var2arg[body1.var_name_ls[0]] = body1_args[0]
          var2arg[body1.var_name_ls[1]] = body1_args[1]
          for body2 in body_atoms[1:]:
            if body2.var_name_ls[0] in var2arg:
              if var2arg[body2.var_name_ls[0]] in pred_arg1_set_arg2[body2.pred_name]:
                for body2_arg2 in pred_arg1_set_arg2[body2.pred_name][var2arg[body2.var_name_ls[0]]]:
                  var2arg[body2.var_name_ls[1]] = body2_arg2
                  grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))
            elif body2.var_name_ls[1] in var2arg:
              if var2arg[body2.var_name_ls[1]] in pred_arg2_set_arg1[body2.pred_name]:
                for body2_arg1 in pred_arg2_set_arg1[body2.pred_name][var2arg[body2.var_name_ls[1]]]:
                  var2arg[body2.var_name_ls[0]] = body2_arg1
                  grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))

    # Collect head atoms derived by grounded formulas
    grounded_obs = dict()
    grounded_hid = dict()
    grounded_hid_score = dict()
    cnt_hid = 0
    for rule_idx in range(len(dataset.rule_ls)):
      rule = dataset.rule_ls[rule_idx]
      for var2arg in grounded_rules[rule_idx]:
        var2arg = dict(var2arg)
        head_atom = rule.atom_ls[-1]
        assert not head_atom.neg    # head atom
        pred = head_atom.pred_name
        args = (var2arg[head_atom.var_name_ls[0]], var2arg[head_atom.var_name_ls[1]])
        if args in pred_fact_set[pred]:
          if (pred, args) not in grounded_obs:
            grounded_obs[(pred, args)] = []
          grounded_obs[(pred, args)].append(rule_idx)
        else:
          if (pred, args) not in grounded_hid:
            grounded_hid[(pred, args)] = []
          grounded_hid[(pred, args)].append(rule_idx)
    tqdm.write('observed: %d, hidden: %d' % (len(grounded_obs), len(grounded_hid)))

    # Aggregate atoms by predicates for fast inference
    pred_aggregated_hid = dict()
    pred_aggregated_hid_args = dict()
    for (pred, args) in grounded_hid:
      if pred not in pred_aggregated_hid:
        pred_aggregated_hid[pred] = []
      if pred not in pred_aggregated_hid_args:
        pred_aggregated_hid_args[pred] = []
      pred_aggregated_hid[pred].append((dataset.const2ind[args[0]], dataset.const2ind[args[1]]))
      pred_aggregated_hid_args[pred].append(args)
    pred_aggregated_hid_list = [[pred, pred_aggregated_hid[pred]] for pred in sorted(pred_aggregated_hid.keys())]

    for current_epoch in range(cmd_args.num_epochs):

      # E-step: optimize the parameters in the posterior model
      num_batches = int(math.ceil(len(dataset.test_fact_ls) / cmd_args.batchsize))

      pbar = tqdm(total=num_batches)
      acc_loss = 0.0
      cur_batch = 0

      for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
          dataset.get_batch_by_q(cmd_args.batchsize):

        node_embeds = gcn(dataset)

        loss = 0.0
        r_cnt = 0
        for ind, samples in enumerate(samples_by_r):
          neg_mask = neg_mask_by_r[ind]
          latent_mask = latent_mask_by_r[ind]
          obs_var = obs_var_by_r[ind]
          neg_var = neg_var_by_r[ind]

          if sum([len(e[1]) for e in neg_mask]) == 0:
            continue

          potential, posterior_prob, obs_xent = posterior_model([samples, neg_mask, latent_mask,
                                                                 obs_var, neg_var],
                                                                node_embeds, fast_mode=True)

          if cmd_args.no_entropy == 1:
            entropy = 0
          else:
            entropy = compute_entropy(posterior_prob) / cmd_args.entropy_temp

          loss += - (potential.sum() * dataset.rule_ls[ind].weight + entropy) / (potential.size(0) + 1e-6) + obs_xent

          r_cnt += 1

        if r_cnt > 0:
          loss /= r_cnt
          acc_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        pbar.update()
        cur_batch += 1
        pbar.set_description(
          'Epoch %d, train loss: %.4f, lr: %.4g' % (current_epoch, acc_loss / cur_batch, get_lr(optimizer)))

      # M-step: optimize the weights of logic rules
      with torch.no_grad():
        posterior_prob = posterior_model(pred_aggregated_hid_list, node_embeds, fast_inference_mode=True)
        for pred_i, (pred, var_ls) in enumerate(pred_aggregated_hid_list):
          for var_i, var in enumerate(var_ls):
            args = pred_aggregated_hid_args[pred][var_i]
            grounded_hid_score[(pred, args)] = posterior_prob[pred_i][var_i]

        rule_weight_gradient = torch.zeros(len(dataset.rule_ls))
        for (pred, args) in grounded_obs:
          for rule_idx in set(grounded_obs[(pred, args)]):
            rule_weight_gradient[rule_idx] += 1.0 - compute_MB_proba(dataset.rule_ls, grounded_obs[(pred, args)])
        for (pred, args) in grounded_hid:
          for rule_idx in set(grounded_hid[(pred, args)]):
            target = grounded_hid_score[(pred, args)]
            rule_weight_gradient[rule_idx] += target - compute_MB_proba(dataset.rule_ls, grounded_hid[(pred, args)])

        for rule_idx, rule in enumerate(dataset.rule_ls):
          rule.weight += cmd_args.learning_rate_rule_weights * rule_weight_gradient[rule_idx]
          print(dataset.rule_ls[rule_idx].weight, end=' ')

      pbar.close()

      # validation
      with torch.no_grad():
        node_embeds = gcn(dataset)

        valid_loss = 0.0
        cnt_batch = 0
        for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
            dataset.get_batch_by_q(cmd_args.batchsize, validation=True):
          loss = 0.0
          r_cnt = 0
          for ind, samples in enumerate(samples_by_r):
            neg_mask = neg_mask_by_r[ind]
            latent_mask = latent_mask_by_r[ind]
            obs_var = obs_var_by_r[ind]
            neg_var = neg_var_by_r[ind]

            if sum([len(e[1]) for e in neg_mask]) == 0:
              continue

            valid_potential, valid_prob, valid_obs_xent = posterior_model([samples, neg_mask, latent_mask,
                                                                           obs_var, neg_var],
                                                                          node_embeds, fast_mode=True)

            if cmd_args.no_entropy == 1:
              valid_entropy = 0
            else:
              valid_entropy = compute_entropy(valid_prob) / cmd_args.entropy_temp

            loss += - (valid_potential.sum() + valid_entropy) / (valid_potential.size(0) + 1e-6) + valid_obs_xent

            r_cnt += 1

          if r_cnt > 0:
            loss /= r_cnt
            valid_loss += loss.item()

          cnt_batch += 1

        tqdm.write('Epoch %d, valid loss: %.4f' % (current_epoch, valid_loss / cnt_batch))

        should_stop = monitor.update(valid_loss)
        scheduler.step(valid_loss)

        is_current_best = monitor.cnt == 0
        if is_current_best:
          savepath = joinpath(cmd_args.exp_path, 'saved_model')
          os.makedirs(savepath, exist_ok=True)
          torch.save(gcn.state_dict(), joinpath(savepath, 'gcn.model'))
          torch.save(posterior_model.state_dict(), joinpath(savepath, 'posterior.model'))

        should_stop = should_stop or (current_epoch + 1 == cmd_args.num_epochs)

        if should_stop:
          tqdm.write('Early stopping')
          break

    # ======================= generate rank list =======================
    node_embeds = gcn(dataset)

    pbar = tqdm(total=len(dataset.test_fact_ls))
    pbar.write('*' * 10 + ' Evaluation ' + '*' * 10)
    rrank = 0.0
    hits = 0.0
    cnt = 0

    rrank_pred = dict([(pred_name, 0.0) for pred_name in PRED_DICT])
    hits_pred = dict([(pred_name, 0.0) for pred_name in PRED_DICT])
    cnt_pred = dict([(pred_name, 0.0) for pred_name in PRED_DICT])

    for pred_name, X, invX, sample in gen_eval_query(dataset, const2ind=kg.ent2idx):
      x_mat = np.array(X)
      invx_mat = np.array(invX)
      sample_mat = np.array(sample)

      tail_score, head_score, true_score = posterior_model([pred_name, x_mat, invx_mat, sample_mat], node_embeds,
                                                           batch_mode=True)

      rank = torch.sum(tail_score >= true_score).item() + 1
      rrank += 1.0 / rank
      hits += 1 if rank <= 10 else 0

      rrank_pred[pred_name] += 1.0 / rank
      hits_pred[pred_name] += 1 if rank <= 10 else 0

      rank = torch.sum(head_score >= true_score).item() + 1
      rrank += 1.0 / rank
      hits += 1 if rank <= 10 else 0

      rrank_pred[pred_name] += 1.0 / rank
      hits_pred[pred_name] += 1 if rank <= 10 else 0

      cnt_pred[pred_name] += 2
      cnt += 2

      if cnt % 100 == 0:
        with open(logpath, 'w') as f:
          f.write('%i sample eval\n' % cnt)
          f.write('mmr %.4f\n' % (rrank / cnt))
          f.write('hits %.4f\n' % (hits / cnt))

          f.write('\n')
          for pred_name in PRED_DICT:
            if cnt_pred[pred_name] == 0:
              continue
            f.write('mmr %s %.4f\n' % (pred_name, rrank_pred[pred_name] / cnt_pred[pred_name]))
            f.write('hits %s %.4f\n' % (pred_name, hits_pred[pred_name] / cnt_pred[pred_name]))

      pbar.update()

    with open(logpath, 'w') as f:
      f.write('complete\n')
      f.write('mmr %.4f\n' % (rrank / cnt))
      f.write('hits %.4f\n' % (hits / cnt))
      f.write('\n')

      tqdm.write('mmr %.4f\n' % (rrank / cnt))
      tqdm.write('hits %.4f\n' % (hits / cnt))

      for pred_name in PRED_DICT:
        if cnt_pred[pred_name] == 0:
          continue
        f.write('mmr %s %.4f\n' % (pred_name, rrank_pred[pred_name] / cnt_pred[pred_name]))
        f.write('hits %s %.4f\n' % (pred_name, hits_pred[pred_name] / cnt_pred[pred_name]))

    os.system('mv %s %s' % (logpath, joinpath(cmd_args.exp_path,
                                              'performance_hits_%.4f_mmr_%.4f.txt' % ((hits / cnt), (rrank / cnt)))))
    pbar.close()

  # for Kinship / UW-CSE / Cora data
  elif cmd_args.load_method == 0:
    for current_epoch in range(cmd_args.num_epochs):
      pbar = tqdm(range(cmd_args.num_batches))
      acc_loss = 0.0

      for k in pbar:
        node_embeds = gcn(dataset)

        batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars = dataset.get_batch_rnd(
          observed_prob=cmd_args.observed_prob,
          filter_latent=cmd_args.filter_latent == 1,
          closed_world=cmd_args.closed_world == 1,
          filter_observed=1)

        posterior_prob = posterior_model(flat_list, node_embeds)

        if cmd_args.no_entropy == 1:
          entropy = 0
        else:
          entropy = compute_entropy(posterior_prob) / cmd_args.entropy_temp

        entropy = entropy.to('cpu')
        posterior_prob = posterior_prob.to('cpu')

        potential = mln(batch_neg_mask, batch_latent_var_inds, observed_rule_cnts, posterior_prob,
                        flat_list, batch_observed_vars)

        optimizer.zero_grad()

        loss = - (potential + entropy) / cmd_args.batchsize
        acc_loss += loss.item()

        loss.backward()

        optimizer.step()

        pbar.set_description('train loss: %.4f, lr: %.4g' % (acc_loss / (k + 1), get_lr(optimizer)))

      # test
      node_embeds = gcn(dataset)
      with torch.no_grad():

        posterior_prob = posterior_model([(e[1], e[2]) for e in dataset.test_fact_ls], node_embeds)
        posterior_prob = posterior_prob.to('cpu')

        label = np.array([e[0] for e in dataset.test_fact_ls])
        test_log_prob = float(np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))

        auc_roc = roc_auc_score(label, posterior_prob.numpy())
        auc_pr = average_precision_score(label, posterior_prob.numpy())

        tqdm.write('Epoch: %d, train loss: %.4f, test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (
          current_epoch, acc_loss / cmd_args.num_batches, auc_roc, auc_pr, test_log_prob))
        # tqdm.write(str(posterior_prob[:10]))

      # validation for early stop
      valid_sample = []
      valid_label = []
      for pred_name in dataset.valid_dict_2:
        for val, consts in dataset.valid_dict_2[pred_name]:
          valid_sample.append((pred_name, consts))
          valid_label.append(val)
      valid_label = np.array(valid_label)
      
      valid_prob = posterior_model(valid_sample, node_embeds)
      valid_prob = valid_prob.to('cpu')
      
      valid_log_prob = float(np.sum(np.log(np.clip(np.abs((1 - valid_label) - valid_prob.numpy()), 1e-6, 1 - 1e-6))))
      
      # tqdm.write('epoch: %d, valid log prob: %.4f' % (current_epoch, valid_log_prob))
      # 
      # should_stop = monitor.update(-valid_log_prob)
      # scheduler.step(valid_log_prob)
      # 
      # is_current_best = monitor.cnt == 0
      # if is_current_best:
      #   savepath = joinpath(cmd_args.exp_path, 'saved_model')
      #   os.makedirs(savepath, exist_ok=True)
      #   torch.save(gcn.state_dict(), joinpath(savepath, 'gcn.model'))
      #   torch.save(posterior_model.state_dict(), joinpath(savepath, 'posterior.model'))
      #
      # should_stop = should_stop or (current_epoch + 1 == cmd_args.num_epochs)
      #
      # if should_stop:
      #   tqdm.write('Early stopping')
      #   break

    # evaluation after training
    node_embeds = gcn(dataset)
    with torch.no_grad():
      posterior_prob = posterior_model([(e[1], e[2]) for e in dataset.test_fact_ls], node_embeds)
      posterior_prob = posterior_prob.to('cpu')

      label = np.array([e[0] for e in dataset.test_fact_ls])
      test_log_prob = float(np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))

      auc_roc = roc_auc_score(label, posterior_prob.numpy())
      auc_pr = average_precision_score(label, posterior_prob.numpy())

      tqdm.write('test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (auc_roc, auc_pr, test_log_prob))


def compute_entropy(posterior_prob):
  eps = 1e-6
  posterior_prob.clamp_(eps, 1 - eps)
  compl_prob = 1 - posterior_prob
  entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
  return entropy


def compute_MB_proba(rule_ls, ls_rule_idx):
  rule_idx_cnt = Counter(ls_rule_idx)
  numerator = 0
  for rule_idx in rule_idx_cnt:
    weight = rule_ls[rule_idx].weight
    cnt = rule_idx_cnt[rule_idx]
    numerator += math.exp(weight * cnt)
  return numerator / (numerator + 1.0)


if __name__ == '__main__':
  random.seed(cmd_args.seed)
  np.random.seed(cmd_args.seed)
  torch.manual_seed(cmd_args.seed)

  train(cmd_args)

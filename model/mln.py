import torch
import torch.nn as nn
from common.predicate import PRED_DICT
from itertools import product
import torch.nn.functional as F


class ConditionalMLN(nn.Module):
  
  def __init__(self, cmd_args, rule_list):
    super(ConditionalMLN, self).__init__()
    
    self.rule_weights_lin = nn.Linear(len(rule_list), 1, bias=False)
    self.num_rules = len(rule_list)
    self.soft_logic = False
    
    self.alpha_table = nn.Parameter(torch.tensor([10.0 for _ in range(len(PRED_DICT))], requires_grad=True))
    
    self.predname2ind = dict(e for e in zip(PRED_DICT.keys(), range(len(PRED_DICT))))
    
    if cmd_args.rule_weights_learning == 0:
      self.rule_weights_lin.weight.data = torch.tensor([[rule.weight for rule in rule_list]], dtype=torch.float)
      print('rule weights fixed as pre-defined values\n')
    else:
      self.rule_weights_lin.weight = nn.Parameter(
        torch.tensor([[rule.weight for rule in rule_list]], dtype=torch.float))
      print('rule weights set to pre-defined values, learning weights\n')
  
  def forward(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list,
              observed_vars_ls_ls):
    """
        compute the MLN potential given the posterior probability of latent variables
    :param neg_mask_ls_ls:
    :param posterior_prob_ls_ls:
    :return:
    """
    
    scores = torch.zeros(self.num_rules, dtype=torch.float)
    
    if self.soft_logic:
      pred_name_ls = [e[0] for e in flat_list]
      pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]
    
    for i in range(len(neg_mask_ls_ls)):
      neg_mask_ls = neg_mask_ls_ls[i]
      latent_var_inds_ls = latent_var_inds_ls_ls[i]
      observed_vars_ls = observed_vars_ls_ls[i]
      
      # sum of scores from gnd rules with latent vars
      for j in range(len(neg_mask_ls)):
        
        latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
        latent_var_inds = latent_var_inds_ls[j]
        observed_vars = observed_vars_ls[j]
        
        z_probs = posterior_prob[latent_var_inds].unsqueeze(0)
        
        z_probs = torch.cat([1 - z_probs, z_probs], dim=0)
        
        cartesian_prod = z_probs[:, 0]
        for j in range(1, z_probs.shape[1]):
          cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
          cartesian_prod = cartesian_prod.view(-1)
        
        view_ls = [2 for _ in range(len(latent_neg_mask))]
        cartesian_prod = cartesian_prod.view(*[view_ls])
        
        if self.soft_logic:
          
          # observed alpha
          obs_vals = [e[0] for e in observed_vars]
          pred_names = [e[1] for e in observed_vars]
          pred_inds = [self.predname2ind[pn] for pn in pred_names]
          alpha = self.alpha_table[pred_inds]  # alphas in this formula
          act_alpha = torch.sigmoid(alpha)
          obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0)
                          for i in range(len(observed_vars))]
          tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)
          
          val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
          obs_score = torch.abs(tn_obs_neg_flag - val)
          
          # latent alpha
          inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
          pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
          alpha = self.alpha_table[pred_inds]  # alphas in this formula
          act_alpha = torch.sigmoid(alpha)
          tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)
          
          for ind in inds:
            val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
            val = torch.abs(tn_latent_neg_mask - val)
            cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))
        
        else:
          
          if sum(observed_neg_mask) == 0:
            cartesian_prod[tuple(latent_neg_mask)] = 0.0
        
        scores[i] += cartesian_prod.sum()
      
      # sum of scores from gnd rule with only observed vars
      scores[i] += observed_rule_cnts[i]
    
    return self.rule_weights_lin(scores)
  
  def weight_update(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list,
                    observed_vars_ls_ls):
    closed_wolrd_potentials = torch.zeros(self.num_rules, dtype=torch.float)
    
    if self.soft_logic:
      pred_name_ls = [e[0] for e in flat_list]
      pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]
    
    for i in range(len(neg_mask_ls_ls)):
      neg_mask_ls = neg_mask_ls_ls[i]
      latent_var_inds_ls = latent_var_inds_ls_ls[i]
      observed_vars_ls = observed_vars_ls_ls[i]
      
      # sum of scores from gnd rules with latent vars
      for j in range(len(neg_mask_ls)):
        
        latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
        latent_var_inds = latent_var_inds_ls[j]
        observed_vars = observed_vars_ls[j]
        
        has_pos_atom = False
        for val in observed_neg_mask + latent_neg_mask:
          if val == 1:
            has_pos_atom = True
            break
        
        if has_pos_atom:
          closed_wolrd_potentials[i] += 1
        
        z_probs = posterior_prob[latent_var_inds].unsqueeze(0)
        
        z_probs = torch.cat([1 - z_probs, z_probs], dim=0)
        
        cartesian_prod = z_probs[:, 0]
        for j in range(1, z_probs.shape[1]):
          cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
          cartesian_prod = cartesian_prod.view(-1)
        
        view_ls = [2 for _ in range(len(latent_neg_mask))]
        cartesian_prod = cartesian_prod.view(*[view_ls])
        
        if self.soft_logic:
          
          # observed alpha
          obs_vals = [e[0] for e in observed_vars]
          pred_names = [e[1] for e in observed_vars]
          pred_inds = [self.predname2ind[pn] for pn in pred_names]
          alpha = self.alpha_table[pred_inds]  # alphas in this formula
          act_alpha = torch.sigmoid(alpha)
          obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0)
                          for i in range(len(observed_vars))]
          tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)
          
          val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
          obs_score = torch.abs(tn_obs_neg_flag - val)
          
          # latent alpha
          inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
          pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
          alpha = self.alpha_table[pred_inds]  # alphas in this formula
          act_alpha = torch.sigmoid(alpha)
          tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)
          
          for ind in inds:
            val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
            val = torch.abs(tn_latent_neg_mask - val)
            cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))
        
        else:
          
          if sum(observed_neg_mask) == 0:
            cartesian_prod[tuple(latent_neg_mask)] = 0.0
        
      weight_grad = closed_wolrd_potentials
      
      return weight_grad

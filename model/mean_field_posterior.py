import torch
import torch.nn as nn
from common.cmd_args import cmd_args
from common.predicate import PRED_DICT
import torch.nn.functional as F


class FactorizedPosterior(nn.Module):
  def __init__(self, graph, latent_dim, slice_dim=5):
    super(FactorizedPosterior, self).__init__()

    self.graph = graph
    self.latent_dim = latent_dim

    self.xent_loss = F.binary_cross_entropy_with_logits

    self.device = cmd_args.device

    self.num_rels = graph.num_rels
    self.ent2idx = graph.ent2idx
    self.rel2idx = graph.rel2idx
    self.idx2rel = graph.idx2rel

    if cmd_args.load_method == 1:
      self.params_u_R = nn.ModuleList()
      self.params_W_R = nn.ModuleList()
      self.params_V_R = nn.ModuleList()
      for idx in range(self.num_rels):
        rel = self.idx2rel[idx]
        num_args = PRED_DICT[rel].num_args
        self.params_W_R.append(nn.Bilinear(num_args * latent_dim, num_args * latent_dim, slice_dim, bias=False))
        self.params_V_R.append(nn.Linear(num_args * latent_dim, slice_dim, bias=True))
        self.params_u_R.append(nn.Linear(slice_dim, 1, bias=False))
    elif cmd_args.load_method == 0:
      self.params_u_R = nn.ParameterList()
      self.params_W_R = nn.ModuleList()
      self.params_V_R = nn.ModuleList()
      self.params_b_R = nn.ParameterList()
      for idx in range(self.num_rels):
        rel = self.idx2rel[idx]
        num_args = PRED_DICT[rel].num_args
        self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
        self.params_W_R.append(nn.Bilinear(num_args * latent_dim, num_args * latent_dim, slice_dim, bias=False))
        self.params_V_R.append(nn.Linear(num_args * latent_dim, slice_dim, bias=False))
        self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))

  def forward(self, latent_vars, node_embeds, batch_mode=False, fast_mode=False, fast_inference_mode=False):
    """
    compute posterior probabilities of specified latent variables

    :param latent_vars:
        list of latent variables (i.e. unobserved facts)
    :param node_embeds:
        node embeddings
    :return:
        n-dim vector, probability of corresponding latent variable being True
    """
    # this mode is only for fast inference on Freebase data
    if fast_inference_mode:
      assert cmd_args.load_method == 1

      samples = latent_vars
      scores = []

      for ind in range(len(samples)):
        pred_name, pred_sample = samples[ind]

        rel_idx = self.rel2idx[pred_name]

        sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(cmd_args.device) # (bsize, 2)

        sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

        sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
                                                           self.params_V_R[rel_idx](sample_query))).view(-1) # (bsize)
        scores.append(torch.sigmoid(sample_score))

      return scores

    # this mode is only for fast training on Freebase data
    elif fast_mode:
      assert cmd_args.load_method == 1

      samples, neg_mask, latent_mask, obs_var, neg_var = latent_vars
      scores = []
      obs_probs = []
      neg_probs = []

      pos_mask_mat = torch.tensor([pred_mask[1] for pred_mask in neg_mask], dtype=torch.float).to(cmd_args.device)
      neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
      latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(cmd_args.device)
      obs_mask_mat = (latent_mask_mat == 0).type(torch.float)

      for ind in range(len(samples)):
        pred_name, pred_sample = samples[ind]
        _, obs_sample = obs_var[ind]
        _, neg_sample = neg_var[ind]

        rel_idx = self.rel2idx[pred_name]

        sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(cmd_args.device)
        obs_mat = torch.tensor(obs_sample, dtype=torch.long).to(cmd_args.device)
        neg_mat = torch.tensor(neg_sample, dtype=torch.long).to(cmd_args.device)

        sample_mat = torch.cat([sample_mat, obs_mat, neg_mat], dim=0)

        sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

        sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
                                                           self.params_V_R[rel_idx](sample_query))).view(-1)
        var_prob = sample_score[len(pred_sample):]
        obs_prob = var_prob[:len(obs_sample)]
        neg_prob = var_prob[len(obs_sample):]
        sample_score = sample_score[:len(pred_sample)]

        scores.append(sample_score)
        obs_probs.append(obs_prob)
        neg_probs.append(neg_prob)

      score_mat = torch.stack(scores, dim=0)
      score_mat = torch.sigmoid(score_mat)

      pos_score = (1 - score_mat) * pos_mask_mat
      neg_score = score_mat * neg_mask_mat

      potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)

      obs_mat = torch.cat(obs_probs, dim=0)

      if obs_mat.size(0) == 0:
        obs_loss = 0.0
      else:
        obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')

      neg_mat = torch.cat(neg_probs, dim=0)
      if neg_mat.size(0) != 0:
        obs_loss += self.xent_loss(obs_mat, torch.zeros_like(neg_mat), reduction='sum')

      obs_loss /= (obs_mat.size(0) + neg_mat.size(0) + 1e-6)

      return potential, (score_mat * latent_mask_mat).view(-1), obs_loss

    elif batch_mode:
      assert cmd_args.load_method == 1

      pred_name, x_mat, invx_mat, sample_mat = latent_vars

      rel_idx = self.rel2idx[pred_name]

      x_mat = torch.tensor(x_mat, dtype=torch.long).to(cmd_args.device)
      invx_mat = torch.tensor(invx_mat, dtype=torch.long).to(cmd_args.device)
      sample_mat = torch.tensor(sample_mat, dtype=torch.long).to(cmd_args.device)

      tail_query = torch.cat([node_embeds[x_mat[:, 0]], node_embeds[x_mat[:, 1]]], dim=1)
      head_query = torch.cat([node_embeds[invx_mat[:, 0]], node_embeds[invx_mat[:, 1]]], dim=1)
      true_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

      tail_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](tail_query, tail_query) +
                                                       self.params_V_R[rel_idx](tail_query))).view(-1)

      head_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](head_query, head_query) +
                                                       self.params_V_R[rel_idx](head_query))).view(-1)

      true_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](true_query, true_query) +
                                                       self.params_V_R[rel_idx](true_query))).view(-1)

      probas_tail = torch.sigmoid(tail_score)
      probas_head = torch.sigmoid(head_score)
      probas_true = torch.sigmoid(true_score)

      return probas_tail, probas_head, probas_true

    else:
      assert cmd_args.load_method == 0

      probas = torch.zeros(len(latent_vars)).to(cmd_args.device)
      for i in range(len(latent_vars)):
        rel, args = latent_vars[i]
        args_embed = torch.cat([node_embeds[self.ent2idx[arg]] for arg in args], 0)
        rel_idx = self.rel2idx[rel]

        score = self.params_u_R[rel_idx].dot(
          torch.tanh(self.params_W_R[rel_idx](args_embed, args_embed) +
                     self.params_V_R[rel_idx](args_embed) +
                     self.params_b_R[rel_idx])
        )
        proba = torch.sigmoid(score)
        probas[i] = proba

      return probas

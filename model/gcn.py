import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.mlp import MLP
from common.cmd_args import cmd_args


def prepare_node_feature(graph, transductive=True):
  if transductive:
    node_feat = torch.zeros(graph.num_nodes,                          # for transductive GCN
                            graph.num_ents + graph.num_rels)
    
    const_nodes = []
    for i in graph.idx2node:
      if isinstance(graph.idx2node[i], str):      # const (entity) node
        const_nodes.append(i)
        node_feat[i][i] = 1
      elif isinstance(graph.idx2node[i], tuple):  # fact node
        rel, args = graph.idx2node[i]
        node_feat[i][graph.num_ents + graph.rel2idx[rel]] = 1
  else:
    node_feat = torch.zeros(graph.num_nodes, 1 + graph.num_rels)      # for inductive GCN
    const_nodes = []
    for i in graph.idx2node:
      if isinstance(graph.idx2node[i], str):      # const (entity) node
        node_feat[i][0] = 1
        const_nodes.append(i)
      elif isinstance(graph.idx2node[i], tuple):  # fact node
        rel, args = graph.idx2node[i]
        node_feat[i][1 + graph.rel2idx[rel]] = 1
  
  return node_feat, torch.LongTensor(const_nodes)


class TrainableEmbedding(nn.Module):
  def __init__(self, graph, latent_dim):
    super(TrainableEmbedding, self).__init__()
    
    self.num_ents = graph.num_ents
    self.ent_embeds = nn.Embedding(self.num_ents, latent_dim)
    self.ents = torch.arange(self.num_ents).to(cmd_args.device)
    
    torch.nn.init.kaiming_uniform_(self.ent_embeds.weight)

  def forward(self, batch_data):
    node_embeds = self.ent_embeds(self.ents)
    return node_embeds
    

class GCN(nn.Module):
  def __init__(self, graph, latent_dim, free_dim, num_hops=5, num_layers=2, transductive=True):
    super(GCN, self).__init__()
    
    self.graph = graph
    self.latent_dim = latent_dim
    self.free_dim = free_dim
    self.num_hops = num_hops
    self.num_layers = num_layers
    
    self.num_ents = graph.num_ents
    self.num_rels = graph.num_rels
    self.num_nodes = graph.num_nodes
    self.num_edges = graph.num_edges
    self.num_edge_types = len(graph.edge_type2idx)
    
    self.edge2node_in, self.edge2node_out, self.node_degree, \
        self.edge_type_masks, self.edge_direction_masks = self.gen_edge2node_mapping()
    
    self.node_feat, self.const_nodes = prepare_node_feature(graph, transductive=transductive)
    
    if not transductive:
      self.node_feat_dim = 1 + self.num_rels
    else:
      self.node_feat_dim = self.num_ents + self.num_rels
    
    self.init_node_linear = nn.Linear(self.node_feat_dim, latent_dim, bias=False)
    
    for param in self.init_node_linear.parameters():
      param.requires_grad = False

    self.node_feat = self.node_feat.to(cmd_args.device)
    self.const_nodes = self.const_nodes.to(cmd_args.device)
    self.edge2node_in = self.edge2node_in.to(cmd_args.device)
    self.edge2node_out = self.edge2node_out.to(cmd_args.device)
    self.edge_type_masks = [mask.to(cmd_args.device) for mask in self.edge_type_masks]
    self.edge_direction_masks = [mask.to(cmd_args.device) for mask in self.edge_direction_masks]

    self.MLPs = nn.ModuleList()
    for _ in range(self.num_hops):
      self.MLPs.append(MLP(input_size=self.latent_dim, num_layers=self.num_layers,
                           hidden_size=self.latent_dim, output_size=self.latent_dim))
    
    self.edge_type_W = nn.ModuleList()
    for _ in range(self.num_edge_types):
      ml_edge_type = nn.ModuleList()
      for _ in range(self.num_hops):
        ml_hop = nn.ModuleList()
        for _ in range(2):    # 2 directions of edges
          ml_hop.append(nn.Linear(latent_dim, latent_dim, bias=False))
        ml_edge_type.append(ml_hop)
      self.edge_type_W.append(ml_edge_type)
    
    self.const_nodes_free_params = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(self.num_ents, free_dim)))

    
  def gen_edge2node_mapping(self):
    ei = 0        # edge index with direction
    edge_idx = 0  # edge index without direction
    edge2node_in = torch.zeros(self.num_edges * 2, dtype=torch.long)
    edge2node_out = torch.zeros(self.num_edges * 2, dtype=torch.long)
    node_degree = torch.zeros(self.num_nodes)

    edge_type_masks = []
    for _ in range(self.num_edge_types):
      edge_type_masks.append(torch.zeros(self.num_edges * 2))
    edge_direction_masks = []
    for _ in range(2):    # 2 directions of edges
      edge_direction_masks.append(torch.zeros(self.num_edges * 2))
    
    for ni, nj in torch.as_tensor(self.graph.edge_pairs):
      edge_type = self.graph.edge_types[edge_idx]
      edge_idx += 1
      
      edge2node_in[ei] = nj
      edge2node_out[ei] = ni
      node_degree[ni] += 1
      edge_type_masks[edge_type][ei] = 1
      edge_direction_masks[0][ei] = 1
      ei += 1
      
      edge2node_in[ei] = ni
      edge2node_out[ei] = nj
      node_degree[nj] += 1
      edge_type_masks[edge_type][ei] = 1
      edge_direction_masks[1][ei] = 1
      ei += 1
    
    edge2node_in = edge2node_in.view(-1, 1).expand(-1, self.latent_dim)
    edge2node_out = edge2node_out.view(-1, 1).expand(-1, self.latent_dim)
    node_degree = node_degree.view(-1, 1)
    return edge2node_in, edge2node_out, node_degree, edge_type_masks, edge_direction_masks
  
  
  def forward(self, batch_data):
    """
        run gcn with knowledge graph and get embeddings for ground predicates (i.e. variables)

    :param batch_data:
        sampled data batch (a set of grounded formulas)
    :return:
        embeddings of all entities and relations
    """
    
    node_embeds = self.init_node_linear(self.node_feat)

    hop = 0
    hidden = node_embeds
    while hop < self.num_hops:
      node_aggregate = torch.zeros_like(hidden)
      for edge_type in set(self.graph.edge_types):
        for direction in range(2):
          W = self.edge_type_W[edge_type][hop][direction]
          W_nodes = W(hidden)
          nodes_attached_on_edges_out = torch.gather(W_nodes, 0, self.edge2node_out)
          nodes_attached_on_edges_out *= self.edge_type_masks[edge_type].view(-1, 1)
          nodes_attached_on_edges_out *= self.edge_direction_masks[direction].view(-1, 1)
          node_aggregate.scatter_add_(0, self.edge2node_in, nodes_attached_on_edges_out)

      hidden = self.MLPs[hop](hidden + node_aggregate)
      hop += 1

    read_out_const_nodes_embed = torch.cat((hidden[self.const_nodes], self.const_nodes_free_params), dim=1)

    return read_out_const_nodes_embed

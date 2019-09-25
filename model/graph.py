import networkx as nx
import numpy as np
from common.predicate import PRED_DICT
from itertools import product


class KnowledgeGraph(object):
  def __init__(self, facts, predicates, dataset):
    self.dataset = dataset
    self.graph, self.edge_type2idx, \
        self.ent2idx, self.idx2ent, self.rel2idx, self.idx2rel, \
        self.node2idx, self.idx2node = gen_graph(facts, predicates, dataset)
    
    self.num_ents = len(self.ent2idx)
    self.num_rels = len(self.rel2idx)
    
    self.num_nodes = len(self.graph.nodes())
    self.num_edges = len(self.graph.edges())
    
    x, y, v = zip(*sorted(self.graph.edges(data=True), key=lambda t: t[:2]))
    self.edge_types = [d['edge_type'] for d in v]
    self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.long)
    self.edge_pairs[:, 0] = x
    self.edge_pairs[:, 1] = y
    
    self.idx2edge = dict()
    idx = 0
    for x, y in self.edge_pairs:
      self.idx2edge[idx] = (self.idx2node[x], self.idx2node[y])
      idx += 1
      self.idx2edge[idx] = (self.idx2node[y], self.idx2node[x])
      idx += 1


def gen_index(facts, predicates, dataset):
  rel2idx = dict()
  idx_rel = 0
  for rel in sorted(predicates.keys()):
    if rel not in rel2idx:
      rel2idx[rel] = idx_rel
      idx_rel += 1
  idx2rel = dict(zip(rel2idx.values(), rel2idx.keys()))
  
  ent2idx = dict()
  idx_ent = 0
  for type_name in sorted(dataset.const_sort_dict.keys()):
    for const in dataset.const_sort_dict[type_name]:
      ent2idx[const] = idx_ent
      idx_ent += 1
  idx2ent = dict(zip(ent2idx.values(), ent2idx.keys()))
  
  node2idx = ent2idx.copy()
  idx_node = len(node2idx)
  for rel in sorted(facts.keys()):
    for fact in sorted(list(facts[rel])):
      val, args = fact
      if (rel, args) not in node2idx:
        node2idx[(rel, args)] = idx_node
        idx_node += 1
  idx2node = dict(zip(node2idx.values(), node2idx.keys()))
  
  return ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node


def gen_edge_type():
  edge_type2idx = dict()
  num_args_set = set()
  for rel in PRED_DICT:
    num_args = PRED_DICT[rel].num_args
    num_args_set.add(num_args)
  idx = 0
  for num_args in sorted(list(num_args_set)):
    for pos_code in product(['0', '1'], repeat=num_args):
      if '1' in pos_code:
        edge_type2idx[(0, ''.join(pos_code))] = idx
        idx += 1
        edge_type2idx[(1, ''.join(pos_code))] = idx
        idx += 1
  return edge_type2idx


def gen_graph(facts, predicates, dataset):
  """
      generate directed knowledge graph, where each edge is from subject to object
  :param facts:
      dictionary of facts
  :param predicates:
      dictionary of predicates
  :param dataset:
      dataset object
  :return:
      graph object, entity to index, index to entity, relation to index, index to relation
  """
  
  # build bipartite graph (constant nodes and hyper predicate nodes)
  g = nx.Graph()
  ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node = gen_index(facts, predicates, dataset)

  edge_type2idx = gen_edge_type()
  
  for node_idx in idx2node:
    g.add_node(node_idx)
  
  for rel in facts.keys():
    for fact in facts[rel]:
      val, args = fact
      fact_node_idx = node2idx[(rel, args)]
      for arg in args:
        pos_code = ''.join(['%d' % (arg == v) for v in args])
        g.add_edge(fact_node_idx, node2idx[arg],
                   edge_type=edge_type2idx[(val, pos_code)])
  return g, edge_type2idx, ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node

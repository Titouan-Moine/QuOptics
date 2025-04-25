import numpy as np
import matplotlib.pyplot as plt
from classes import *


def create_tenser(shape):
  return np.zeros(shape)
def add_tensers(tenser1, tenser2):
  return tenser1 + tenser2  
def multiply_tensers(tenser1, tenser2):
  return tenser1 * tenser2
def transpose_tenser(tenser):
  return tenser.T
def reshape_tenser(tenser, new_shape):
  return tenser.reshape(new_shape)

import networkx as nx
def create_graph():
  return nx.Graph()

def add_node(graph, node, data=None):
  graph.add_node(node, data=data)
def add_edge(graph, node1, node2, data=None):
  graph.add_edge(node1, node2, data=data)
def get_node_data(graph, node):
  return graph.nodes[node]['data']

def ket_0():
  ket_0 = create_tenser((2, 1))
  ket_0[0, 0] = 1
  ket_0[1, 0] = 0
  return ket_0
def ket_1():
  ket_1 = create_tenser((2, 1))
  ket_1[0, 0] = 0
  ket_1[1, 0] = 1
  return ket_1
def c_not():
  c_not = create_tenser((2, 2, 2, 2))
  c_not[0, 0, 0, 0] = 1
  c_not[0, 0, 1, 1] = 1
  c_not[1, 1, 1, 0] = 1
  c_not[1, 1, 0, 1] = 1
  
  return c_not
def hadamard():
  hadamard = create_tenser((2, 2))
  hadamard[0, 0] = 1 / np.sqrt(2)
  hadamard[1, 1] = -1 / np.sqrt(2)
  hadamard[0, 1] = 1 / np.sqrt(2)
  hadamard[1, 0] = 1 / np.sqrt(2)
  
  return hadamard
def exemple():
  g=create_graph()

  ket_0=ket_0()
  hadamard=hadamard()
  c_not=c_not()

  empty_tenser=create_tenser(())
  add_node(g, "empty_tenser1", empty_tenser)
  add_node(g, "empty_tenser2", empty_tenser)
  add_node(g, "hadamard", hadamard)
  add_node(g, "c_not", c_not)
  add_node(g, "Qbit_0", ket_0)
  add_node(g, "Qbit_1", ket_0)

  add_edge(g, "Qbit_0", "hadamard")
  add_edge(g, "hadamard", "c_not")
  add_edge(g, "Qbit_1", "c_not")
  add_edge(g, "c_not", "empty_tenser1")
  add_edge(g, "c_not", "empty_tenser2")
  node_colors=['red' if g.nodes[node]['data']==ket_0 else 'blue' if g.nodes[node]['data']==hadamard else 'green' if g.nodes[node]['data']==c_not else 'black' for node in g.nodes()]
  pos = nx.spring_layout(g)  # Positions for all nodes
  nx.draw(g, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=12, font_weight='bold')

def exemple2():
  g=create_graph()

  ket_0=Tensor.ket_0()
  hadamard=Tensor.hadamard()
  c_not=Tensor.c_not()
  empty_tenser=Tensor.empty_tenser()
  add_node(g, "output1", empty_tenser)
  add_node(g, "output2", empty_tenser)
  add_node(g, "hadamard", hadamard)
  add_node(g, "c_not", c_not)
  add_node(g, "Qbit_0", ket_0)
  add_node(g, "Qbit_1", ket_0)

  add_edge(g, "Qbit_0", "hadamard")
  add_edge(g, "hadamard", "c_not")
  add_edge(g, "Qbit_1", "c_not")
  add_edge(g, "c_not", "empty_tenser1")
  add_edge(g, "c_not", "empty_tenser2")
  node_colors=['red' if g.nodes[node]['data']==ket_0 else 'blue' if g.nodes[node]['data']==hadamard else 'green' if g.nodes[node]['data']==c_not else 'black' for node in g.nodes()]
  pos = nx.spring_layout(g)  # Positions for all nodes
  nx.draw(g, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=12, font_weight='bold')

def exemple3():
  c_not_content = [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
                      [[[0, 0], [0, 1]], [[0, 1], [0, 0]]]]
  c_not_content[0][0][0][0] = -1
  c_not_content[0][0][1][1] = -1
  # c_not_content[1,1,1,0] = -1
  # c_not_content[1,1,0,1] = -1
  print(c_not_content)
#exemple()
exemple2()
plt.show()

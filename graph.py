import tensor
import networkx as nx
import numpy as np

def create_graph():
  return nx.Graph()

def add_node(graph, node):
  graph.add_node(node)
  return graph
def add_edge(graph, node1, node2):
  graph.add_edge(node1, node2)
  return graph
def remove_node(graph, node):
  graph.remove_node(node)
  return graph
def remove_edge(graph, node1, node2):
  graph.remove_edge(node1, node2)
  return graph
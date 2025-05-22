from qiskit.quantum_info import Operator
from classe import *
from qiskit import QuantumCircuit
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def instruction_to_tensor(instr):
    """
    Convertit une instruction Qiskit en un objet Tensor.
    """
    name = instr.operation.name
    try:
        mat = Operator(instr.operation).data  # numpy array
        dim = mat.shape
        return Tensor(dimension=dim, content=mat, name=name)
    except Exception as e:
        print(f"Opération '{name}' ignorée (non-unitaire ou autre) : {e}")
        return None
    
def circuit_to_tensor_graph(circuit):
    G = nx.DiGraph()
    last_node_on_qubit = {}

    for i in range(circuit.num_qubits):
        qubit_name = f'init_{i}'
        G.add_node(qubit_name)
        last_node_on_qubit[i] = qubit_name

    for instr in circuit.data:
        name = instr.operation.name
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        gate_id = f'{name}_' + '_'.join(f'{q}' for q in qubits)

        # Convertir en Tensor
        """tensor = instruction_to_tensor(instr)
        if tensor:
            G.add_edge(gate_id, tensor)"""

        if gate_id not in G.nodes:
            G.add_node(gate_id)

        for q in qubits:
            prev = last_node_on_qubit[q]
            G.add_edge(prev,gate_id)
            last_node_on_qubit[q] = gate_id

    for i in range(circuit.num_qubits):
        end_node = f'fin_{i}'
        G.add_node(end_node)
        G.add_edge(last_node_on_qubit[i],end_node)

    return G

def build_qft(n):
   circuit = QuantumCircuit(n)
   for j in reversed(range(n)):
       circuit.h(j)
       for k in reversed(range(j)):
           circuit.cp(np.pi * 2. ** (k - j), j, k)

   for j in range(n // 2):
       circuit.swap(j, n - j - 1)

   return circuit

nqbit =2
c = build_qft(nqbit)
graph = circuit_to_tensor_graph(c)

def positi(G, nqbit):
    n = len(G)  # nb de point
    j = 0
    pos = {}
    for i in range(nqbit):
        print(i)
        pos[i] = (0, i)
        pos[n-i] = (n-2*nqbit +1, nqbit-i-1)
    for i in range(nqbit, n-nqbit):
        print(i-nqbit+1)
        pos[i] = (i-nqbit+1, j)
        if j == nqbit:
            j=0
        else :
            j+=1
    return pos


def afficher_graphe_with_networkx(G, n):
    #pos = nx.spring_layout(G) 
    pos = positi(G, n)
    # Afficher les noeuds
    nx.draw_networkx_nodes(G, pos,node_size=700, node_color='lightgreen')
    
    # Afficher les arêtes
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.6, edge_color='gray')

    # Afficher les étiquettes des noeuds (portes)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    plt.title("Graphe du Circuit Quantique")
    plt.axis("off")  # Désactive les axes
    plt.show()

afficher_graphe_with_networkx(graph, nqbit)
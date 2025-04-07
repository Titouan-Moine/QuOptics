from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli
import math

def circuit2():
    c = QuantumCircuit(2)

    c.h(0)
    c.cx(0,1)
    c.z(1)
    print(c)

from qiskit.circuit.random import random_circuit
 
c = random_circuit(4, 4)
print(c)

####------graphe

def circuit_to_graph(circuit):
    graph = {}
    last_node_on_qubit = {}

    # Initialiser les noeuds de départ pour chaque qubit
    for i in range(circuit.num_qubits):
        qubit_name = f'init_{i}'
        graph[qubit_name] = []
        last_node_on_qubit[i] = qubit_name

    # Parcourir chaque porte
    for instr in circuit.data:
        name = instr.operation.name
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        gate_id = f'{name}_' + '_'.join(f'{q}' for q in qubits)

        # Créer le noeud si pas encore là
        if gate_id not in graph:
            graph[gate_id] = []

        # Connecter les derniers noeuds de chaque qubit à la nouvelle porte
        for q in qubits:
            prev = last_node_on_qubit[q]
            graph[prev].append(gate_id)
            last_node_on_qubit[q] = gate_id

    # Finir le graph pour chaque qubit

    for i in range(circuit.num_qubits):
        
        q_sortie = f'fin_{i}'
        graph[q_sortie] = []  # aucun successeur
        graph[last_node_on_qubit[i]].append(q_sortie)
    return graph

adj_list = circuit_to_graph(c)
print(adj_list)



def calculer_positions_grille(nodes):
    n = len(nodes)
    cols = math.ceil(math.sqrt(n))-1  # nb de colonnes
    pos = {}
    for i, node in enumerate(nodes):
        print(i)
        y = -float(node.split('_')[1])
        x = i
        pos[node] = (x, y)
    return pos


def afficher_graphe_depuis_adjacence(adjacence):
    nodes = list(adjacence.keys())
    for succs in adjacence.values():
        for node in succs:
            if node not in nodes:
                nodes.append(node)

    # 💡 Générer une grille de positions 2D
    pos = calculer_positions_grille(nodes)
    print(pos)
    fig, ax = plt.subplots()
    for node, (x, y) in pos.items():
        ax.plot(x, y, 'o', markersize=10, color='lightgreen')
        ax.text(x+0.1, y+0.1, node, ha='center', fontsize=9)

    for src, dsts in adjacence.items():
        for dst in dsts:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            ax.annotate("",
                        xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color='gray'))

    ax.set_axis_off()
    plt.title("Graphe du circuit quantique")
    plt.tight_layout()
    plt.show()


afficher_graphe_depuis_adjacence(adj_list)
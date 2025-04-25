import numpy as np
import qiskit as qk 
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator
import tkinter as tk
import random
import numpy as np
from time import sleep

############## Fonctions de construction de tenseurs ###########################
##############################################################################
def swap_tensor():
    swap = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    
    for i1 in range(2):
        for i2 in range(2):
            swap[i1, i2, i2, i1] = 1
    
    return swap

def cnot_tensor():
    cnot = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    
    for i1 in range(2):
        for i2 in range(2):
            cnot[i1, i2, i1, i1^i2] = 1
    
    return cnot

def hadamard_tensor():
    hadamard = np.zeros((2, 2), dtype=np.complex128)
    
    for i in range(2):
        for j in range(2):
            hadamard[i, j] = 1/np.sqrt(2)
    hadamard[1,1]=-1/np.sqrt(2)
    
    return hadamard

swap_tensor=swap_tensor()
cnot_tensor=cnot_tensor()
hadamard_tensor=hadamard_tensor()

print(cnot_tensor)
tensors={'swap': swap_tensor, 'cx': cnot_tensor, 'h': hadamard_tensor}

######### Circuit d'exemple ###########################################
#######################################################################

def build_qft(n, n_gates=5):
   circuit = qk.QuantumCircuit(n)
#    for j in reversed(range(n)):
#        circuit.h(j)
#        for k in reversed(range(j)):
#            circuit.cp(np.pi * 2. ** (k - j), j, k)

#    for j in range(n // 2):
#        circuit.swap(j, n - j - 1)
   for i in range(n_gates):
       q1, q2 = random.sample(range(n), 2)
       circuit.cx(q1, q2)
   return circuit
def build_simple_qft():
   circuit = qk.QuantumCircuit(2)
#    for j in reversed(range(n)):
#        circuit.h(j)
#        for k in reversed(range(j)):
#            circuit.cp(np.pi * 2. ** (k - j), j, k)

#    for j in range(n // 2):
#        circuit.swap(j, n - j - 1)
   circuit.swap(0, 1)
   circuit.swap(0, 1)
   return circuit
#qft = build_qft(7)


#print(operations)

#print(qft)


#######################################################################
#######################################################################


def instruction_list(circuit):
    operations = circuit.data
    #print(operations)
    liste_instructions = []
    for instruction in operations:
        operation = instruction.operation  # Access the operation part of the CircuitInstruction
        gate_name = operation.name  # Get the gate name
        params = operation.params  # Get the parameters for the operation (if any)
        qubits = [q._index for q in instruction.qubits]
            
        liste_instructions.append( (gate_name, params, qubits))
    return(liste_instructions)


#print(instruction_list(qft))
#instruction_list(qft)


############### Circuit Qiskit aléatoire################################## 
from qiskit.circuit.random import random_circuit
 
circ = random_circuit(5, 5, measure=False, max_operands=2)
circ.draw(output='mpl')
###########################################################################

class TensorNode:
    def __init__(self, name, qubits, params=None):
        self.name = name  # Nom de l'opération (ex: 'H', 'CNOT', etc.)
        self.nb_qubits = len(qubits)  # Nombre de qubits impliqués dans l'opération
        self.qubits = qubits  # Liste des qubits impliqués dans l'opération
        self.params = params if params is not None else []  # Paramètres de l'opération (s'il y en a)
        self.tensor = tensors.get(name)  # Tenseur associé à l'opération (initialisé à None)
        self.neighbors = [None]*2*self.nb_qubits
        self.neighbors_ind = [None]*2*self.nb_qubits
        self.wires = {}
        self.x = 0
        self.y = 0
        for i in range(self.nb_qubits):
            self.wires[qubits[i]] = i
    def set_neighbor(self, neighbor, i):
        self.neighbors[i] = neighbor
    def __str__(self):
        s=f"TensorNode(name={self.name}, qubits={self.qubits}, params={self.params})"
        for i in range(len(self.neighbors)):
            s+=f"\n\tNeighbor {i}: {self.neighbors[i].name + str(self.neighbors[i].params) if self.neighbors[i] else None}"
        return s

class SingleTensorNode(TensorNode):
    def __init__(self, name, qubit, params=[]):
        super().__init__(name, [qubit], params)
        self.neighbors = [None]
        self.neighbors_ind = [None]

def from_instruction_to_graph(instruction_list, n_lanes):
    nodes = [SingleTensorNode('start', i, params=[i]) for i in range(n_lanes)]
    
    latest = [nodes[i] for i in range(n_lanes)]
    
    def add_to_graph(node): 
        for i in range(len(node.qubits)):
            k=node.qubits[i]
            if k < len(latest):
                latest[k].set_neighbor(node, latest[k].wires[k] + (latest[k].nb_qubits if latest[k].nb_qubits>1 else 0))
                latest[k].neighbors_ind[latest[k].wires[k] + (latest[k].nb_qubits if latest[k].nb_qubits>1 else 0)]= node.wires[k]
                node.set_neighbor(latest[k], node.wires[k])
                node.neighbors_ind[node.wires[k]] = latest[k].wires[k] + (latest[k].nb_qubits if latest[k].nb_qubits>1 else 0)
        for i in range(len(node.qubits)):
            k=node.qubits[i]
            if k < len(latest):
                latest[k] = node
        nodes.append(node)
    
    for i in range(len(instruction_list)):
        instruction = instruction_list[i]
        name, params, qubits = instruction
        node = TensorNode(name, qubits, params=[i])
        add_to_graph(node)
    
    for i in range(n_lanes):
        add_to_graph(SingleTensorNode('end', i, params=[i]))
    print("before")
    for i in nodes:
        print(i)
    print("end before")
    return nodes

def print_nodes(nodes):
    print("start")
    for node in nodes:
        print(node)
    print("end")
############## Contraction de tenseurs ###########################
##################################################################

def contract_tensors(tensorNode1, tensorNode2):
    common_indices = set()
    taken_indices1 = set()
    taken_indices2 = set()
    for i in range(len(tensorNode1.neighbors)):
        if tensorNode1.neighbors[i] == tensorNode2:
            common_indices.add((i, tensorNode1.neighbors_ind[i]))
            taken_indices1.add(i)
            taken_indices2.add(tensorNode1.neighbors_ind[i])
    
    for ind in common_indices:
        if tensorNode1.tensor.shape[ind[0]] != tensorNode2.tensor.shape[ind[1]]:
            raise ValueError("Mauvaise taille mon reuf t'as fait de la D")
    indices1 = []
    for i in range(len(tensorNode1.neighbors)):
        if i not in taken_indices1:
            indices1.append(i)
    indices2 = []
    for i in range(len(tensorNode2.neighbors)):
        if i not in taken_indices2:
            indices2.append(i)

    dim = []
    for i in indices1:
        dim.append(tensorNode1.tensor.shape[i])
    for i in indices2:
        dim.append(tensorNode2.tensor.shape[i])
    
    tensor3 = np.zeros(dim, dtype=np.complex128)
    
    def summ_commons(parc):
        s=0 
        parc_int=[]
        common_list=list(common_indices)
        def browse_int(i):
            nonlocal parc_int, parc, s
            if i==len(common_indices):
                ind_for_1=[None]*len(tensorNode1.neighbors)
                ind_for_2=[None]*len(tensorNode2.neighbors)
                for j in range(len(indices1)):
                    ind_for_1[indices1[j]] = parc[j]
                for j in range(len(indices2)):
                    ind_for_2[indices2[j]] = parc[j+len(indices1)]
                for j in range(len(common_indices)):
                    ind_for_1[common_list[j][0]] = parc_int[j]
                    ind_for_2[common_list[j][1]] = parc_int[j]
                s+=tensorNode1.tensor[tuple(ind_for_1)]*tensorNode2.tensor[tuple(ind_for_2)]
                return
                
            if i<len(common_indices):
                for j in range(tensorNode1.tensor.shape[common_list[i][0]]):
                    parc_int.append(j)
                    browse_int(i+1)
                    parc_int.pop()
        browse_int(0)     
        
        return s
        
        
    parc=[]
    def browse_inds(i):
        nonlocal parc
        if i==len(indices1)+len(indices2):
            tensor3[tuple(parc)] = summ_commons(tuple(parc))
            return
        if i<len(indices1):
            for j in range(tensorNode1.tensor.shape[indices1[i]]):
                parc.append(j)
                browse_inds(i+1)
                parc.pop()
        else:
            for j in range(tensorNode2.tensor.shape[indices2[i-len(indices1)]]):
                parc.append(j)
                browse_inds(i+1)
                parc.pop()
    browse_inds(0)        
    return tensor3, common_indices, indices1, indices2

def contract_nodes(tensorNode1, tensorNode2):
    tensor3, common_indices, indices1, indices2 = contract_tensors(tensorNode1, tensorNode2)
    qubits3 = tensorNode1.qubits.copy()
    for i in range(len(tensorNode2.qubits)):
        if tensorNode2.qubits[i] not in qubits3:
            qubits3.append(tensorNode2.qubits[i])
    neighbors3 = [None]*(len(tensorNode1.neighbors) + len(tensorNode2.neighbors) - 2*len(common_indices))
    neighbors3_ind = [None]*(len(tensorNode1.neighbors) + len(tensorNode2.neighbors) - 2*len(common_indices))
    tensorNode3 = TensorNode("Contraction:"+tensorNode1.name + tensorNode2.name, qubits3)
    for i in range(len(indices1)):
        neighbors3[i] = tensorNode1.neighbors[indices1[i]]
        # print(tensorNode1.neighbors[i])
        # for j in tensorNode1.neighbors[i].neighbors:
        #     print(j)
        tensorNode1.neighbors[indices1[i]].set_neighbor(tensorNode3, tensorNode1.neighbors_ind[indices1[i]])
        neighbors3_ind[i] = tensorNode1.neighbors_ind[indices1[i]]
        tensorNode1.neighbors[indices1[i]].neighbors_ind[tensorNode1.neighbors_ind[indices1[i]]] = i
    for i in range(len(indices2)):
        neighbors3[i+len(indices1)] = tensorNode2.neighbors[indices2[i]]
        tensorNode2.neighbors[indices2[i]].set_neighbor(tensorNode3, tensorNode2.neighbors_ind[indices2[i]])
        neighbors3_ind[i+len(indices1)] = tensorNode2.neighbors_ind[indices2[i]]
        tensorNode2.neighbors[indices2[i]].neighbors_ind[tensorNode2.neighbors_ind[indices2[i]]] = i+len(indices1)

    tensorNode3.neighbors = neighbors3
    tensorNode3.tensor = tensor3
    tensorNode3.neighbors_ind = neighbors3_ind
    print("contracted result : ", tensorNode3)
    print("tensor : ", tensor3)
    return tensorNode3
    
def display_graph_tkinter(nodesOrigin, n_lanes):
    nodes = nodesOrigin.copy()
    # Create a Tkinter window
    
    # Create a canvas to draw the graph
    canvas_width = 1200
    canvas_height = 800
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()
    
    def merge():
        nonlocal nodes, canvas, n_lanes
        canvas.delete("all")
        visited = [False] * len(nodes)
        reached_fusion_point=False
        def dfs(node):
            nonlocal reached_fusion_point, visited, nodes
            if reached_fusion_point:
                return
            visited[nodes.index(node)] = True
            for neighbor in node.neighbors:
                if neighbor is not None and not visited[nodes.index(neighbor)]:
                    if len(node.neighbors)>1 and len(neighbor.neighbors)>1:
                        reached_fusion_point=True
                        node3 = contract_nodes(node, neighbor)
                        nodes.remove(node)
                        nodes.remove(neighbor)
                        nodes = nodes[:n_lanes] + [node3] + nodes[n_lanes:]
                        return
                    dfs(neighbor)
        for i in range(len(nodes)):
            if not visited[i]:
                dfs(nodes[i])
                if reached_fusion_point:
                    break
        # for i in range(len(nodes)):
        #     if len(nodes[i].neighbors)>1 and len(nodes[i+1].neighbors)>1 and nodes[i] in nodes[i+1].neighbors:
        #         CTensorNode = contract_nodes(nodes[i], nodes[i+1])
        #         nodes=nodes[:i]+[CTensorNode]+nodes[i+2:]
        #         break
        # print_nodes(nodes)
        draw_graph()
    def draw_graph():
        nonlocal canvas
    # Calculate spacing
        node_spacing_x = canvas_width // (len(nodes) + 3 - n_lanes*2)
        node_spacing_y = canvas_height // (n_lanes + 1)

        #ChatGPT est sous frozen ça clc
        k=0
        def draw_node(node, x, y):
            nonlocal k
            color = "blue" if node.name != 'start' else "red" if node.name == 'end' else "green"
            canvas.create_oval(x-25, y-25, x + 25, y + 25, fill=color)
            canvas.create_text(x, y, text=node.name, fill="white")
            node.x = x
            node.y = y
            k += 1
        def draw_gate(node):
            nonlocal k
            x = node_spacing_x * (k - n_lanes +2)
            y = node_spacing_y * ((max(node.qubits) + min(node.qubits))/2 +1)
            y1 =node_spacing_y * (max(node.qubits)+1)
            y2 = node_spacing_y * (min(node.qubits)+1)
            color = "green" if node.name == 'start' else "red" if node.name == 'end' else "pink"
            # canvas.create_rectangle(x-2, y1, x+2, y2, fill="black")
            canvas.create_rectangle(x-25, y-25, x + 25, y + 25, fill=color)
            canvas.create_text(x, y, text=node.name, fill="black")
            node.x = x
            node.y = y
            k += 1
        for i in range(n_lanes):
            draw_node(nodes[i], node_spacing_x, node_spacing_y * (i + 1))
        for i in range(len(nodes)-2*k):
            draw_gate(nodes[k])
        k_f=k-n_lanes + 2
        for i in range(n_lanes):
            draw_node(nodes[k], node_spacing_x*k_f, node_spacing_y * (i + 1))
            
        for node in nodes:
            color = random.choice(["red", "blue", "green", "orange", "purple", "brown", "light blue"])
            for neighbor in node.neighbors:
                if neighbor is not None:
                    x1, y1 = node.x, node.y
                    x2, y2 = neighbor.x, neighbor.y
                    canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
    button = tk.Button(window, text="Merge", command=merge)
    button.pack()
    draw_graph()
    # Run the Tkinter event loop
    

# Example usage
qft = build_qft(3, 3)
print(qft)
print(qft.data, len(qft.data))

instruction_list_example = instruction_list(qft)
graph_nodes = from_instruction_to_graph(instruction_list_example, n_lanes=3)
print(len(graph_nodes))
window = tk.Tk()
window.title("Graph Visualization")
display_graph_tkinter(graph_nodes, n_lanes=3)
window.mainloop()

# t1=TensorNode('swap', [0, 1], [0, 1])
# t2=TensorNode('swap', [1, 0], [0, 1])

# t1.set_neighbor(t2, 2)
# t1.neighbors_ind[2]=0
# t1.set_neighbor(t2, 3)
# t1.neighbors_ind[3]=1
# t2.set_neighbor(t1, 0)
# t2.neighbors_ind[0]=2
# t2.set_neighbor(t1, 1)
# t2.neighbors_ind[1]=3

# t3=contract_tensors(t1, t2)
# print(t3.shape)
# print(t3)
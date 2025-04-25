from qiskit.quantum_info import Operator
from classe import *
from qiskit import QuantumCircuit
import math
import numpy as np

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
    graph = {}
    last_node_on_qubit = {}
    tensor_nodes = {}

    for i in range(circuit.num_qubits):
        qubit_name = f'init_{i}'
        graph[qubit_name] = []
        last_node_on_qubit[i] = qubit_name

    for instr in circuit.data:
        name = instr.operation.name
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        gate_id = f'{name}_' + '_'.join(f'{q}' for q in qubits)

        # Convertir en Tensor
        tensor = instruction_to_tensor(instr)
        if tensor:
            tensor_nodes[gate_id] = tensor

        if gate_id not in graph:
            graph[gate_id] = []

        for q in qubits:
            prev = last_node_on_qubit[q]
            graph[prev].append(gate_id)
            last_node_on_qubit[q] = gate_id

    for i in range(circuit.num_qubits):
        end_node = f'fin_{i}'
        graph[end_node] = []
        graph[last_node_on_qubit[i]].append(end_node)

    return graph, tensor_nodes

def build_qft(n):
   circuit = QuantumCircuit(n)
   for j in reversed(range(n)):
       circuit.h(j)
       for k in reversed(range(j)):
           circuit.cp(np.pi * 2. ** (k - j), j, k)

   for j in range(n // 2):
       circuit.swap(j, n - j - 1)

   return circuit

c = build_qft(2)
graph, tensor_nodes = circuit_to_tensor_graph(c)

for node, tensor in tensor_nodes.items():
    print(f"{node}: {tensor}")
    print(tensor.get_content())



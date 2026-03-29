"""Tensor network module.

"""
import sys
import os
import warnings
from typing import Optional
import math
import copy
# import numpy as np
import sparse
from sparse_backend import sparse_tensordot_via_scipy
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from fock_amplitude import fock_tensor_bs, fock_tensor_ps

class TensorGate:
    """A gate in a tensor network, representing a tensor and its connections to other tensors.

    Attributes:
    tensor (sparse.COO): The tensor associated with this gate.
    inmodes (list[tuple[int]]): The list of input modes (or wires) that this gate connects to.
        Each mode is represented as a tuple of (mode_index, counter), where
        counter corresponds to the number of times this mode has been used
        in previous gates, to ensure unique labeling. The order of modes in
        this list corresponds to the order of the tensor's dimensions.
    outmodes (list[tuple[int]]): The list of output modes (or wires) that this gate connects to.
        Each mode is represented as a tuple of (mode_index, counter), similar to inmodes.
    axis_map (Optional[dict[tuple[int, int], int]]): An optional mapping from mode tuples to
        tensor axes. This can be used to specify the order of modes in the tensor explicitly.
        If not provided, a default mapping will be created based on the order of inmodes and
        outmodes.
    name (Optional[str]): A unique identifier for the gate.
    tags (Optional[set[str]]): A set of tags for categorizing or annotating the gate.
    """
    def __init__(self,
                 tensor: sparse.COO,
                 inmodes: list[tuple[int]],
                 outmodes: list[tuple[int]],
                 axis_map: Optional[dict[tuple[int, int], int]]=None,
                 name: Optional[str]=None,
                 tags: Optional[set[str]]=None,
                 warn: bool=False):
        self.name = name if name is not None else f"Gate_{id(self)}"
        self.tensor = tensor
        self.inmodes = inmodes
        self.outmodes = outmodes
        if axis_map is None:
            if warn:
                warnings.warn("No axis_map provided. Creating a default one.", UserWarning)
            axis_map = {}
            for i, mode in enumerate(inmodes + outmodes):
                axis_map[mode] = i
        self.axis_map = axis_map
        self.tags = tags if tags is not None else set()

    def inmode_to_axis(self, inmode: int) -> int:
        """Convert an inmode to the corresponding axis in the tensor.

        Args:
            inmode (int): The inmode to convert.

        Returns:
            int: The corresponding axis in the tensor.
        """
        inmode_indices = [e[0] for e in self.inmodes]
        return inmode_indices.index(inmode)

    def outmode_to_axis(self, outmode: int) -> int:
        """Convert an outmode to the corresponding axis in the tensor.

        Args:
            outmode (int): The outmode to convert.

        Returns:
            int: The corresponding axis in the tensor.
        """
        outmode_indices = [e[0] for e in self.outmodes]
        return outmode_indices.index(outmode)

    def contract(self,
                 other: 'TensorGate',
                 contract_modes: Optional[list[tuple[int]]]=None,
                 new_name: Optional[str]=None,
                 new_tags: Optional[set[str]]=None
                 ) -> 'TensorGate':
        """Contract this gate with another gate along specified modes.
        
        Args:
            other (TensorGate): The other gate to contract with.
            contract_modes (Optional[list[tuple[int]]]): The modes to contract over. If None, will
                automatically contract over all common modes between this gate's outmodes and the
                other gate's inmodes.
            new_name (Optional[str]): An optional name for the resulting contracted gate.
            new_tags (Optional[set[str]]): An optional set of tags for the resulting contracted gate.

        Returns:
            TensorGate: The resulting contracted gate.
        """
        if contract_modes is None:
            contract_modes = list(set(self.outmodes) & set(other.inmodes))
            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify contract_modes explicitly.")
        
        axes_a = [self.axis_map[mode] for mode in contract_modes]
        axes_b = [other.axis_map[mode] for mode in contract_modes]
        if not axes_a or not axes_b:
            raise ValueError("Incompatible contraction modes. Check axis mapping")
        
        new_axis_map = {mode: i for i, mode in enumerate(self.inmodes +self.outmodes) if mode not in contract_modes}
        new_axis_map.update({mode: i+len(new_axis_map) for i, mode in enumerate(other.inmodes + other.outmodes) if mode not in contract_modes})
        new_inmodes = [mode for mode in self.inmodes if mode not in contract_modes] + \
                      [mode for mode in other.inmodes if mode not in contract_modes]
        new_outmodes = [mode for mode in self.outmodes if mode not in contract_modes] + \
                       [mode for mode in other.outmodes if mode not in contract_modes]

        # Perform the contraction using the specified modes
        result_tensor = sparse_tensordot_via_scipy(self.tensor,
                                                   other.tensor,
                                                   axes_a=axes_a,
                                                   axes_b=axes_b)

        # Create a new gate for the result
        result_gate = TensorGate(tensor=result_tensor,
                                 inmodes=new_inmodes,
                                 outmodes=new_outmodes,
                                 axis_map=new_axis_map,
                                 name=new_name if new_name is not None else None,
                                 tags=new_tags if new_tags is not None else self.tags.union(other.tags))

        return result_gate

    def __repr__(self):
        return f"TensorGate(name={self.name}, tensor={self.tensor},\
            inmodes={self.inmodes}, outmodes={self.outmodes}, tags={self.tags})"

class TensorNetworkCircuit:
    """A tensor network circuit, consisting of multiple TensorGates and their connections.

    Attributes:
        gates (dict[str, TensorGate]): A dictionary mapping gate names to TensorGate objects.
        length (int): The number of gates in the network.
        n_modes (int): The total number of modes in the network.
        n_photons (int): The total number of photons in the network.
        gate_graph (dict[str, set[str]]): A directed graph representation of the gates and
            their connections, where each key is a gate name and the value is a set of gate
            names that are directly connected to it (i.e., share at least one mode).
        name (Optional[str]): An optional name for the tensor network circuit.
    """
    def __init__(self, n_modes: int, n_photons: int, gates: Optional[list[TensorGate]]=None, name: Optional[str]=None):
        if gates is not None and len(gates) > 0:
            warnings.warn("Initializing with a non-empty list of gates may lead \
                to inconsistent mode labeling. Please ensure that the gates are \
                labeled correctly or initialize with an empty list and append \
                gates one by one.", UserWarning)
        self.gates = {gate.name: gate for gate in (gates if gates is not None else [])}
        self.length = len(self.gates)
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.name = name if name is not None else "unnamed_circuit"
        self.gate_graph = {}
        if self.gates:
            for gate in self.gates.values():
                for gate2 in self.gates.values():
                    if gate != gate2 and set(gate.outmodes) & set(gate2.inmodes):
                        if gate.name not in self.gate_graph:
                            self.gate_graph[gate.name] = set()
                        self.gate_graph[gate.name].add(gate2.name)

        mode_counters = {i: 0 for i in range(n_modes)}
        for gate in self.gates.values():
            for outmode in gate.outmodes:
                mode_counters[outmode[0]] = max(mode_counters[outmode[0]], outmode[1])
        self._current_mode_counters = mode_counters
    
    def contract(self,
                 gate1_name: str,
                 gate2_name: str,
                 contract_modes: Optional[list[tuple[int]]]=None,
                 new_gate_name: Optional[str]=None,
                 new_gate_tags: Optional[set[str]]=None
                 ):
        """Contract two gates in the network along specified modes.

        Args:
            gate1_name (str): The name of the first gate.
            gate2_name (str): The name of the second gate.
            contract_modes (Optional[list[tuple[int]]]): The modes to contract over.

        Returns:
            TensorGate: The resulting contracted gate.
        """
        gate1 = self.gates[gate1_name]
        gate2 = self.gates[gate2_name]

        if gate1 is None:
            raise ValueError(f"gate1 must be part of the network, {gate1_name} not found.")
        if gate2 is None:
            raise ValueError(f"gate2 must be part of the network, {gate2_name} not found.")

        if contract_modes is None:
            contract_modes = list(set(gate1.outmodes) & set(gate2.inmodes))
            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify \
                    contract_modes explicitly.")
        if not set(contract_modes).issubset(set(gate1.outmodes) & set(gate2.inmodes)):
            raise ValueError("Contract modes must be a subset of the common modes \
                between the two gates.")

        result_gate = gate1.contract(gate2, contract_modes=contract_modes)
        if new_gate_name is not None:
            result_gate.name = new_gate_name
        if new_gate_tags is not None:
            result_gate.tags.update(new_gate_tags)

        self.gates.pop(gate1_name)
        self.gates.pop(gate2_name)
        self.gates[result_gate.name] = result_gate
        self.length -= 1
        # Update the gate graph to reflect the contraction
        following_gates = self.gate_graph.get(gate1_name, set()) | self.gate_graph.get(gate2_name, set())
        following_gates.discard(gate1_name)
        following_gates.discard(gate2_name)
        self.gate_graph[result_gate.name] = following_gates
        for neighbors in self.gate_graph.values(): # iterate over all preceding gates' neighbors
            if gate1_name in neighbors or gate2_name in neighbors:
                neighbors.discard(gate1_name)
                neighbors.discard(gate2_name)
                neighbors.add(result_gate.name)
        self.gate_graph.pop(gate1_name, None)
        self.gate_graph.pop(gate2_name, None)

    def append(self,
               data: TensorGate | sparse.COO,
               target: Optional[list[int] | tuple[int]]=None,
               name: Optional[str]=None,
               tags: Optional[set[str]]=None):
        """Append a TensorGate or a sparse.COO tensor to the network.

        Args:
            data: TensorGate | sparse.COOThe tensor to append.
            target (Optional[list[int]], optional): The target modes to append to. Defaults to None.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.

        Raises:
            ValueError: If the input modes of the gate are not a subset of the current mode labels.
            ValueError: If the target mode is not specified when appending a raw tensor.
            ValueError: If the input modes of the gate are not compatible with the current network
                structure.
            ValueError: If the target modes are out of bounds for the number of modes in the network
                when appending a raw tensor.
            ValueError: If the tensor dimensions do not match the target modes when appending a raw
                tensor.
        """

        if isinstance(data, TensorGate):
            warnings.warn("Appending a TensorGate directly may lead to inconsistent mode labeling. \
                Please ensure that the gate's input modes are a subset of the current mode labels \
                in the network, and that the output modes are labeled correctly. It is recommended \
                to append raw tensors and let the network handle mode labeling automatically.",
                UserWarning)
            if not set(data.inmodes).issubset(set((i, self._current_mode_counters[i])
                                                for i in range(self.n_modes))):
                raise ValueError("The input modes of the gate must be a subset of the \
                    current mode labels in the network.")
            gate = data
            for outmode in gate.outmodes:
                self._current_mode_counters[outmode[0]] = outmode[1]

        elif isinstance(data, sparse.COO):
            if target is None:
                raise ValueError("Must provide target when appending a raw tensor.")
            for t in target:
                if t < 0 or t >= self.n_modes:
                    raise ValueError(f"Target mode {t} is out of bounds for a network with {self.n_modes} modes.")
            if data.ndim != 2*len(target):
                raise ValueError(f"Tensor has {data.ndim} dimensions but target has {len(target)} \
                    modes. Dimensions of the tensor must be twice the number of target modes.")
            inmodes = [(i, self._current_mode_counters[i]) for i in target]
            outmodes = [(i, self._current_mode_counters[i]+1) for i in target]
            for t in target:
                self._current_mode_counters[t] += 1
            gate = TensorGate(tensor=data, inmodes=inmodes, outmodes=outmodes,
                              name=name, tags=tags)
        
        else:
            raise ValueError("Can only append a TensorGate or a sparse.COO tensor.")

        self.gates[gate.name] = gate
        self.length += 1
        self.gate_graph[gate.name] = set()
        for other_gate in self.gates.values():
            if other_gate != gate:
                if set(gate.outmodes) & set(other_gate.inmodes):
                    self.gate_graph[gate.name].add(other_gate.name)
                if set(other_gate.outmodes) & set(gate.inmodes):
                    self.gate_graph[other_gate.name].add(gate.name)

    def append_bs(self,
                  target : tuple[int],
                  angles : tuple[float],
                  name: Optional[str]=None,
                  tags: Optional[set[str]]=None):
        """Append a parameterized beam splitter gate to the network.

        Args:
            target (tuple[int]): The target modes for the gate.
            angles (tuple[float]): The angles for the beam splitter in the format (phi, theta).
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        
        if len(target) != 2:
            raise ValueError("Target must be a tuple of two modes for a parameterized \
                beam splitter gate.")
        if len(angles) != 2:
            raise ValueError("Angles must be a tuple of two angles (phi, theta) for a \
                parameterized beam splitter gate.")
        if not angles or len(angles) != len(target):
            raise ValueError("Angles must be a tuple of the same length as target modes.")

        bs_tensor = fock_tensor_bs(angles[0], angles[1], self.n_photons,
                                   sparse_tensor=True, check=False)
        self.append(data=bs_tensor, target=target, name=name, tags=tags)
    
    def append_ps(self,
                  target : int,
                  angle : float,
                  name: Optional[str]=None,
                  tags: Optional[set[str]]=None):
        """Append a parameterized phase shifter gate to the network.

        Args:
            target (int): The target mode for the gate.
            angle (float): The angle for the phase shifter.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        if target < 0 or target >= self.n_modes:
            raise ValueError(f"Target mode {target} is out of bounds for a network with \
                {self.n_modes} modes.")
        ps_tensor = fock_tensor_ps(angle, self.n_photons, sparse_tensor=True, check=False)
        self.append(data=ps_tensor, target=(target,), name=name, tags=tags)

    def contract_all(self, method: Optional[str]="greedy") -> sparse.COO:
        """Contract all gates in the network to obtain the final output tensor.

        Args:
            method (str): The contraction method to use. Currently only supports 'naive'
            and 'greedy'.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if method.lower() == 'naive':
            return self._contract_all_naive()
        elif method.lower() == 'greedy':
            return self._contract_all_greedy()
        else:
            raise ValueError(f"Unsupported contraction method: {method}. Supported \
                methods are 'naive' and 'greedy'.")

    def _contract_all_naive(self) -> sparse.COO:
        """Contract all gates in the network using a naive left-to-right approach.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")

        while self.length > 1:
            gate1_name = next(iter(self.gates.keys()))  # get an arbitrary gate name
            gate2_name = next(iter(self.gate_graph.get(gate1_name, set())))  # get an arbitrary neighbor
            self.contract(gate1_name, gate2_name)

    def _contract_all_greedy(self) -> sparse.COO:
        """Contract all gates in the network using a greedy approach based on the number of modes.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")
        
        while self.length > 1:
            best_score = float('inf')
            best_pair = None
            for gate1_name in self.gates.keys():
                for gate2_name in self.gate_graph.get(gate1_name, set()):
                    score = self.contraction_score(gate1_name, gate2_name)
                    if score < best_score:
                        best_score = score
                        best_pair = (gate1_name, gate2_name)
            if best_pair is None:
                # raise ValueError("No valid pairs of gates to contract. The network may be disconnected.")
                 break  # if no valid pairs to contract, break the loop and return the remaining gates
            self.contract(*best_pair)
    
    def contraction_score(self, gate1_name: str, gate2_name: str) -> float:
        """Compute the contraction score between two gates.

        Args:
            gate1_name (str): The name of the first gate.
            gate2_name (str): The name of the second gate.

        Returns:
            float: The contraction score between the two gates.
        """
        gate1 = self.gates[gate1_name]
        gate2 = self.gates[gate2_name]
        dim1 = gate1.tensor.ndim
        dim2 = gate2.tensor.ndim
        nb_common_modes = len(set(gate1.outmodes) & set(gate2.inmodes))
        if nb_common_modes == 0:
            return float('inf')  # cannot contract if no common modes
        dim3 = dim1 + dim2 - 2*nb_common_modes
        cost1 = self.space_cost(dim1)
        cost2 = self.space_cost(dim2)
        cost3 = self.space_cost(dim3)
        return (cost1 + cost2) / cost3  # higher score means more efficient contraction
    
    def space_cost(self, dim: int) -> int:
        """Compute an estimate of the space cost of a gate.

        Args:
            dim (int): The dimension of the gate.

        Returns:
            int: The space cost of the gate.
        """
        m = dim // 2  # number of modes that the gate acts on
        cost = m * math.comb(m + self.n_photons - 1, self.n_photons)  # number of Fock states for each mode
        return cost

    def topological_sort(self) -> list[str]:
        """Perform a topological sort of the gates in the network based on their connections.

        Returns:
            list[str]: A list of gate names in topologically sorted order.
        """
        graph = copy.deepcopy(self.gate_graph)
        sorted_gates = []
        while graph:
            # Find a gate with no outgoing edges
            for gate_name, neighbors in graph.items():
                if not neighbors:
                    break
            else:
                # If all gates have outgoing edges, the graph is cyclic
                print("Remaining graph:", graph)
                raise ValueError("Cyclic dependency detected in gate graph.")

            # Remove the gate from the graph and add it to the sorted list
            graph.pop(gate_name)
            for neighbors in graph.values():
                neighbors.discard(gate_name)
            sorted_gates.append(gate_name)

        return sorted_gates[::-1]  # reverse the stack to get the correct order

    def display(self):
        """Display the gates in the network and their connections."""
        tnd = TensorNetworkDrawer(n_modes=self.n_modes, name=self.name)
        for gate_name in self.topological_sort():
            gate = self.gates[gate_name]
            tnd.add_gate(gate_name, startmode=gate.inmodes[0][0], endmode=gate.inmodes[-1][0])
        tnd.draw()

    def __repr__(self):
        return f"TensorNetwork(gates={self.gates})"


class TensorNetworkDrawer:
    """A class for drawing tensor networks in a human-readable format.

    Attributes:
        network (TensorNetworkCircuit): The tensor network circuit to draw.
    """
    def __init__(self, n_modes: int, name: Optional[str]=None):
        self.n_modes = n_modes
        self.name = name if name is not None else "unnamed_network"
        self._grid = {i: {"top": "", "mid": f"m{i}: ──", "bot": ""} for i in range(n_modes)}

    def add_gate(self, gate_name: str, startmode: int, endmode: Optional[int]=None):
        """Add a gate to the tensor network.

        Args:
            gate_name (str): The name of the gate.
            startmode (int): The starting mode of the gate.
            endmode (Optional[int], optional): The ending mode of the gate. Defaults to None.
        """
        
        if endmode is None:
            endmode = startmode

        max_char_length = max(*[len(self._grid[i]["top"]) for i in range(startmode, endmode + 1)],
                              *[len(self._grid[i]["mid"]) for i in range(startmode, endmode + 1)],
                              *[len(self._grid[i]["bot"]) for i in range(startmode, endmode + 1)])
        for i in range(startmode, endmode + 1):
            self._grid[i]["top"] += " " * (max_char_length - len(self._grid[i]["top"]))
            self._grid[i]["mid"] += "─" * (max_char_length - len(self._grid[i]["mid"]))
            self._grid[i]["bot"] += " " * (max_char_length - len(self._grid[i]["bot"]))

        for i in range(startmode, endmode + 1):
            if i == startmode:
                self._grid[i]["top"] += "┌" + "─" * (len(gate_name) + 2) + "┐ "
            else:
                self._grid[i]["top"] += "│" + " " * (len(gate_name) + 2) + "│ "

            if i == (startmode + endmode) // 2:
                self._grid[i]["mid"] += f"┤ {gate_name} ├─"
            else:
                self._grid[i]["mid"] += "┤" + " " * (len(gate_name) + 2) + "├─"

            if i == endmode:
                self._grid[i]["bot"] += "└" + "─" * (len(gate_name) + 2) + "┘ "
            else:
                self._grid[i]["bot"] += "│" + " " * (len(gate_name) + 2) + "│ "
    
    def draw(self):
        """Draw the tensor network."""
        max_char_length = max(*[len(self._grid[i]["top"]) for i in range(self.n_modes)],
                              *[len(self._grid[i]["mid"]) for i in range(self.n_modes)],
                              *[len(self._grid[i]["bot"]) for i in range(self.n_modes)])
        for i in range(self.n_modes):
            self._grid[i]["top"] += " " * (max_char_length - len(self._grid[i]["top"]))
            self._grid[i]["mid"] += "─" * (max_char_length - len(self._grid[i]["mid"]))
            self._grid[i]["bot"] += " " * (max_char_length - len(self._grid[i]["bot"]))

        result = [self._grid[i]["top"] + "\n" + self._grid[i]["mid"] + "\n" + self._grid[i]["bot"] for i in range(self.n_modes)]
        print(f"Tensor Network: {self.name}")
        print("\n".join(result))


if __name__ == "__main__":
    # Example usage
    nb_modes = 4
    nb_photons = 2
    circuit = TensorNetworkCircuit(nb_modes, nb_photons)
    circuit.append_bs(target=(0, 1), angles=(math.pi/4, math.pi/4), name="BS1")
    circuit.append_ps(target=0, angle=math.pi/2, name="PS1")
    circuit.append_bs(target=(1, 2), angles=(math.pi/3, math.pi/6), name="BS2")
    circuit.append_ps(target=1, angle=math.pi/3, name="PS2")
    circuit.append_ps(target=3, angle=math.pi/4, name="PS3")
    # print(circuit.gate_graph)
    circuit.display()
    circuit.contract_all(method="greedy")
    circuit.display()
    
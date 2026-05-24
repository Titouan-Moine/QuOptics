"""Tensor network module.

"""
#import sys
#import os
import warnings
from typing import Optional
import math
import string
import copy
from collections import defaultdict
from itertools import chain
import numpy as np
import sparse
from TeNCo.draw import TNSketch, TNPlot
from TeNCo.utils import find_duplicates
# from TeNCo.backend import sparse_tensordot_via_scipy
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
from PyFock.fock import fock_tensor_bs, fock_tensor_ps

class TensorGate:
    """A gate in a tensor network, representing a tensor and its connections to other tensors.

    Attributes:
    tensor (sparse.COO): The tensor associated with this gate.
    inmodes (list[tuple[int]]): The list of input modes (or wires) that this gate connects to.
        Each mode is represented as a tuple of (mode_index, counter), where counter corresponds
        to the number of times this mode has been used in previous gates, to ensure unique
        labeling. The order of modes in this list corresponds to the order of the tensor's
        dimensions.
    outmodes (list[tuple[int]]): The list of output modes (or wires) that this gate connects to.
        Each mode is represented as a tuple of (mode_index, counter), similar to inmodes.
    axis_map (Optional[dict[tuple[int, int], list[int]]]): An optional mapping from mode tuples to
        tensor axes. This can be used to specify the order of modes in the tensor explicitly.
        If not provided, a default mapping will be created based on the order of inmodes and
        outmodes.
    modes_order (Optional[list[tuple[int, int]]]): An optional list specifying the order of modes.
        Should contain all modes from inmodes and outmodes. Supports duplicates for self-
        -contractions. If not provided, the order will be determined by the order of inmodes
        followed by outmodes.
    name (Optional[str]): A unique identifier for the gate.
    tags (Optional[set[str]]): A set of tags for categorizing or annotating the gate.
    """
    def __init__(self,
                 tensor: sparse.COO,
                 inmodes: list[tuple[int, int]],
                 outmodes: list[tuple[int, int]],
                 axis_map: Optional[dict[tuple[int, int], list[int]]]=None,
                 modes_order: Optional[list[tuple[int, int]]]=None,
                 name: Optional[str]=None,
                 tags: Optional[set[str]]=None,
                 params: Optional[dict]=None,
                 warn: bool=False):
        self.name = name if name is not None else f"Gate_{id(self)}"
        self.tensor = tensor
        sorted_inmodes = sorted(inmodes)
        sorted_outmodes = sorted(outmodes)
        self.inmodes = sorted_inmodes
        self.outmodes = sorted_outmodes
        if modes_order is None:
            if warn:
                warnings.warn("No modes_order provided. Creating a default one.", UserWarning)
            modes_order = sorted_inmodes + sorted_outmodes
        self.modes_order = modes_order
        if axis_map is None:
            if warn:
                warnings.warn("No axis_map provided. Creating a default one.", UserWarning)
            axis_map = defaultdict(list)
            for i, mode in enumerate(modes_order):
                axis_map[mode].append(i)
        self.axis_map = axis_map
        self.tags = tags if tags is not None else set()
        self.params = params if params is not None else {}

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

    def _rebuild_axis_map(self) -> None:
        """Rebuild the axis_map based on the current order of inmodes and outmodes."""
        self.axis_map = defaultdict(list)
        for i, mode in enumerate(self.modes_order):
            self.axis_map[mode].append(i)

    def update_structure(self, new_modes_order: list[tuple[int, int]],
                         new_inmodes: Optional[list[tuple[int, int]]]=None,
                         new_outmodes: Optional[list[tuple[int, int]]]=None) -> None:
        """Update the structure of the gate (inmodes, outmodes, axis_map) based on a new order of modes.
        Does not modify the tensor itself.

        Args:
            new_modes_order (list[tuple[int, int]]): The new order of modes. Should contain all
                modes from inmodes and outmodes.
            new_inmodes (Optional[list[tuple[int, int]]]): An optional new list of inmodes. If None,
                the existing inmodes will be used.
            new_outmodes (Optional[list[tuple[int, int]]]): An optional new list of outmodes. If None,
                the existing outmodes will be used.
        """
        self.modes_order = new_modes_order
        self.inmodes = [mode for mode in new_modes_order if mode in self.inmodes] if new_inmodes is None else new_inmodes
        self.outmodes = [mode for mode in new_modes_order if mode in self.outmodes] if new_outmodes is None else new_outmodes
        self._rebuild_axis_map()

    def canonicalize_axes(self) -> None:
        """Canonicalize the axes of the tensor by sorting the mode tuples and reshaping the
        tensor accordingly. This ensures a consistent ordering of modes and axes for easier
        contraction and comparison. The order of the axis will then correspond to the order
        of modes in the sorted inmodes, then outmodes lists."""
        # print(f"Canonicalizing axes for gate {self.name}...")
        # print("Original inmodes:", self.inmodes)
        # print("Original outmodes:", self.outmodes)
        # print("Original axis_map:", self.axis_map)
        self.inmodes.sort()
        self.outmodes.sort()
        new_modes_order = self.inmodes + self.outmodes
        new_axis_order = []
        for mode in new_modes_order:
            new_axis_order.append(self.axis_map[mode][0])
            self.axis_map[mode].pop(0)
        self.tensor = self.tensor.transpose(new_axis_order)
        self.update_structure(new_modes_order, new_inmodes=self.inmodes, new_outmodes=self.outmodes)

    """
    def contract2(self,
                 other: 'TensorGate',
                 contract_modes: Optional[list[tuple[int, int]] | set[tuple[int, int]]]=None,
                 new_name: Optional[str]=None,
                 new_tags: Optional[set[str]]=None
                 ) -> 'TensorGate':
        modes_order_both = self.modes_order + other.modes_order
        contract_modes = None
        if contract_modes is None:
            contract_modes = find_duplicates(modes_order_both)
            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify contract_modes explicitly.")
        
        result_inmodes = [mode for mode in self.modes_order if mode in self.inmodes and mode not in contract_modes] + \
                         [mode for mode in other.modes_order if mode in other.inmodes and mode not in contract_modes]
        result_outmodes = [mode for mode in self.modes_order if mode in self.outmodes and mode not in contract_modes] + \
                          [mode for mode in other.modes_order if mode in other.outmodes and mode not in contract_modes]
        result_inmodes.sort()
        result_outmodes.sort()
        result_modes_order = result_inmodes + result_outmodes
        if find_duplicates(result_modes_order):
            raise ValueError("Contract modes cannot have duplicates in the resulting gate. Please resolve self-contractions first.")
        alphabet = string.ascii_letters
        mode_to_char = {mode: alphabet[i] for i, mode in enumerate(set(modes_order_both))}
        
        subscript_in_a = "".join(mode_to_char[mode] for mode in self.modes_order)
        subscript_in_b = "".join(mode_to_char[mode] for mode in other.modes_order)
        subscript_in = f"{subscript_in_a},{subscript_in_b}"
        subscript_out = "".join(mode_to_char[mode] for mode in result_modes_order)
        einsum_str = f"{subscript_in}->{subscript_out}"
        print(f"Einstein summation notation: {einsum_str}")
        result_tensor = sparse.einsum(einsum_str, self.tensor, other.tensor)
        result_gate = TensorGate(result_tensor,
                                 inmodes=result_inmodes,
                                 outmodes=result_outmodes,
                                 modes_order=result_modes_order,
                                 name=new_name,
                                 tags=new_tags)
        result_gate.update_structure(result_modes_order)
        result_gate.canonicalize_axes()
        return result_gate
    """

    def contract(self,
                 other: 'TensorGate',
                 contract_modes: Optional[list[tuple[int, int]]]=None,
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
        # print(f"-----Contracting gates: {self.name} and {other.name}-----")
        common_modes = set(self.outmodes) & set(other.inmodes) | set(self.inmodes) & set(other.outmodes)
        if contract_modes is None:
            contract_modes = list(common_modes)
            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify contract_modes explicitly.")

        contract_modes = sorted(contract_modes, key=lambda x: x[0])
        if any(len(self.axis_map[mode]) > 1  for mode in contract_modes) or any(len(other.axis_map[mode]) > 1 for mode in contract_modes):
            raise ValueError("Contract modes cannot have duplicates in either gate. Please resolve self-contractions first.")
        # axes_a = [self.axis_map[mode][0] for mode in contract_modes]
        # axes_b = [other.axis_map[mode][0] for mode in contract_modes]
        axes_a = [self.modes_order.index(mode) for mode in contract_modes]
        axes_b = [other.modes_order.index(mode) for mode in contract_modes]
        # axes_a = list(chain.from_iterable(self.axis_map[mode] for mode in contract_modes))
        # axes_b = list(chain.from_iterable(other.axis_map[mode] for mode in contract_modes))
        # print(f"Contracting {self.name} and {other.name} along axes {axes_a} and {axes_b}")
        if not axes_a or not axes_b:
            raise ValueError("Incompatible contraction modes. Check axis mapping")

        # remaining_modes_a = [mode for mode in self.inmodes + self.outmodes if mode not in contract_modes]
        # remaining_modes_a.sort(key=lambda m: self.axis_map[m])
        # remaining_modes_b = [mode for mode in other.inmodes + other.outmodes if mode not in contract_modes]
        # remaining_modes_b.sort(key=lambda m: other.axis_map[m])
        remaining_modes_a = [mode for mode in self.modes_order if mode not in contract_modes]
        remaining_modes_b = [mode for mode in other.modes_order if mode not in contract_modes]
        remaining_modes = remaining_modes_a + remaining_modes_b
        result_axis_map = defaultdict(list)
        for i, mode in enumerate(remaining_modes):
            result_axis_map[mode].append(i)

        # Perform the contraction using the specified modes
        result_tensor = sparse.tensordot(self.tensor,
                                         other.tensor,
                                         axes=(axes_a, axes_b))

        new_inmodes = [mode for mode in self.inmodes if mode not in contract_modes] + \
                      [mode for mode in other.inmodes if mode not in contract_modes]
        new_outmodes = [mode for mode in self.outmodes if mode not in contract_modes] + \
                       [mode for mode in other.outmodes if mode not in contract_modes]
        # print("self.inmodes:", self.inmodes)
        # print("other.inmodes:", other.inmodes)
        # print("self.outmodes:", self.outmodes)
        # print("other.outmodes:", other.outmodes)
        # print("contract_modes:", contract_modes)
        result_gate = TensorGate(tensor=result_tensor,
                                 inmodes=new_inmodes,
                                 outmodes=new_outmodes,
                                 axis_map=result_axis_map,
                                 modes_order=remaining_modes,
                                 name=new_name if new_name is not None else None,
                                 tags=new_tags if new_tags is not None else {'contracted'})
        result_gate.self_contract()  # Automatically contract any self-loops that may have been created
        result_gate.canonicalize_axes()  # Ensure consistent ordering of modes and axes in the result
        return result_gate

        # new_inmodes = [m for m in remaining_modes_a if m in self.inmodes] + \
        #           [m for m in remaining_modes_b if m in other.inmodes]
        # new_inmodes.sort(key=lambda m: m[0])  # Sort by mode index for consistency
        # new_outmodes = [m for m in remaining_modes_a if m in self.outmodes] + \
        #            [m for m in remaining_modes_b if m in other.outmodes]
        # new_outmodes.sort(key=lambda m: m[0])  # Sort by mode index for consistency

        # final_modes_order = sorted(new_inmodes) + sorted(new_outmodes)
        # transpose_axes = [result_axis_map[m] for m in final_modes_order]
        # print("Final modes order:", final_modes_order)
        # print("Transpose axes:", transpose_axes)
        # result_tensor = result_tensor.transpose(transpose_axes)
        # final_axis_map = {mode: i for i, mode in enumerate(final_modes_order)}

        # # Create a new gate for the result
        # result_gate = TensorGate(tensor=result_tensor,
        #                          inmodes=new_inmodes,
        #                          outmodes=new_outmodes,
        #                          axis_map=final_axis_map,
        #                          name=new_name if new_name is not None else None,
        #                          tags=new_tags if new_tags is not None else {'contracted'})
        # result_gate.canonicalize_axes()  # Ensure consistent ordering of modes and axes in the result

        # return result_gate

    def self_contract(self,
                      contract_modes: Optional[list[tuple[int, int]] | set[tuple[int, int]]]=None,
                      new_name: Optional[str]=None,
                      new_tags: Optional[set[str]]=None
                      ) -> None:
        """Contract this gate with itself along specified modes. This effectively traces out the
        specified modes, which must be present at least twice across the gate's inmodes and
        outmodes.

        Args:
            contract_modes (Optional[list[tuple[int, int]]]): The modes to contract over. The modes
                need to be present at least twice across both inmodes and outmodes. If None, will
                automatically contract over all common modes between this gate's inmodes and outmodes.
            new_name (Optional[str]): An optional name for the resulting contracted gate.
            new_tags (Optional[set[str]]): An optional set of tags for the resulting contracted gate.

        Returns:
            TensorGate: The resulting contracted gate.
        """
        old_inmodes = copy.deepcopy(self.inmodes)
        old_outmodes = copy.deepcopy(self.outmodes)
        old_modes_order = copy.deepcopy(self.modes_order)
        old_axis_map = copy.deepcopy(self.axis_map)
        if contract_modes is None:
            contract_modes = find_duplicates(self.modes_order)
            if not contract_modes:
                return None  # No modes to contract, return without modification

        resulting_modes = [mode for mode in self.modes_order if mode not in contract_modes]
        alphabet = string.ascii_letters
        if len(set(self.modes_order)) > len(alphabet):
            # TODO: implement a more robust labeling system for modes when there are more modes than letters in the alphabet
            raise ValueError("Too many modes to represent with single letters.")
        mode_to_char = {mode: alphabet[i] for i, mode in enumerate(set(self.modes_order))}

        subscript_in = "".join(mode_to_char[mode] for mode in self.modes_order)
        subscript_out = "".join(mode_to_char[mode] for mode in resulting_modes)
        einsum_str = f"{subscript_in}->{subscript_out}"
        # print(einsum_str)
        self.tensor = sparse.einsum(einsum_str, self.tensor)
        self.update_structure(resulting_modes)
        self.canonicalize_axes()
        # self.inmodes = [mode for mode in self.inmodes if mode not in contract_modes]
        # self.outmodes = [mode for mode in self.outmodes if mode not in contract_modes]
        # new_axis_map = defaultdict(list)
        # for i, mode in enumerate(resulting_modes):
        #     new_axis_map[mode].append(i)
        # self.axis_map = new_axis_map
        self.tags = self.tags.union(new_tags if new_tags is not None else {'self_contracted'})
        if new_name is not None:
            self.name = new_name
        # print(f"Self-contracted {self.name} along modes {contract_modes}. \n\
        #       Old inmodes: {old_inmodes}, old outmodes: {old_outmodes} \n\
        #       New inmodes: {self.inmodes}, new outmodes: {self.outmodes}")
        # print("old axis map:", old_axis_map)
        # print("axis map after self-contraction:", self.axis_map)
        # print("old modes order:", old_modes_order, "new modes order:", self.modes_order)
        # print("resulting modes:", resulting_modes)

    # TODO: fix this method to take into account the position of modes in both tensors, notably
    # when the gates are intertwined or connected.
    def kron_prod(self,
                  other: 'TensorGate',
                  new_name: Optional[str]=None,
                  new_tags: Optional[set[str]]=None) -> 'TensorGate':
        """Compute the Kronecker (outer) product of this gate with another gate.
        ...
        """
        # print(f"-----Computing Kronecker product of {self.name} and {other.name}-----")
        new_tensor = sparse.tensordot(self.tensor, other.tensor, axes=0)

        new_inmodes = self.inmodes + other.inmodes
        new_outmodes = self.outmodes + other.outmodes
        new_modes_order = self.modes_order + other.modes_order
        new_axis_map = defaultdict(list)

        n_modes_self = len(self.modes_order)

        for mode, local_axis in self.axis_map.items():
            new_axis_map[mode] = local_axis.copy()

        for mode, local_axis in other.axis_map.items():
            new_axis_map[mode] = [axis + n_modes_self for axis in local_axis]

        if new_tags is not None:
            new_tags = set(new_tags).union({'tensor_product'})

        gate = TensorGate(tensor=new_tensor,
                          inmodes=new_inmodes,
                          outmodes=new_outmodes,
                          axis_map=new_axis_map,
                          modes_order=new_modes_order,
                          name=new_name if new_name is not None else None,
                          tags=new_tags)

        gate.canonicalize_axes()
        return gate

    def prune_invalid_states(self, max_photons: int) -> None:
        """Remove elements where the total photon number exceeds max_photons.

        Args:
            max_photons (int): The maximum number of photons allowed at every given step.
        """
        # Create a mask for valid states based on the sum of photon counts across all modes
        real_inmodes_mask = [False] * len(self.modes_order)
        real_outmodes_mask = [False] * len(self.modes_order)
        feedback_inmodes_mask = [False] * len(self.modes_order)
        feedback_outmodes_mask = [False] * len(self.modes_order)
        inmodes = sorted(self.inmodes, key=lambda x: x[1])
        real_inmodes = []
        seen = set()
        for mode in inmodes:
            if mode[0] not in seen:
                real_inmodes.append(mode)
                seen.add(mode[0])
            else:
                feedback_inmodes_mask[self.modes_order.index(mode)] = True
        outmodes = sorted(self.outmodes, key=lambda x: -x[1])
        real_outmodes = []
        seen = set()
        for mode in outmodes:
            if mode[0] not in seen:
                real_outmodes.append(mode)
                seen.add(mode[0])
            else:
                feedback_outmodes_mask[self.modes_order.index(mode)] = True
        for mode in real_inmodes:
            real_inmodes_mask[self.modes_order.index(mode)] = True
        for mode in real_outmodes:
            real_outmodes_mask[self.modes_order.index(mode)] = True
        photon_counts_in_real = np.sum(self.tensor.coords[real_inmodes_mask], axis=0)
        photon_counts_out_real = np.sum(self.tensor.coords[real_outmodes_mask], axis=0)
        photon_counts_in_feedback = np.sum(self.tensor.coords[feedback_inmodes_mask], axis=0)
        photon_counts_out_feedback = np.sum(self.tensor.coords[feedback_outmodes_mask], axis=0)
        mask = (photon_counts_in_real - photon_counts_out_feedback <= max_photons) & (photon_counts_out_real - photon_counts_in_feedback <= max_photons)
        # print(f"inmodes: {self.inmodes}, outmodes: {self.outmodes}, real_inmodes: {real_inmodes}, real_outmodes: {real_outmodes}")
        # print(f"self.tensor.coords[real_inmodes_mask]: {self.tensor.coords[real_inmodes_mask]}")
        # print(f"real_inmodes_mask: {real_inmodes_mask}, real_outmodes_mask: {real_outmodes_mask}")
        # print("coords of the tensor:", self.tensor.coords)
        # print("photon counts in:", photon_counts_in_real, photon_counts_in_feedback)
        # print("photon counts out:", photon_counts_out_real, photon_counts_out_feedback)

        # Reconstruire le tenseur avec seulement les éléments valides
        self.tensor = sparse.COO(
            coords=self.tensor.coords[:, mask],
            data=self.tensor.data[mask],
            shape=self.tensor.shape
        )

    def __repr__(self):
        return f"TensorGate(name={self.name}, tensor={self.tensor},\
            inmodes={self.inmodes}, outmodes={self.outmodes}, tags={self.tags})"

class Lattice:
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
    def __init__(self,
                 n_modes: int,
                 n_photons: int,
                 gates: Optional[list[TensorGate]]=None,
                 name: Optional[str]=None):
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
        self.gate_graph: dict[str, set[str]] = {}
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
                 contract_modes: Optional[list[tuple[int, int]]]=None,
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
        gate1.self_contract()
        gate2.self_contract()
        gate1.canonicalize_axes()
        gate2.canonicalize_axes()

        if gate1 is None:
            raise ValueError(f"gate1 must be part of the network, {gate1_name} not found.")
        if gate2 is None:
            raise ValueError(f"gate2 must be part of the network, {gate2_name} not found.")

        common_modes = (set(gate1.outmodes) & set(gate2.inmodes)) | (set(gate1.inmodes) & set(gate2.outmodes))
        if contract_modes is None:
            contract_modes = list(common_modes)
            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify \
                    contract_modes explicitly.")
        if not set(contract_modes).issubset(common_modes):
            raise ValueError("Contract modes must be a subset of the common modes \
                between the two gates.")

        result_gate = gate1.contract(gate2, contract_modes=contract_modes)
        result_gate.prune_invalid_states(max_photons=self.n_photons)  # Prune invalid states after contraction
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
        # print(f"contracted {gate1_name} and {gate2_name} along modes {contract_modes}")
        # print(f"{gate1_name} inmodes: {gate1.inmodes}, {gate1_name} outmodes: {gate1.outmodes}")
        # print(f"{gate2_name} inmodes: {gate2.inmodes}, {gate2_name} outmodes: {gate2.outmodes}")
        # print(f"resulting shape: {result_gate.tensor.shape}")
        # print(f"resulting axis map: {result_gate.axis_map}")
        # result_gate.self_contract()
        # self.gate_graph[result_gate.name].discard(result_gate.name)  # Remove self-loop if it exists after self-contraction
        # print(f"Contracting gates: {gate1_name}, {gate2_name}, resulting gate: {result_gate.name}")
        # print(f"gates in the network after contraction: {list(self.gates.keys())}")

    def append(self,
               data: TensorGate | sparse.COO,
               target: Optional[list[int] | tuple[int, ...]]=None,
               name: Optional[str]=None,
               tags: Optional[set[str]]=None,
               params: Optional[dict]=None,
               allow_overwrite: bool=False,
               warn: bool=True):
        """Append a TensorGate or a sparse.COO tensor to the network.

        Args:
            data: TensorGate | sparse.COOThe tensor to append.
            target (Optional[list[int]], optional): The target modes to append to. Defaults to None.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
            params (Optional[dict], optional): The parameters for the gate. Defaults to None.
            allow_overwrite (bool, optional): Whether to allow overwriting an existing gate with
                the same name. Defaults to False.

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
        
        if name is not None and not allow_overwrite and name in self.gates:
            raise ValueError(f"A gate with name {name} already exists in the network. \
                To overwrite it, set allow_overwrite=True.")

        if isinstance(data, TensorGate):
            if warn:
                warnings.warn("Appending a TensorGate directly may lead to inconsistent mode labeling. "
                    "Please ensure that the gate's input modes are a subset of the current mode labels "
                    "in the network, and that the output modes are labeled correctly. It is recommended "
                    "to append raw tensors and let the network handle mode labeling automatically.",
                    UserWarning)
            if not set(data.inmodes).issubset(set((i, self._current_mode_counters[i])
                                                for i in range(self.n_modes))):
                raise ValueError("The input modes of the gate must be a subset of the \
                    current mode labels in the network.")
            gate = data
            gate.canonicalize_axes()  # Ensure consistent ordering of modes and axes
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
            # target = sorted(target)
            inmodes = [(i, self._current_mode_counters[i]) for i in target]
            outmodes = [(i, self._current_mode_counters[i]+1) for i in target]
            for t in target:
                self._current_mode_counters[t] += 1
            gate = TensorGate(tensor=data, inmodes=inmodes, outmodes=outmodes,
                              name=name, tags=tags, params=params)
            gate.canonicalize_axes()  # Ensure consistent ordering of modes and axes

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
                  target : tuple[int, int],
                  angles : tuple[float, float],
                  name: Optional[str]=None,
                  tags: Optional[set[str]]=None):
        """Append a parameterized beam splitter gate to the network.

        Args:
            target (tuple[int, int]): The target modes for the gate.
            angles (tuple[float, float]): The angles for the beam splitter in the format (phi, theta).
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

        if tags is None:
            tags = {'bs'}
        params = {'phi': angles[0], 'theta': angles[1]}
        bs_tensor = fock_tensor_bs(angles[0], angles[1], self.n_photons,
                                   sparse_tensor=True, check=False)
        self.append(data=bs_tensor, target=target, name=name, tags=tags, params=params)

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
            raise ValueError(f"Target mode {target} is out of bounds for a network with "
                f"{self.n_modes} modes.")
        
        if tags is None:
            tags = {'ps'}
        params = {'angle': angle}
        ps_tensor = fock_tensor_ps(angle, self.n_photons, sparse_tensor=True, check=False)
        self.append(data=ps_tensor, target=(target,), name=name, tags=tags, params=params)

    def append_input_state_single_tensor(self,
                           state: list[int] | tuple[int, ...] | np.ndarray | sparse.COO,
                           target: Optional[list[int] | tuple[int, ...]]=None,
                           name: Optional[str]=None,
                           tags: Optional[set[str]]=None):
        """Append an input state preparation gate to the network. Only supports Fock states for now,
        which can be provided as a list of photon counts per mode or as a sparse.COO tensor.

        Args:
            target (list[int] | tuple[int, ...]): The target modes for the input state.
            state (list[int] | tuple[int, ...] | sparse.COO): The Fock state for the input state.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        if target is None:
            target = tuple(range(len(state)))
        
        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) != len(target):
                raise ValueError(f"State has length {len(state)} but target has length {len(target)}. \
                    Length of the state must match the number of target modes.")
            coords = np.array(state).reshape(-1, 1)
            state = sparse.COO(coords=coords, data=1, shape=(self.n_photons+1,)*len(target))

        if tags is None:
            tags = {'input'}

        params = {'state': "|" + "".join(str(s) for s in state) + "⟩"}
        gate = TensorGate(tensor=state,
                          inmodes=[],
                          outmodes=[(t, 0) for t in target],
                          name=name,
                          tags=tags,
                          params=params)
        self.append(gate, warn=False)
    
    def append_input_state(self,
                           state: list[int] | tuple[int, ...] | np.ndarray,
                           target: Optional[list[int] | tuple[int, ...]]=None,
                           name: Optional[str]=None,
                           tags: Optional[set[str]]=None):
        """Append an input state preparation gate to the network. Only supports Fock states for now,
        which can be provided as a list of photon counts per mode.

        Args:
            target (list[int] | tuple[int, ...]): The target modes for the input state.
            state (list[int] | tuple[int, ...] | sparse.COO): The Fock state for the input state.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        if target is None:
            target = tuple(range(len(state)))
        
        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) != len(target):
                raise ValueError(f"State has length {len(state)} but target has length {len(target)}. \
                    Length of the state must match the number of target modes.")

        if tags is None:
            tags = {'input'}

        tensor_list = []
        for s in state:
            if s < 0 or s > self.n_photons:
                raise ValueError(f"Photon count {s} is out of bounds for a network with {self.n_photons} photons.")
            coords = np.array([s]).reshape(-1, 1)
            tensor_list.append(sparse.COO(coords=coords, data=1, shape=(self.n_photons+1,)))

        gate_list = []
        for i, t in enumerate(target):
            gate_list.append(TensorGate(tensor=tensor_list[i],
                                        inmodes=[],
                                        outmodes=[(t, 0)],
                                        name=f"{name}_mode{t}" if name else None,
                                        tags=tags,
                                        params={'state': "|" + str(state[i]) + "⟩"}))
        
        for gate in gate_list:
            self.append(gate, warn=False)

    def append_output_state_single_tensor(self,
                            state: list[int] | tuple[int, ...] | np.ndarray | sparse.COO,
                            target: Optional[list[int] | tuple[int, ...]]=None,
                            name: Optional[str]=None,
                            tags: Optional[set[str]]=None):
        """Append an output state projection gate to the network. Only supports Fock states for now,
        which can be provided as a list of photon counts per mode or as a sparse.COO tensor.

        Args:
            target (list[int] | tuple[int, ...]): The target modes for the output state.
            state (list[int] | tuple[int, ...] | sparse.COO): The Fock state for the output state.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        if target is None:
            target = tuple(range(len(state)))
        
        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) != len(target):
                raise ValueError(f"State has length {len(state)} but target has length {len(target)}. \
                    Length of the state must match the number of target modes.")
            coords = np.array(state).reshape(-1, 1)
            state = sparse.COO(coords=coords, data=1, shape=(self.n_photons+1,)*len(target))

        if tags is None:
            tags = {'output'}
        else:
            tags = set(tags).union({'output'})

        params = {'state': "|" + "".join(str(s) for s in state) + "⟩"}
        gate = TensorGate(tensor=state,
                          inmodes=[(t, self._current_mode_counters[t]) for t in target],
                          outmodes=[],
                          name=name,
                          tags=tags,
                          params=params)
        self.append(gate, warn=False)

    def append_output_state(self,
                        state: list[int] | tuple[int, ...] | np.ndarray,
                        target: Optional[list[int] | tuple[int, ...]]=None,
                        name: Optional[str]=None,
                        tags: Optional[set[str]]=None):
        """Append an output state preparation gate to the network. Only supports Fock states for now,
        which can be provided as a list of photon counts per mode.

        Args:
            target (list[int] | tuple[int, ...]): The target modes for the output state.
            state (list[int] | tuple[int, ...] | sparse.COO): The Fock state for the output state.
            name (Optional[str], optional): The name of the gate. Defaults to None.
            tags (Optional[set[str]], optional): The tags for the gate. Defaults to None.
        """
        if target is None:
            target = tuple(range(len(state)))
        
        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) != len(target):
                raise ValueError(f"State has length {len(state)} but target has length {len(target)}. \
                    Length of the state must match the number of target modes.")

        if tags is None:
            tags = {'output'}

        tensor_list = []
        for s in state:
            if s < 0 or s > self.n_photons:
                raise ValueError(f"Photon count {s} is out of bounds for a network with {self.n_photons} photons.")
            coords = np.array([s]).reshape(-1, 1)
            tensor_list.append(sparse.COO(coords=coords, data=1, shape=(self.n_photons+1,)))

        gate_list = []
        for i, t in enumerate(target):
            gate_list.append(TensorGate(tensor=tensor_list[i],
                                        inmodes=[(t, self._current_mode_counters[t])],
                                        outmodes=[],
                                        name=f"{name}_mode{t}" if name else None,
                                        tags=tags,
                                        params={'state': "|" + str(state[i]) + "⟩"}))
        
        for gate in gate_list:
            self.append(gate, warn=False)

    def _kron_prod_all(self) -> None:
        """Compute the Kronecker product of all gates in the network to obtain a single gate
        representing the entire network.

        Returns:
            TensorGate: A single gate representing the entire network after taking the Kronecker
                product of all gates.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")

        gates_list = list(self.gates.values())
        present_modes = {mode[0] for gate in gates_list for mode in gate.inmodes + gate.outmodes}
        missing_modes = [mode for mode in range(self.n_modes) if mode not in present_modes]

        for mode in missing_modes:
            identity_tensor = fock_tensor_ps(0.0, self.n_photons, sparse_tensor=True, check=False)
            gates_list.append(
                TensorGate(
                    tensor=identity_tensor,
                    inmodes=[(mode, 0)],
                    outmodes=[(mode, 1)],
                    name=f"I{mode}",
                    tags={"identity"},
                    params={"angle": 0.0},
                )
            )

        gates_list.sort(key=lambda gate: min(mode[0] for mode in gate.inmodes + gate.outmodes))
        result_gate = gates_list[0]
        for gate in gates_list[1:]:
            result_gate = result_gate.kron_prod(gate)
        result_gate.prune_invalid_states(max_photons=self.n_photons)

        self.gates = {result_gate.name: result_gate}
        self.length = 1
        self.gate_graph = {result_gate.name: set()}

    def contract_all(self, method: Optional[str]=None, kron: bool=False) -> None:
        """Contract all gates in the network to obtain the final output tensor.

        Args:
            method (str): The contraction method to use. Currently only supports 'naive',
            'greedy' and 'propagation'.
            kron (bool): Whether to compute the Kronecker product of all gates at the end. Unless 
                you need to have the network represented as a single gate, it is recommended to
                keep kron=False to avoid unnecessary computations, and for memory efficiency.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if method is None:
            input_gates = [gate_name for gate_name, gate in self.gates.items() if 'input' in gate.tags]
            if not input_gates:
                method = 'greedy'  # default to greedy method if not specified
            else:
                method = 'propagation'  # default to propagation method if input state is present

        if method.lower() == 'naive':
            self._contract_all_naive()
        elif method.lower() == 'greedy':
            self._contract_all_greedy()
        elif method.lower() == 'propagation':
            self._contract_all_propagation()
        else:
            raise ValueError(f"Unsupported contraction method: {method}. Supported \
                methods are 'naive', 'greedy', and 'propagation'.")

        if kron:
            self._kron_prod_all()

    def _contract_all_naive(self) -> None:
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

    def _contract_all_greedy(self) -> None:
        """Contract all gates in the network using a greedy approach based on the number of modes.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")

        # self.display(method="txt", label_mode="short")
        while self.length > 1:
            best_score = float('-inf')
            best_pair = None
            for gate1_name in self.gates.keys():
                for gate2_name in self.gate_graph.get(gate1_name, set()):
                    score = self.contraction_score(gate1_name, gate2_name)
                    if score > best_score:
                        best_score = score
                        best_pair = (gate1_name, gate2_name)
            if best_pair is None:
                # raise ValueError("No valid pairs of gates to contract."
                # " The network may be disconnected.")
                # print(self.gate_graph, self.gates.keys())
                break
            # print(f"Best pair to contract: {best_pair} with score {best_score}")
            # print(f"gate graph: {self.gate_graph}")
            # print(f"Common wires: {best_wires}, common modes: {best_modes}")
            self.contract(*best_pair)

    def _contract_all_propagation2(self) -> None:
        """Contract all gates in the network using a propagation-based approach, where we iteratively
        contract gates that are directly connected to the input state until we obtain the final output tensor.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")

        # Find gates that are directly connected to the input state (i.e., have 'input' tag)
        input_gates = [gate_name for gate_name, gate in self.gates.items() if 'input' in gate.tags]
        if not input_gates:
            raise ValueError("No input state found in the network. Please append an input state before \
                contracting with the 'propagation' method.")

        # Initialize a queue with the input gates
        queue = input_gates.copy()
        while queue:
            input_name = queue.pop(0)
            while self.gate_graph.get(input_name, set()):
                gate2_name = next(iter(self.gate_graph.get(input_name, set())))
                self.contract(input_name, gate2_name, new_gate_name=f"{input_name}@{gate2_name}")
                input_name = f"{input_name}@{gate2_name}"  # Update input_name to the name of the contracted gate for further propagation
                # queue.append(f"{input_name}@{gate2_name}")  # Add the contracted gate to the queue for further contraction

    def _contract_all_propagation(self) -> None:
        """Contract all gates in the network using a propagation-based approach, where we iteratively
        contract gates that are directly connected to the input state until we obtain the final output tensor.

        Returns:
            sparse.COO: The resulting tensor after contracting all gates in the network.
        """
        if not self.gates:
            raise ValueError("No gates to contract in the network.")

        # Find gates that are directly connected to the input state (i.e., have 'input' tag)
        input_gates = [gate_name for gate_name, gate in self.gates.items() if 'input' in gate.tags]
        if not input_gates:
            raise ValueError("No input state found in the network. Please append an input state before \
                contracting with the 'propagation' method.")

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
        common_wires = ((set(gate1.outmodes) & set(gate2.inmodes))
                        | (set(gate1.inmodes) & set(gate2.outmodes)))
        nb_common_wires = len(common_wires)

        # compute the number of common modes between the two gates, to see if some of them are not
        # connected in which case the resulting gate will have a weird shape (feedbacks, arches)
        # and the contraction will be less efficient and error-prone. We can use this information
        # to penalize contractions that would lead to such shapes.
        common_modes = (set(mode[0] for mode in gate1.modes_order)
                        & set(mode[0] for mode in gate2.modes_order))
        if len(common_modes) != nb_common_wires:
            return float('-inf')
        
        dim3 = dim1 + dim2 - 2*nb_common_wires
        #print(dim3)
        # if dim3 == 0:
        #    return float('inf')  # avoid division by zero, treat as best score since it means the resulting gate is a scalar
        cost1 = self.space_cost(dim1)
        cost2 = self.space_cost(dim2)
        cost3 = self.space_cost(dim3)
        return cost1 + cost2 - cost3  # higher score means more efficient contraction

    def space_cost(self, dim: int, state_gate: bool=False) -> int:
        """Compute an estimate of the space cost of a gate.

        Args:
            dim (int): The dimension of the gate.
            state_gate (bool): Whether the gate represents a quantum state.

        Returns:
            int: The space cost of the gate.
        """
        if state_gate:
            cost = math.comb(dim + self.n_photons, self.n_photons)  # number of Fock states for each mode
        elif dim == 0:
            cost = 1  # scalar
        else:
            cost = np.sum([math.comb(dim + i - 1, i)**2 for i in range(self.n_photons + 1)])  # number of Fock states for each mode
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
                # print("Remaining graph:", graph)
                raise ValueError("Cyclic dependency detected in gate graph.")

            # Remove the gate from the graph and add it to the sorted list
            graph.pop(gate_name)
            for neighbors in graph.values():
                neighbors.discard(gate_name)
            sorted_gates.append(gate_name)

        return sorted_gates[::-1]  # reverse the stack to get the correct order

    # TODO: implement a more robust display method that can handle more complex networks notably
        # with cycles or multiple connections between gates. This may involve implementing a custom
        # graph drawing algorithm or using a library like Graphviz to visualize the gate graph structure.
    def display_text(self, label_mode: Optional[str]="full"):
        """Display the tensor network in text format."""
        tnd = TNSketch(n_modes=self.n_modes, name=self.name, label_mode=label_mode)
        for gate_name in self.topological_sort():
            gate = self.gates[gate_name]
            tnd.add_gate(gate_name,
                         startmode=gate.inmodes[0][0],
                         endmode=gate.inmodes[-1][0],
                         tags=gate.tags,
                         params=gate.params)
        tnd.draw()

    def display_plt(self, label_mode: Optional[str]="full"):
        """Display the tensor network using Matplotlib."""
        tnd = TNPlot(n_modes=self.n_modes, name=self.name, label_mode=label_mode)
        for gate_name in self.topological_sort():
            gate = self.gates[gate_name]
            tnd.add_gate(gate_name,
                         startmode=gate.inmodes[0][0],
                         endmode=gate.inmodes[-1][0],
                         tags=gate.tags,
                         params=gate.params)
        tnd.finalize()

    def display(self, method: str="text", label_mode: str="full"):
        """Display the gates in the network and their connections.
        Only graphs without cycles or multiple connections between the same gates can be displayed for now,
        attempting to display graphs with cycles will raise an error. The display method can be chosen between
        a simple text-based representation and a more visual Matplotlib-based representation.
        
        Args:
            method (str): The method to use for displaying the network. Supported methods are 'text' and 'plt'.
            label_mode (str): The level of detail to include in gate labels. Supported modes are 'full', 'short', 'minimal', and 'no_values'.
        """
        if method.lower() in ["text", "txt"]:
            self.display_text(label_mode=label_mode)
        elif method.lower() in ["plt", "matplotlib", "pyplot", "plot"]:
            self.display_plt(label_mode=label_mode)
        else:
            raise ValueError(f"Unknown display method: {method}")

    def __repr__(self):
        return f"TensorNetwork(gates={self.gates})"

def main():
    """Example usage"""
    nb_modes = 4
    nb_photons = 2
    circuit = Lattice(nb_modes, nb_photons)
    circuit.append_bs(target=(0, 1), angles=(math.pi/4, math.pi/4), name="BS1")
    circuit.append_ps(target=0, angle=math.pi/2, name="PS1")
    circuit.append_bs(target=(1, 2), angles=(math.pi/3, math.pi/6), name="BS2")
    circuit.append_ps(target=1, angle=math.pi/3, name="PS2")
    circuit.append_ps(target=3, angle=math.pi/4, name="PS3")
    # print(circuit.gate_graph)
    circuit.display(method="plt", label_mode="short")
    circuit.contract_all(method="greedy")
    circuit.display(method='plt', label_mode="minimal")

if __name__ == "__main__":
    main()

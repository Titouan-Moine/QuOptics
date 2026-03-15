"""Tensor network module.

"""
import warnings
from typing import Optional
# import numpy as np
import sparse
from sparse_backend import sparse_tensordot_via_scipy

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
    name (Optional[str]): A unique identifier for the gate.
    tags (Optional[set[str]]): A set of tags for categorizing or annotating the gate.
    """
    def __init__(self,
                 tensor: sparse.COO,
                 inmodes: list[tuple[int]],
                 outmodes: list[tuple[int]],
                 axis_map: Optional[dict[tuple[int, int], int]]=None,
                 name: Optional[str]=None,
                 tags: Optional[set[str]]=None
                 ):
        self.name = name if name is not None else f"Gate_{id(self)}"
        self.tensor = tensor
        self.inmodes = inmodes
        self.outmodes = outmodes
        if axis_map is None:
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
                 contract_modes: Optional[list[tuple[int]]]=None
                 ) -> 'TensorGate':
        if contract_modes is None:
            contract_modes = list(set(self.outmodes) & set(other.inmodes))

            if not contract_modes:
                raise ValueError("No common modes to contract on. Please specify contract_modes explicitly.")
        
        axes_a = [self.inmode_to_axis(mode) for mode in contract_modes]
        axes_b = [other.outmode_to_axis(mode) for mode in contract_modes]
        axes_map_a = {mode: axis for mode, axis in self.axis_map.items() if mode[0] in contract_modes}
        axes_map_b = {mode: axis for mode, axis in other.axis_map.items() if mode[0] in contract_modes}

        # Perform the contraction using the specified modes
        result_tensor = sparse_tensordot_via_scipy(self.tensor,
                                                   other.tensor,
                                                   axes_a=axes_a,
                                                   axes_b=axes_b)

        # Create a new gate for the result
        result_gate = TensorGate(tensor=result_tensor, inmodes=self.inmodes, outmodes=other.outmodes)

        return result_gate

    def __repr__(self):
        return f"TensorGate(name={self.name}, tensor={self.tensor},\
            inmodes={self.inmodes}, outmodes={self.outmodes}, tags={self.tags})"

class TensorNetworkCircuit:
    """A tensor network circuit, consisting of multiple TensorGates and their connections.

    Attributes:
        nodes (list[TensorGate]): The list of gates in the network.
    """
    def __init__(self, n_modes: int, gates: Optional[list[TensorGate]]=None):
        if gates is not None and len(gates) > 0:
            warnings.warn("Initializing with a non-empty list of gates may lead \
                to inconsistent mode labeling. Please ensure that the gates are \
                labeled correctly or initialize with an empty list and append \
                gates one by one.", UserWarning)
        self.gates = gates if gates is not None else []
        self.length = len(self.gates)
        self.n_modes = n_modes
        
        mode_counters = {i: 0 for i in range(n_modes)}
        for gate in self.gates:
            for outmode in gate.outmodes:
                mode_counters[outmode[0]] = max(mode_counters[outmode[0]], outmode[1])
        self._current_mode_counters = mode_counters

    def append(self,
               other: TensorGate | sparse.COO,
               target: Optional[list[int]]=None,
               name: Optional[str]=None,
               tags: Optional[set[str]]=None):
        """Append a TensorGate or a sparse.COO tensor to the network.

        Args:
            other: TensorGate | sparse.COOThe tensor to append.
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

        if isinstance(other, TensorGate):
            warnings.warn("Appending a TensorGate directly may lead to inconsistent mode labeling. \
                Please ensure that the gate's input modes are a subset of the current mode labels \
                in the network, and that the output modes are labeled correctly. It is recommended \
                to append raw tensors and let the network handle mode labeling automatically.",
                UserWarning)
            if not set(other.inmodes).issubset(set((i, self._current_mode_counters[i])
                                                for i in range(self.n_modes))):
                raise ValueError("The input modes of the gate must be a subset of the \
                    current mode labels in the network.")
            gate = other
            for outmode in gate.outmodes:
                self._current_mode_counters[outmode[0]] = outmode[1]
        
        elif isinstance(other, sparse.COO):
            if target is None:
                raise ValueError("Must provide target when appending a raw tensor.")
            for t in target:
                if t < 0 or t >= self.n_modes:
                    raise ValueError(f"Target mode {t} is out of bounds for a network with {self.n_modes} modes.")
            if other.ndim != 2*len(target):
                raise ValueError(f"Tensor has {other.ndim} dimensions but target has {len(target)} \
                    modes. Dimensions of the tensor must be twice the number of target modes.")
            inmodes = [(i, self._current_mode_counters[i]) for i in target]
            outmodes = [(i, self._current_mode_counters[i]+1) for i in target]
            for t in target:
                self._current_mode_counters[t] += 1
            gate = TensorGate(tensor=other, inmodes=inmodes, outmodes=outmodes,
                              name=name, tags=tags)
        
        else:
            raise ValueError("Can only append a TensorGate or a sparse.COO tensor.")

        self.gates.append(gate)
        self.length += 1

    def __repr__(self):
        return f"TensorNetwork(gates={self.gates})"

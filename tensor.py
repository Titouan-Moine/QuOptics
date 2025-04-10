
import numpy as np
import itertools as it

tensor_types_dict = {
    'H' : (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]), # Hadamard
    'CX' : np.array([])
}


class Node:
    def __init__(self, inwires, outwires, tensor=None, tensor_type=None, rank=None, dimensions=None):
        self.tensor_type = tensor_type
        self.tensor = np.array(tensor) # The tensor itself
        if tensor is None:
            self.tensor = tensor_types_dict[tensor_type]
        self.dimensions = np.array(dimensions) # List of dimensions of the tensor
        self.rank = rank # Rank of the tensor
        if dimensions is None:
            self.dimensions = np.array(list(tensor.shape))
        if rank is None:
            self.rank = len(self.dimensions)
        self.inwires = np.array(inwires) # List of wires (indices) entering this node in order
        self.outwires = np.array(outwires) # List of wires (indices) leaving this node in order
    
    def get_indimension(self):
        return self.dimensions[:len(self.inwires)]
    
    def get_outdimension(self):
        return self.dimensions[len(self.inwires):]
    
    
    def contract(self, other):
        """contract this tensor with another tensor (which is on the output side of the first) along the common wires.

        Args:
            other (Node): another tensor to contract with
        """
        # find the wires that will be contracted
        temp = {e : 0 for e in self.outwires}
        for e in other.inwires:
            temp[e] = 0
        for e in self.outwires:
            temp[e] += 1
        for e in other.inwires:
            temp[e] += 1
        common_wires = [e for e in temp if temp[e] == 2]
        common_wires = np.array(common_wires)
        
        # lists of inwires and outwires of the resulting tensor
        other_inwires_mask = ~np.isin(other.inwires, common_wires)
        self_outwires_mask = ~np.isin(self.outwires, common_wires)
        other_inwires = other.inwires[other_inwires_mask]
        self_outwires = self.outwires[self_outwires_mask]
        contracted_inwires = np.concatenate((self.inwires, other_inwires))
        contracted_outwires = np.concatenate((self_outwires, other.outwires))
        
        # initialize the resulting tensor
        A_indimension = self.get_indimension()
        B_indimension = other.get_indimension()[other_inwires_mask]
        A_outdimension = self.get_outdimension()[self_outwires_mask]
        B_outdimension = other.get_outdimension()
        common_dimension = self.get_outdimension()[~self_outwires_mask] # common dimension of the contracted wires
        temp = other.get_indimension()[~other_inwires_mask]
        assert np.array_equal(common_dimension, temp), f"common dimensions of the contracted wires should be equal.\n Left tensor has common outdimension {common_dimension} while right tensor has common indimension {temp}"
        C = np.zeros(np.concatenate((A_indimension, B_indimension, A_outdimension, B_outdimension)), dtype=int)
        
        
        for A_in_index in it.product(*[range(A_indimension[i]) for i in range(len(A_indimension))]):
            for B_in_index in it.product(*[range(B_indimension[i]) for i in range(len(B_indimension))]):
                for A_out_index in it.product(*[range(A_outdimension[i]) for i in range(len(A_outdimension))]):
                    for B_out_index in it.product(*[range(B_outdimension[i]) for i in range(len(B_outdimension))]):
                        A_temp = np.zeros(len(self.outwires), dtype=int)
                        A_temp[self_outwires_mask] = A_out_index
                        B_temp = np.zeros(len(other.inwires), dtype=int)
                        B_temp[other_inwires_mask] = B_in_index
                        for common_index in it.product(*[range(common_dimension[i]) for i in range(len(common_dimension))]):
                            A_temp[~self_outwires_mask] = common_index
                            B_temp[~other_inwires_mask] = common_index
                            A_full_index = np.concatenate((A_in_index, A_temp))
                            B_full_index = np.concatenate((B_temp, B_out_index))
                            C[tuple(np.concatenate((A_in_index, B_in_index, A_out_index, B_out_index)).astype(int))] +=\
                                self.tensor[tuple(A_full_index)]*\
                                other.tensor[tuple(B_full_index)]
        
        self.tensor = C
        self.inwires = contracted_inwires
        self.outwires = contracted_outwires
        self.dimensions = np.concatenate((A_indimension, B_indimension, A_outdimension, B_outdimension))
        self.rank = len(self.dimensions)


A = Node(np.array([0, 1]), np.array([0, 1, 2]), np.ones((5, 3, 4, 8, 5)))
B = Node(np.array([1, 2, 3]), np.array([3, 1]), np.ones((8, 5, 10, 5, 7)))
A.contract(B)
print(A.tensor)
print(A.dimensions)
print(A.inwires, A.outwires)

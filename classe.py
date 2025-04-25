import numpy as np
class Tensor:
  def __init__(self, dimension, content, name="Unnamed Tensor"):
    """
    Initialize a Tensor object.

    :param dimension: Tuple representing the dimensions of the tensor.
    :param content: The content of the tensor (e.g., a list, numpy array, etc.).
    :param name: Optional name for the tensor.
    """
    self.dimension = dimension
    self.content = content
    self.name = name

  def __repr__(self):
    return f"Tensor(name={self.name}, dimension={self.dimension})"

  # def reshape(self, new_dimension):
  #   """
  #   Reshape the tensor to a new dimension.

  #   :param new_dimension: Tuple representing the new dimensions.
  #   """
  #   # Add logic to reshape content if necessary
  #   self.dimension = new_dimension


  def get_content(self):
    """
    Get the content of the tensor.

    :return: The content of the tensor.
    """
    return self.content

  def set_content(self, new_content):
    """
    Set new content for the tensor.

    :param new_content: The new content to set.
    """
    self.content = new_content

  """@staticmethod
  def empty_tenser(name="Empty Tensor"):
    "
    Create an empty tensor.
      
    :return: A Tensor object with no content.
    "
    return Tensor(dimension=(), content=None, name=name)

    
  def ket_0():
    "
    :return: A Tensor object representing |0>.
    "
    ket_0_content = [[1], [0]]  # The vector (1, 0) as a tensor of shape (2, 1)
    return Tensor(dimension=(2, 1), content=ket_0_content, name="Ket 0")
  
  def ket_1():
    return Tensor(dimension=(2, 1), content=[[0], [1]], name="Ket 1")   

  def c_not():
    "
    :return: A Tensor object representing the CNOT gate. With control on bit 1 and target bit 2.
    "
    c_not_content = [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 0], [0, 1]], [[0, 1], [0, 0]]]]
    return Tensor(dimension=(2, 2, 2, 2), content=c_not_content, name="CNOT")

  def hadamard():
    hadamard_content = [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
    return Tensor(dimension=(2, 2), content=hadamard_content, name="Hadamard")
"""
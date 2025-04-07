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

  def reshape(self, new_dimension):
    """
    Reshape the tensor to a new dimension.

    :param new_dimension: Tuple representing the new dimensions.
    """
    # Add logic to reshape content if necessary
    self.dimension = new_dimension

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

  @staticmethod
  def ket_0():
    """
    Create a specific tensor representing the quantum state |0>.

    :return: A Tensor object representing |0>.
    """
    ket_0_content = [[1], [0]]  # The vector (1, 0) as a tensor of shape (2, 1)
    return Tensor(dimension=(2, 1), content=ket_0_content, name="Ket 0")
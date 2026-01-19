import numpy as np
import matplotlib.pyplot as plt
import time
from clements_scheme.clements_scheme import full_clements
from clements_scheme.rnd_unitary import random_unitary
import small_matrix as sm
import fock_amplitude as fa
#from fock_amplitude import clements_fock_tensor
# Configuration for benchmarking
def transformationA(U):
  return sm.contract_circuit_then_fock(U)

def transformationB(U):
  a,b=U
  return fa.clements_fock_tensor(a,b)

def main():
  max_size=10
  circuit_sizes = [i for i in range (2,max_size+1)]  # Vary the circuit sizes
  #num_trials_per_size = 10  # Number of trials for each size to get average execution time

  # Initialize storage for results
  results_transfo_A = {'sizes': [], 'times': [], 'errors': []}
  results_transfo_B = {'sizes': [], 'times': [], 'errors': []}

  times_A = []
  times_B = []

  # Benchmarking loop
  for size in circuit_sizes:
     
      # Generate a random unitary matrix
      U = random_unitary(size)
      
      # Apply full_clements decomposition as input to transformations
      to_test = full_clements(U)
      
      # ===== Transformation A =====
      start_time = time.time()
      result_A = transformationA(to_test)
      elapsed_time_A = time.time() - start_time
      times_A.append(elapsed_time_A)

      
      # ===== Transformation B =====
      
      start_time = time.time()
      result_B = transformationB(to_test)
      elapsed_time_B = time.time() - start_time
      times_B.append(elapsed_time_B)
          
  plt.plot(circuit_sizes,times_A, color='blue')
  plt.plot(circuit_sizes,times_B, color='red')
  plt.show()

main()
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
  return sm.contract_circuit(U)

def transformationB(U):
  a,b=U
  return fa.clements_fock_tensor(a,b)

def main():
  circuit_sizes = [2, 3, 4, 5, 6]  # Vary the circuit sizes
  num_trials_per_size = 10  # Number of trials for each size to get average execution time

  # Initialize storage for results
  results_transfo_A = {'sizes': [], 'times': [], 'errors': []}
  results_transfo_B = {'sizes': [], 'times': [], 'errors': []}



  # Benchmarking loop
  for size in circuit_sizes:
      times_A = []
      times_B = []
      
      for trial in range(num_trials_per_size):
          # Generate a random unitary matrix
          U = random_unitary(size)
          
          # Apply full_clements decomposition as input to transformations
          to_test = full_clements(U)
          
          # ===== Transformation A =====
          try:
              start_time = time.time()
              result_A = transformationA(to_test)
              elapsed_time_A = time.time() - start_time
              times_A.append(elapsed_time_A)
          except Exception as e:
              times_A.append(None)
          
          # ===== Transformation B =====
          
          start_time = time.time()
          result_B = transformationB(to_test)
          elapsed_time_B = time.time() - start_time
          times_B.append(elapsed_time_B)
          
      
      # Calculate average times and standard deviations (filtering out None values)
      valid_times_A = [t for t in times_A if t is not None]
      valid_times_B = [t for t in times_B if t is not None]
      
      avg_time_A = np.mean(valid_times_A) if valid_times_A else None
      avg_time_B = np.mean(valid_times_B) if valid_times_B else None
      
      std_time_A = np.std(valid_times_A) if valid_times_A else None
      std_time_B = np.std(valid_times_B) if valid_times_B else None
      
      # Store results for later plotting and analysis
      results_transfo_A['sizes'].append(size)
      results_transfo_A['times'].append(avg_time_A)
      results_transfo_A['errors'].append(std_time_A)
      
      results_transfo_B['sizes'].append(size)
      results_transfo_B['times'].append(avg_time_B)
      results_transfo_B['errors'].append(std_time_B)


  # Sanitize results: replace None with NaN for times, and 0.0 for errors
  sizes_A = results_transfo_A['sizes']
  times_A = [t if t is not None else np.nan for t in results_transfo_A['times']]
  errs_A  = [e if e is not None else 0.0   for e in results_transfo_A['errors']]

  sizes_B = results_transfo_B['sizes']
  times_B = [t if t is not None else np.nan for t in results_transfo_B['times']]
  errs_B  = [e if e is not None else 0.0   for e in results_transfo_B['errors']]

  plt.errorbar(sizes_A, times_A, yerr=errs_A, fmt='-o', color='blue', label='transfo_A', capsize=4)
  plt.errorbar(sizes_B, times_B, yerr=errs_B, fmt='-o', color='red',  label='transfo_B', capsize=4)
  plt.show()

main()
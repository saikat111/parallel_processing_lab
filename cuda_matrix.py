!pip install cupy-cuda12x
import cupy as cp
import numpy as np
import time

def batch_matrix_multiplication_cuda(K, M, N, P):
    # Generate random matrices on CPU
    A_cpu = np.random.rand(K, M, N)
    B_cpu = np.random.rand(K, N, P)

    # Move matrices to GPU
    A = cp.asarray(A_cpu)
    B = cp.asarray(B_cpu)

    # Start timing
    start_time = time.time()

    # Perform batch matrix multiplication
    C = cp.matmul(A, B)

    # End timing
    end_time = time.time()

    print(f"\nTotal Time Taken: {end_time - start_time:.4f} seconds\n")
    print(f"Total {K} matrices multiplied.")

# --------- Test --------------

# Set Inputs
K = 100  # Number of matrices
M = 50   # Rows in A
N = 50   # Columns in A = Rows in B
P = 50   # Columns in B

batch_matrix_multiplication_cuda(K, M, N, P)

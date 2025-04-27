
from mpi4py import MPI
import numpy as np
import time

def multiply_matrices(A_list, B_list):
    result_list = []
    for A, B in zip(A_list, B_list):
        result_list.append(np.dot(A, B))
    return result_list

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Input K, M, N, P
        K = int(input("Enter K: "))
        M = int(input("Enter M: "))
        N = int(input("Enter N: "))
        P = int(input("Enter P: "))

        # Generate random matrices A and B
        A = [np.random.rand(M, N) for _ in range(K)]
        B = [np.random.rand(N, P) for _ in range(K)]

        # Split matrices among processes
        chunk_size = K // size
        A_chunks = [A[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
        B_chunks = [B[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

        if len(A_chunks) > 0:
            A_chunks[-1].extend(A[size*chunk_size:])
            B_chunks[-1].extend(B[size*chunk_size:])
    else:
        A_chunks = None
        B_chunks = None
        K = None
        M = None
        N = None
        P = None

    # Broadcast K, M, N, P to all processes
    K = comm.bcast(K, root=0)
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    P = comm.bcast(P, root=0)

    # Scatter A and B chunks
    local_A = comm.scatter(A_chunks, root=0)
    local_B = comm.scatter(B_chunks, root=0)

    # Start timing
    start_time = MPI.Wtime()

    # Local multiplication
    local_result = multiply_matrices(local_A, local_B)

    # Gather all results
    all_results = comm.gather(local_result, root=0)

    if rank == 0:
        end_time = MPI.Wtime()
        total_time = end_time - start_time
        print(f"\nTotal Time Taken: {total_time:.4f} seconds\n")
        # Flatten list of lists
        results = [item for sublist in all_results for item in sublist]
        print(f"Total {len(results)} matrices multiplied.")

if __name__ == "__main__":
    main()

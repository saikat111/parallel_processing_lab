from mpi4py import MPI
import time

def count_pattern(text_chunk, pattern):
    return text_chunk.count(pattern)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "paragraph.txt"
    pattern = "%x%"  # Pattern to search

    if rank == 0:
        # Read paragraph from file
        with open(filename, 'r') as f:
            text = f.read()

        # Divide text into parts
        chunk_size = len(text) // size
        chunks = [text[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

        if len(chunks) > 0:
            chunks[-1] += text[size*chunk_size:]  # Last chunk gets extra
    else:
        chunks = None

    # Scatter text chunks
    text_chunk = comm.scatter(chunks, root=0)

    # Start timing
    start_time = MPI.Wtime()

    # Each process counts pattern in its chunk
    local_count = count_pattern(text_chunk, pattern)

    # Gather counts
    all_counts = comm.gather(local_count, root=0)

    if rank == 0:
        total_count = sum(all_counts)

        # End timing
        end_time = MPI.Wtime()
        total_time = end_time - start_time

        # Output
        print(f"\nTotal Time Taken: {total_time:.4f} seconds\n")
        print(f"Pattern '{pattern}' found {total_count} times.")

if __name__ == "__main__":
    main()

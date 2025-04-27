# file: mpi_phonebook_search.py

from mpi4py import MPI
import time

def search_contacts(phonebook_chunk, search_name):
    matches = []
    for entry in phonebook_chunk:
        if entry[0].lower() == search_name.lower():
            matches.append(entry)
    return matches

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "phonebook.txt"
    search_name = "Alice"  # You can change this! --> write the word you want to search for in the phonebook

    if rank == 0:
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Preprocess into list of (name, number)
        phonebook = [line.strip().split() for line in lines]

        # Split phonebook among processes
        chunk_size = len(phonebook) // size
        chunks = [phonebook[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

        if len(chunks) > 0:
            chunks[-1].extend(phonebook[size*chunk_size:])
    else:
        chunks = None

    # Scatter chunks
    phonebook_chunk = comm.scatter(chunks, root=0)

    # Start timing
    start_time = MPI.Wtime()

    # Each process searches its chunk
    local_matches = search_contacts(phonebook_chunk, search_name)

    # Gather results at root
    all_matches = comm.gather(local_matches, root=0)

    if rank == 0:
        # Merge matches
        matches = []
        for match_list in all_matches:
            matches.extend(match_list)

        # End timing
        end_time = MPI.Wtime()
        total_time = end_time - start_time

        # Output
        print(f"\nTotal Time Taken: {total_time:.4f} seconds\n")
        print(f"Matches for name '{search_name}':")
        for name, number in matches:
            print(f"{name} {number}")

if __name__ == "__main__":
    main()

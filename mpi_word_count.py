# file: mpi_word_count.py

from mpi4py import MPI
from collections import Counter
import time
import re

def clean_and_split(text):
    # Remove punctuation and make lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Keep only words and spaces
    text = text.lower()
    words = text.split()
    return words

def count_words(word_list):
    return Counter(word_list)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "input.txt"

    if rank == 0:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean and split text into words
        words = clean_and_split(text)

        # Divide words into chunks
        chunk_size = len(words) // size
        chunks = [words[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

        # Last process gets remaining words
        if len(chunks) > 0:
            chunks[-1].extend(words[size*chunk_size:])
    else:
        chunks = None

    # Distribute word chunks to all processes
    word_chunk = comm.scatter(chunks, root=0)

    # Start timing
    start_time = MPI.Wtime()

    # Each process counts its own words
    local_counter = count_words(word_chunk)

    # Gather all local counters to the root process
    all_counters = comm.gather(local_counter, root=0)

    if rank == 0:
        # Merge all counters
        total_counter = Counter()
        for counter in all_counters:
            total_counter.update(counter)

        # Sort by frequency
        sorted_words = total_counter.most_common()

        # Stop timing
        end_time = MPI.Wtime()
        total_time = end_time - start_time

        # Display results
        print(f"\nTotal Time Taken: {total_time:.4f} seconds\n")
        print("Top 10 Words:")
        for word, freq in sorted_words[:10]:
            print(f"{word}: {freq}")

if __name__ == "__main__":
    main()

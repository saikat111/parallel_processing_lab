!pip install cupy-cuda12x
import cupy as cp
import numpy as np
import time
import re

def clean_text(text):
    return text.strip().lower()

def load_phonebook(filename):
    names = []
    numbers = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                names.append(parts[0])
                numbers.append(parts[1])
    return names, numbers

def search_phonebook_cuda(search_name, names, numbers):
    start_time = time.time()

    # Preprocessing
    names_lower = [clean_text(name) for name in names]
    search_name_lower = clean_text(search_name)

    # Encode names to bytes
    name_bytes = [name.encode('utf-8') for name in names_lower]
    max_len = max(len(name) for name in name_bytes)

    # Pad names to same length
    name_bytes_padded = [name.ljust(max_len, b'\0') for name in name_bytes]

    # Convert names into uint8 numpy array
    names_np = np.array([list(name) for name in name_bytes_padded], dtype=np.uint8)

    # Move names to GPU
    d_names = cp.asarray(names_np)

    # Prepare search name
    search_np = np.array(list(search_name_lower.encode('utf-8').ljust(max_len, b'\0')), dtype=np.uint8)

    d_search = cp.asarray(search_np)

    # Search on GPU
    matches = cp.all(d_names == d_search, axis=1)

    # Get matching indices
    match_indices = cp.where(matches)[0].get()

    end_time = time.time()

    # Output
    print(f"\nTotal Time Taken: {end_time - start_time:.4f} seconds\n")
    if len(match_indices) == 0:
        print(f"No matches found for '{search_name}'.")
    else:
        print(f"Matches for '{search_name}':")
        for idx in match_indices:
            print(f"{names[idx]} {numbers[idx]}")

# --------- TEST ------------

# Create sample phonebook
with open("phonebook.txt", "w") as f:
    f.write("""Alice 12345
Bob 67890
Charlie 54321
Alice 11111
Eve 99999
Bob 22222""")

# Load phonebook
names, numbers = load_phonebook("phonebook.txt")

# Search
search_name = "Alice"
search_phonebook_cuda(search_name, names, numbers)

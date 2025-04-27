!pip install cupy-cuda12x
import cupy as cp
import numpy as np
import re
import time

def clean_and_split(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    words = text.split()
    return words

def map_words_to_ints(words):
    word_to_int = {}
    int_to_word = {}
    current = 0
    for word in words:
        if word not in word_to_int:
            word_to_int[word] = current
            int_to_word[current] = word
            current += 1
    return word_to_int, int_to_word

def word_count_cupy(text):
    start_time = time.time()

    # Preprocessing
    words = clean_and_split(text)
    word_to_int, int_to_word = map_words_to_ints(words)
    mapped_words = np.array([word_to_int[word] for word in words], dtype=np.int32)

    # Move data to GPU
    d_mapped_words = cp.asarray(mapped_words)

    # Sort words on GPU
    d_mapped_words_sorted = cp.sort(d_mapped_words)

    # Use CuPy unique and counts
    unique_words, counts = cp.unique(d_mapped_words_sorted, return_counts=True)

    # Bring back to CPU
    unique_words = unique_words.get()
    counts = counts.get()

    result = {}
    for idx, word_idx in enumerate(unique_words):
        result[int_to_word[int(word_idx)]] = int(counts[idx])

    # Sort result by frequency
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()

    # Print results
    print(f"\nTotal Time Taken: {end_time - start_time:.4f} seconds\n")
    print("Top 10 Words:")
    for word, freq in sorted_result[:10]:
        print(f"{word}: {freq}")

# Sample Text
text = """
parallel programming can significantly reduce execution time and improve efficiency when solving large scale computational problems using multiple processors or computers working together in a coordinated way mpi is a standard for message passing in distributed memory systems enabling scalable and portable applications across different architectures
"""

# Run
word_count_cupy(text)

!pip install cupy-cuda12x
import cupy as cp
import numpy as np
import time

def load_paragraph(filename):
    with open(filename, 'r') as f:
        return f.read()

def search_pattern_cuda(paragraph, pattern):
    start_time = time.time()

    # Encode paragraph and pattern
    paragraph_bytes = paragraph.encode('utf-8')
    pattern_bytes = pattern.encode('utf-8')

    paragraph_np = np.frombuffer(paragraph_bytes, dtype=np.uint8)
    pattern_np = np.frombuffer(pattern_bytes, dtype=np.uint8)

    # Transfer to GPU
    d_paragraph = cp.asarray(paragraph_np)
    d_pattern = cp.asarray(pattern_np)

    pattern_len = len(d_pattern)
    paragraph_len = len(d_paragraph)

    # Create sliding windows (shape: [paragraph_len - pattern_len + 1, pattern_len])
    strided = cp.lib.stride_tricks.sliding_window_view(d_paragraph, pattern_len)

    # Compare each window with the pattern
    matches = cp.all(strided == d_pattern, axis=1)

    # Count matches
    total_matches = cp.sum(matches).item()

    end_time = time.time()

    # Output
    print(f"\nTotal Time Taken: {end_time - start_time:.4f} seconds\n")
    print(f"Pattern '{pattern}' found {total_matches} times.")

# --------- Test --------------

# Create paragraph file (optional if not already created)
with open("paragraph.txt", "w") as f:
    f.write("This is an example paragraph where %x% is a pattern. Sometimes %x% occurs again like %x%.")

# Load paragraph
paragraph = load_paragraph("paragraph.txt")

# Pattern to search
pattern = "%x%"

# Run GPU search
search_pattern_cuda(paragraph, pattern)

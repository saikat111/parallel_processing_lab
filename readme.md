

```
# Parallel Programming  (MPI and CUDA Solutions)

This repository contains solutions for common parallel programming problems using:
- (a) **MPI** (Message Passing Interface)
- (b) **CUDA** (GPU Programming)

Each problem is solved two ways: MPI for CPU parallelism and CUDA for GPU acceleration.

---

## 🛠 Setup Instructions

### 1. Install MPI (for MPI Programs)

🔗 Download and install **Microsoft MPI**:

👉 [Download MS-MPI](https://www.microsoft.com/en-us/download/details.aspx?id=105289)

- Install both **Redistributable** and **SDK**.
- Add the following path to your system **Environment Variables** → **Path**:

```
C:\Program Files\Microsoft MPI\Bin\
```

- Verify MPI is installed by running:

```bash
mpiexec
```
✅ You should see MPI usage instructions if installed correctly.

---

### 2. Install Python Libraries

Install required Python packages:

```bash
pip install mpi4py numpy
```

For CUDA (only if you have a GPU):

```bash
pip install cupy-cuda11x
```

**Or in Google Colab:**

```python
!pip install cupy-cuda12x
```

- In Colab: **Runtime > Change runtime type > GPU** (important).

---


## 🏃 How to Run

### Running MPI Codes (Locally)

Use the `mpiexec` command:

```bash
mpiexec -n <number_of_processes> python <file_name>.py
```

Example:

```bash
mpiexec -n 4 python mpi_word_count.py
```

✅ Make sure MPI is installed and working.

---

### Running CUDA Codes (Colab or GPU PC)

In Colab or PC:

- Install CuPy first:

```bash
!pip install cupy-cuda12x
```

- Then run the `.py` files normally:

```python
python cuda_word_count.py
```

✅ Ensure your runtime has GPU enabled.

---

## 📋 Example Inputs and Outputs

---

### 1. Word Count (mpi_word_count.py / cuda_word_count.py)

**Input File: `input.txt`**

```
hello world hello mpi mpi mpi
```

**Output:**

```
Total Time Taken: 0.0021 seconds

Top 10 Words:
mpi: 3
hello: 2
world: 1
```

---

### 2. Phonebook Search (mpi_phonebook_search.py / cuda_phonebook_search.py)

**Input File: `phonebook.txt`**

```
Alice 12345
Bob 67890
Charlie 54321
Alice 11111
Eve 99999
```

**Searching for:** `Alice`

**Output:**

```
Total Time Taken: 0.0019 seconds

Matches for name 'Alice':
Alice 12345
Alice 11111
```

---

### 3. Pattern Search (mpi_pattern_search.py / cuda_pattern_search.py)

**Input File: `paragraph.txt`**

```
This is an example %x% paragraph with %x% inside.
```

**Searching for Pattern:** `%x%`

**Output:**

```
Total Time Taken: 0.0024 seconds

Pattern '%x%' found 2 times.
```

---

### 4. Matrix Multiplication (mpi_matrix_multiplication.py / cuda_matrix_multiplication.py)

**Inputs:**

- Number of matrices (K) = 100
- Matrix A size: 50×50
- Matrix B size: 50×50

**Output:**

```
Total Time Taken: 0.5234 seconds

Total 100 matrices multiplied.
```

(For CUDA version, time will be even faster.)

---

## 📚 Notes for Students

- All MPI programs **must** be run using `mpiexec`.
- CUDA codes are easier to run on **Google Colab** with GPU.
- CUDA is faster for large data; MPI is useful for multi-core CPU operations.
- Carefully follow input file formats (like phonebook.txt, input.txt, paragraph.txt).

✅ All codes are **tested and working** in local PC (MPI) and Google Colab (CUDA).

---

## 📎 References

- [MS-MPI Official Download](https://www.microsoft.com/en-us/download/details.aspx?id=105289)
- [mpi4py Documentation](https://mpi4py.readthedocs.io/en/stable/)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)

---

# 🚀 Good Luck with Parallel Programming!
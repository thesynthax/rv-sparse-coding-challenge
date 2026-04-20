# RV-Sparse: Sparse Matrix-Vector Multiplication
RISC-V LFX Mentorship: RV-sparse project coding challenge

## Problem Statement

Implement the `sparse_multiply` function that:

1. Scans a row-major dense matrix **`A`** and identifies its non-zero elements
2. Extracts them into Compressed Sparse Row (CSR) format using caller-provided buffers
3. Computes the matrix-vector product **`y = A * x`** using the extracted CSR data
4. Writes the result directly into a caller-provided output buffer

## Constraints

- **Zero dynamic memory allocation**: All memory is pre-allocated by the caller
- Function must work with any matrix dimensions
- Result must match reference dense multiplication exactly

## CSR Format

Compressed Sparse Row (CSR) format stores sparse matrices using three arrays:

| Array | Description |
|-------|-------------|
| `values` | Non-zero matrix elements, stored contiguously row-by-row |
| `col_indices` | Column index for each value in `values` |
| `row_ptrs` | Row start offsets in `values` (size: rows + 1) |

### Example

Matrix A (3x4):
```
[1  0  0  2]
[0  3  0  0]
[0  0  4  5]
```

CSR representation:
```
values      = [1, 2, 3, 4, 5]
col_indices= [0, 3, 1, 2, 3]
row_ptrs   = [0, 2, 3, 5]
```

- Row 0: `values[0:2]` = columns 0 and 3
- Row 1: `values[2:3]` = column 1
- Row 2: `values[3:5]` = columns 2 and 3

## Function Signature

```c
void sparse_multiply(
    int rows, int cols,
    const double* A,          // Input: dense matrix (row-major)
    const double* x,           // Input: dense vector
    int* out_nnz,             // Output: number of non-zeros
    double* values,           // Output: CSR non-zero values
    int* col_indices,         // Output: CSR column indices
    int* row_ptrs,            // Output: CSR row pointers
    double* y                 // Output: result vector y = A * x
);
```

## Approach and Implementation

The function uses a **single-pass algorithm** that builds CSR format and performs SpMV simultaneously:

```c
row_ptrs[0] = 0;

for (int i = 0; i < rows; ++i) {
    y[i] = 0.0;
    const double *row = &A[i * cols];

    for (int j = 0; j < cols; ++j) {
        double val = row[j];

        if (val != 0.0) {
            values[*out_nnz] = val;
            col_indices[*out_nnz] = j;
            (*out_nnz)++;

            y[i] += val * x[j];
        }
    }

    row_ptrs[i + 1] = *out_nnz;
}
```

**How it works:**
1. Initializes `row_ptrs[0] = 0`
2. For each row, caches the row pointer to avoid repeated indexing
3. Scans each element, populating CSR arrays and accumulating into `y[i]` in the same pass
4. Updates `row_ptrs[i+1]` to mark where next row starts

## Optimizations

1. **Pointer caching**: Cache `&A[i * cols]` once per row (line 34) — avoids computing `i * cols` and pointer arithmetic in the inner loop
2. **Direct array access**: Use `row[j]` instead of `A[i * cols + j]` — cleaner, same performance
3. **Single-pass algorithm**: Builds CSR and computes y simultaneously — only one scan of the matrix instead of two

## Building and Running

```bash
gcc -lm -o run challenge.c
./run
```

## Expected Output

```
Iter   0 [ 20x 30, density=0.38, nnz= 242]: PASS (Max error: 0.00e+00)
Iter   1 [ 33x 35, density=0.18, nnz= 191]: PASS (Max error: 0.00e+00)
Iter   2 [ 24x 12, density=0.21, nnz=  63]: PASS (Max error: 0.00e+00)
Iter   3 [ 42x 31, density=0.05, nnz=  68]: PASS (Max error: 0.00e+00)
Iter   4 [ 28x  8, density=0.23, nnz=  52]: PASS (Max error: 0.00e+00)
Iter   5 [ 31x 21, density=0.38, nnz= 243]: PASS (Max error: 0.00e+00)
...
Iter  99 [ 39x 41, density=0.31, nnz= 498]: PASS (Max error: 0.00e+00)
All tests passed!
```

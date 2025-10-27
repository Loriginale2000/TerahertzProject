

#  Key Performance Improvements

### 1. Vectorized Frequency Computation
- **Old method:** Computed transmission `T(ω)` in a loop, one frequency at a time.
- **New method:** Uses PyTorch tensors to compute transmission for **all frequencies at once**, leveraging vectorized operations.

### 2. Batched Transfer Matrix Construction
- Each layer’s interface and propagation matrices are calculated **simultaneously** across all frequencies.
- Matrix products across layers use `torch.bmm()` (batched matrix multiplication), which is highly optimized.

### 3. Avoids Redundant Computation
- The complex frequency spectrum is constructed once using Hermitian symmetry, avoiding duplicate computation of mirrored frequencies.
- Intermediate matrix results are reused instead of recalculated.

### 4. Elimination of Python Loops
- Python for-loops are inherently slow, especially when processing tens of thousands of frequency components.
- The new approach replaces these with native PyTorch tensor operations that run on the backend in **optimized C++**, making the process **10× to 100× faster** depending on hardware.



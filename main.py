import numpy as np
from fractions import Fraction

# ----------------------------
# Multiplication counter model
# ----------------------------
MULT_COUNT = 0

class MulCounter:
    """
    Counts ONLY bilinear multiplications (MulCounter * MulCounter).
    Multiplication/division by constants is treated as free (rank model).
    Values are stored exactly as Fractions when possible, or as complex numbers.
    """
    __slots__ = ("val",)

    def __init__(self, val):
        if isinstance(val, MulCounter):
            self.val = val.val
        elif isinstance(val, Fraction):
            self.val = val
        elif isinstance(val, complex) or (isinstance(val, (int, float)) and isinstance(getattr(val, 'imag', 0), (int, float)) and getattr(val, 'imag', 0) != 0):
            # Store complex values as complex
            self.val = complex(val)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            # Try to store as Fraction for exact arithmetic
            try:
                self.val = Fraction(val).limit_denominator(10**12)
            except (ValueError, TypeError):
                self.val = complex(val)
        else:
            try:
                self.val = Fraction(val)
            except (ValueError, TypeError):
                self.val = complex(val)

    def _v(self, other):
        if isinstance(other, MulCounter):
            return other.val
        if isinstance(other, complex):
            return other
        try:
            return Fraction(other)
        except (ValueError, TypeError):
            return complex(other)

    def _combine(self, result):
        """Wrap result in MulCounter, handling type appropriately."""
        return MulCounter(result)

    def __add__(self, other):  return self._combine(self.val + self._v(other))
    def __radd__(self, other): return self._combine(self._v(other) + self.val)
    def __sub__(self, other):  return self._combine(self.val - self._v(other))
    def __rsub__(self, other): return self._combine(self._v(other) - self.val)

    def __mul__(self, other):
        global MULT_COUNT
        if isinstance(other, MulCounter):
            MULT_COUNT += 1
            return self._combine(self.val * other.val)
        return self._combine(self.val * self._v(other))

    def __rmul__(self, other): 
        # Scalar * MulCounter - does NOT count as bilinear multiplication
        return self._combine(self._v(other) * self.val)
    
    def __truediv__(self, other): return self._combine(self.val / self._v(other))
    def __neg__(self): return MulCounter(-self.val)

    def __repr__(self): return str(self.val)


# ----------------------------
# 4x4 "Rosowski-style" (46 mults)
# ----------------------------
def rosowski_4x4_commutative(A, B):
    """
    4x4 matrix multiplication using a 46-multiplication scheme over a commutative ring.
    (Counts only bilinear multiplies via MulCounter.)
    """
    n = 4
    half = 2
    C = np.zeros((n, n), dtype=object)

    # Precompute Q terms (involves B)
    Q = np.zeros((half, n), dtype=object)
    for k in range(half):
        odd = 2 * k
        even = 2 * k + 1
        for j in range(1, n):
            Q[k, j] = B[even, j] * (B[odd, 0] + B[odd, j])

    # Row-wise computations
    for i in range(n):
        # P terms (involves A and B col 0)
        P = [None] * half
        for k in range(half):
            odd = 2 * k
            P[k] = A[i, odd] * (B[odd, 0] + A[i, odd + 1])

        # Column 0
        s = 0
        for k in range(half):
            odd = 2 * k
            even = 2 * k + 1
            term_R = A[i, even] * (B[even, 0] - A[i, odd])
            s += P[k] + term_R
        C[i, 0] = s

        # Remaining columns
        for j in range(1, n):
            t = 0
            for k in range(half):
                odd = 2 * k
                even = 2 * k + 1
                term_M = (A[i, odd] + B[even, j]) * (A[i, even] + B[odd, 0] + B[odd, j])
                t += term_M - P[k] - Q[k, j]
            C[i, j] = t

    return C


# ----------------------------
# Block wrapper: makes * mean 4x4 matrix multiply (Rosowski) not elementwise
# ----------------------------
class Block:
    """
    4x4 block whose entries are MulCounter (or plain Fractions/ints).
    - Block + Block, Block - Block are elementwise.
    - Block * Block means 4x4 matrix multiplication via rosowski_4x4_commutative.
    - Scaling by constants is elementwise and does NOT increment MULT_COUNT.
    """
    __slots__ = ("M",)

    def __init__(self, M):
        self.M = np.array(M, dtype=object)
        assert self.M.shape == (4, 4)

    def __add__(self, other):
        if isinstance(other, Block):
            return Block(self.M + other.M)
        return Block(self.M + other)

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Block):
            return Block(self.M - other.M)
        return Block(self.M - other)

    def __rsub__(self, other):
        if isinstance(other, Block):
            return Block(other.M - self.M)
        return Block(other - self.M)

    def __mul__(self, other):
        if isinstance(other, Block):
            return Block(rosowski_4x4_commutative(self.M, other.M))
        return Block(self.M * other)  # scalar scale

    def __rmul__(self, other): return Block(other * self.M)  # scalar scale
    def __truediv__(self, other): return Block(self.M / other)

    def to_array(self): return self.M

    def __repr__(self): return f"Block({self.M})"


# ----------------------------
# Rank-48 4x4 algorithm (SLP from paper, with tested correction)
# ----------------------------
def alphaevolve_rank48_4x4(A, B):
    """
    AlphaEvolve's optimized algorithm for 4×4 matrices.
    Uses exactly 48 scalar multiplications.
    """
    # Check if we're dealing with MulCounter or Block objects (need object dtype)
    sample = A[0, 0]
    use_object_dtype = isinstance(sample, (MulCounter, Block))
    
    # Check if we're dealing with real matrices for potential optimizations
    is_real_input = not use_object_dtype and np.isrealobj(A) and np.isrealobj(B)
    
    # Initialize the result matrix - use appropriate dtype
    out_dtype = object if use_object_dtype else np.complex128
    C = np.zeros((4, 4), dtype=out_dtype)
    
    # Cache commonly used constants
    half = 0.5
    half_j = 0.5j
    half_p_half_j = 0.5 + 0.5j
    half_m_half_j = 0.5 - 0.5j
    neg_half = -0.5
    neg_half_j = -0.5j
    
    # Cache matrix values to avoid repeated memory access
    A00, A01, A02, A03 = A[0,0], A[0,1], A[0,2], A[0,3]
    A10, A11, A12, A13 = A[1,0], A[1,1], A[1,2], A[1,3]
    A20, A21, A22, A23 = A[2,0], A[2,1], A[2,2], A[2,3]
    A30, A31, A32, A33 = A[3,0], A[3,1], A[3,2], A[3,3]
    
    B00, B01, B02, B03 = B[0,0], B[0,1], B[0,2], B[0,3]
    B10, B11, B12, B13 = B[1,0], B[1,1], B[1,2], B[1,3]
    B20, B21, B22, B23 = B[2,0], B[2,1], B[2,2], B[2,3]
    B30, B31, B32, B33 = B[3,0], B[3,1], B[3,2], B[3,3]
    
    # Linear combinations of elements from A - computed once and stored
    a0 = half_p_half_j*A00 + half_p_half_j*A01 + half_m_half_j*A10 + half_m_half_j*A11 + half_m_half_j*A20 + half_m_half_j*A21 + half_m_half_j*A30 + half_m_half_j*A31
    a1 = half_p_half_j*A00 + (neg_half+half_j)*A03 + half_p_half_j*A10 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A20 + half_m_half_j*A23 + half_m_half_j*A30 + half_p_half_j*A33
    a2 = neg_half*A01 + half*A02 + neg_half_j*A11 + half_j*A12 + half_j*A21 + neg_half_j*A22 + neg_half_j*A31 + half_j*A32
    a3 = neg_half_j*A00 + neg_half*A01 + half*A02 + neg_half*A03 + half_j*A10 + neg_half*A11 + half*A12 + half*A13 + neg_half_j*A20 + neg_half*A21 + half*A22 + neg_half*A23 + neg_half*A30 + neg_half_j*A31 + half_j*A32 + half_j*A33
    a4 = half_p_half_j*A00 + (neg_half+neg_half_j)*A01 + (neg_half+half_j)*A10 + half_m_half_j*A11 + (neg_half+half_j)*A20 + half_m_half_j*A21 + half_m_half_j*A30 + (neg_half+half_j)*A31
    a5 = half_m_half_j*A02 + (neg_half+neg_half_j)*A03 + half_m_half_j*A12 + (neg_half+neg_half_j)*A13 + (neg_half+half_j)*A22 + half_p_half_j*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a6 = half_j*A00 + half*A03 + neg_half*A10 + half_j*A13 + half*A20 + neg_half_j*A23 + neg_half*A30 + half_j*A33
    a7 = half_p_half_j*A00 + (neg_half+neg_half_j)*A01 + (neg_half+neg_half_j)*A10 + half_p_half_j*A11 + (neg_half+neg_half_j)*A20 + half_p_half_j*A21 + (neg_half+half_j)*A30 + half_m_half_j*A31
    a8 = neg_half_j*A00 + neg_half_j*A01 + neg_half*A02 + neg_half_j*A03 + half*A10 + half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + neg_half*A21 + neg_half_j*A22 + half*A23 + half*A30 + half*A31 + half_j*A32 + neg_half*A33
    a9 = (neg_half+half_j)*A00 + (neg_half+neg_half_j)*A03 + half_p_half_j*A10 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A20 + half_m_half_j*A23 + (neg_half+neg_half_j)*A30 + half_m_half_j*A33
    a10 = (neg_half+half_j)*A00 + half_m_half_j*A01 + (neg_half+half_j)*A10 + half_m_half_j*A11 + half_m_half_j*A20 + (neg_half+half_j)*A21 + half_p_half_j*A30 + (neg_half+neg_half_j)*A31
    
    # Continue with the remaining a values
    a11 = half*A00 + half*A01 + neg_half_j*A02 + neg_half*A03 + neg_half*A10 + neg_half*A11 + half_j*A12 + half*A13 + half*A20 + half*A21 + half_j*A22 + half*A23 + neg_half_j*A30 + neg_half_j*A31 + half*A32 + neg_half_j*A33
    a12 = half_p_half_j*A01 + (neg_half+neg_half_j)*A02 + (neg_half+half_j)*A11 + half_m_half_j*A12 + (neg_half+half_j)*A21 + half_m_half_j*A22 + half_m_half_j*A31 + (neg_half+half_j)*A32
    a13 = half_m_half_j*A01 + (neg_half+half_j)*A02 + half_m_half_j*A11 + (neg_half+half_j)*A12 + half_m_half_j*A21 + (neg_half+half_j)*A22 + half_p_half_j*A31 + (neg_half+neg_half_j)*A32
    a14 = half_j*A00 + neg_half*A01 + half*A02 + neg_half*A03 + half*A10 + neg_half_j*A11 + half_j*A12 + half_j*A13 + half*A20 + half_j*A21 + neg_half_j*A22 + half_j*A23 + half*A30 + neg_half_j*A31 + half_j*A32 + half_j*A33
    a15 = (neg_half+half_j)*A02 + half_p_half_j*A03 + half_m_half_j*A12 + (neg_half+neg_half_j)*A13 + half_m_half_j*A22 + (neg_half+neg_half_j)*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a16 = neg_half*A00 + half_j*A01 + half_j*A02 + neg_half_j*A03 + neg_half*A10 + neg_half_j*A11 + neg_half_j*A12 + neg_half_j*A13 + neg_half*A20 + half_j*A21 + half_j*A22 + neg_half_j*A23 + neg_half_j*A30 + half*A31 + half*A32 + half*A33
    a17 = half_p_half_j*A00 + half_p_half_j*A01 + half_p_half_j*A10 + half_p_half_j*A11 + half_p_half_j*A20 + half_p_half_j*A21 + (neg_half+half_j)*A30 + (neg_half+half_j)*A31
    a18 = half_j*A00 + half_j*A01 + neg_half*A02 + half_j*A03 + half_j*A10 + half_j*A11 + neg_half*A12 + half_j*A13 + half_j*A20 + half_j*A21 + half*A22 + neg_half_j*A23 + neg_half*A30 + neg_half*A31 + half_j*A32 + half*A33
    a19 = half_m_half_j*A02 + half_p_half_j*A03 + half_m_half_j*A12 + half_p_half_j*A13 + half_m_half_j*A22 + half_p_half_j*A23 + half_p_half_j*A32 + (neg_half+half_j)*A33
    a20 = half_p_half_j*A01 + (neg_half+neg_half_j)*A02 + half_p_half_j*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + half_p_half_j*A22 + half_m_half_j*A31 + (neg_half+half_j)*A32
    
    # Complete a21 to a47
    a21 = half_j*A00 + neg_half_j*A01 + neg_half*A02 + neg_half_j*A03 + neg_half_j*A10 + half_j*A11 + half*A12 + half_j*A13 + neg_half_j*A20 + half_j*A21 + neg_half*A22 + neg_half_j*A23 + neg_half*A30 + half*A31 + half_j*A32 + neg_half*A33
    a22 = (neg_half+neg_half_j)*A00 + (neg_half+half_j)*A03 + half_m_half_j*A10 + (neg_half+neg_half_j)*A13 + half_m_half_j*A20 + (neg_half+neg_half_j)*A23 + (neg_half+half_j)*A30 + half_p_half_j*A33
    a23 = (neg_half+neg_half_j)*A02 + half_m_half_j*A03 + half_m_half_j*A12 + half_p_half_j*A13 + half_m_half_j*A22 + half_p_half_j*A23 + (neg_half+half_j)*A32 + (neg_half+neg_half_j)*A33
    a24 = neg_half*A00 + half*A01 + neg_half_j*A02 + neg_half*A03 + neg_half_j*A10 + half_j*A11 + half*A12 + neg_half_j*A13 + neg_half_j*A20 + half_j*A21 + neg_half*A22 + half_j*A23 + half_j*A30 + neg_half_j*A31 + half*A32 + neg_half_j*A33
    a25 = half_m_half_j*A02 + half_p_half_j*A03 + (neg_half+neg_half_j)*A12 + half_m_half_j*A13 + half_p_half_j*A22 + (neg_half+half_j)*A23 + half_p_half_j*A32 + (neg_half+half_j)*A33
    a26 = half_p_half_j*A01 + half_p_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + half_p_half_j*A21 + half_p_half_j*A22 + half_m_half_j*A31 + half_m_half_j*A32
    a27 = neg_half_j*A00 + neg_half_j*A01 + half*A02 + half_j*A03 + neg_half*A10 + neg_half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + neg_half*A21 + half_j*A22 + neg_half*A23 + neg_half*A30 + neg_half*A31 + half_j*A32 + neg_half*A33
    a28 = (neg_half+half_j)*A00 + (neg_half+half_j)*A01 + (neg_half+neg_half_j)*A10 + (neg_half+neg_half_j)*A11 + half_p_half_j*A20 + half_p_half_j*A21 + (neg_half+neg_half_j)*A30 + (neg_half+neg_half_j)*A31
    a29 = half_p_half_j*A00 + half_m_half_j*A03 + (neg_half+neg_half_j)*A10 + (neg_half+half_j)*A13 + half_p_half_j*A20 + half_m_half_j*A23 + half_m_half_j*A30 + (neg_half+neg_half_j)*A33
    a30 = half_p_half_j*A01 + half_p_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + (neg_half+neg_half_j)*A22 + (neg_half+half_j)*A31 + (neg_half+half_j)*A32
    a31 = half*A00 + neg_half*A01 + neg_half_j*A02 + half*A03 + half*A10 + neg_half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + half*A21 + neg_half_j*A22 + half*A23 + neg_half_j*A30 + half_j*A31 + half*A32 + half_j*A33
    a32 = half_p_half_j*A02 + half_m_half_j*A03 + (neg_half+half_j)*A12 + half_p_half_j*A13 + half_m_half_j*A22 + (neg_half+neg_half_j)*A23 + (neg_half+half_j)*A32 + half_p_half_j*A33
    a33 = half*A00 + half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + neg_half*A10 + half_j*A11 + neg_half_j*A12 + half_j*A13 + neg_half*A20 + neg_half_j*A21 + half_j*A22 + half_j*A23 + half_j*A30 + half*A31 + neg_half*A32 + half*A33
    a34 = neg_half_j*A00 + half_j*A01 + neg_half*A02 + half_j*A03 + neg_half*A10 + half*A11 + half_j*A12 + half*A13 + half*A20 + neg_half*A21 + half_j*A22 + half*A23 + half*A30 + neg_half*A31 + half_j*A32 + half*A33
    a35 = half_m_half_j*A02 + half_p_half_j*A03 + (neg_half+half_j)*A12 + (neg_half+neg_half_j)*A13 + half_m_half_j*A22 + half_p_half_j*A23 + (neg_half+neg_half_j)*A32 + half_m_half_j*A33
    a36 = (neg_half+neg_half_j)*A01 + (neg_half+neg_half_j)*A02 + (neg_half+half_j)*A11 + (neg_half+half_j)*A12 + half_m_half_j*A21 + half_m_half_j*A22 + half_m_half_j*A31 + half_m_half_j*A32
    a37 = half*A00 + neg_half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + half_j*A10 + neg_half*A11 + neg_half*A12 + half*A13 + half_j*A20 + half*A21 + half*A22 + half*A23 + neg_half_j*A30 + half*A31 + half*A32 + neg_half*A33
    a38 = half_m_half_j*A01 + half_m_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + (neg_half+neg_half_j)*A22 + (neg_half+neg_half_j)*A31 + (neg_half+neg_half_j)*A32
    a39 = neg_half*A00 + neg_half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + neg_half*A10 + half_j*A11 + half_j*A12 + neg_half_j*A13 + half*A20 + half_j*A21 + half_j*A22 + half_j*A23 + half_j*A30 + half*A31 + half*A32 + neg_half*A33
    a40 = (neg_half+neg_half_j)*A00 + (neg_half+neg_half_j)*A01 + half_p_half_j*A10 + half_p_half_j*A11 + (neg_half+neg_half_j)*A20 + (neg_half+neg_half_j)*A21 + (neg_half+half_j)*A30 + (neg_half+half_j)*A31
    a41 = half_m_half_j*A00 + (neg_half+neg_half_j)*A03 + (neg_half+half_j)*A10 + half_p_half_j*A13 + (neg_half+half_j)*A20 + half_p_half_j*A23 + half_p_half_j*A30 + half_m_half_j*A33
    a42 = half_p_half_j*A00 + (neg_half+half_j)*A03 + half_m_half_j*A10 + half_p_half_j*A13 + half_m_half_j*A20 + half_p_half_j*A23 + half_m_half_j*A30 + half_p_half_j*A33
    a43 = half_j*A00 + half*A01 + neg_half*A02 + neg_half*A03 + half*A10 + half_j*A11 + neg_half_j*A12 + half_j*A13 + neg_half*A20 + half_j*A21 + neg_half_j*A22 + neg_half_j*A23 + neg_half*A30 + neg_half_j*A31 + half_j*A32 + neg_half_j*A33
    a44 = half_m_half_j*A02 + (neg_half+neg_half_j)*A03 + (neg_half+neg_half_j)*A12 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A22 + (neg_half+half_j)*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a45 = (neg_half+half_j)*A00 + half_m_half_j*A01 + half_p_half_j*A10 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A20 + half_p_half_j*A21 + (neg_half+neg_half_j)*A30 + half_p_half_j*A31
    a46 = half_m_half_j*A00 + half_p_half_j*A03 + half_m_half_j*A10 + half_p_half_j*A13 + half_m_half_j*A20 + half_p_half_j*A23 + half_p_half_j*A30 + (neg_half+half_j)*A33
    a47 = half*A00 + half_j*A01 + half_j*A02 + neg_half_j*A03 + half_j*A10 + half*A11 + half*A12 + half*A13 + neg_half_j*A20 + half*A21 + half*A22 + neg_half*A23 + half_j*A30 + half*A31 + half*A32 + half*A33
    
    # Linear combinations of elements from B (optimized)
    b0 = neg_half*B00 + neg_half*B10 + half*B20 + neg_half_j*B30
    b1 = half_j*B01 + half_j*B03 + half_j*B11 + half_j*B13 + half_j*B21 + half_j*B23 + half*B31 + half*B33
    b2 = half_p_half_j*B01 + (neg_half+neg_half_j)*B11 + half_p_half_j*B21 + half_m_half_j*B31
    b3 = neg_half_j*B00 + half_j*B02 + neg_half_j*B11 + neg_half_j*B12 + half_j*B21 + half_j*B22 + half*B30 + neg_half*B32
    b4 = neg_half*B00 + half*B02 + half*B03 + half*B10 + neg_half*B12 + neg_half*B13 + half*B20 + neg_half*B22 + neg_half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b5 = half*B01 + half*B03 + half*B11 + half*B13 + half*B21 + half*B23 + half_j*B31 + half_j*B33
    b6 = (neg_half+neg_half_j)*B01 + half_p_half_j*B11 + half_p_half_j*B21 + half_m_half_j*B31
    b7 = neg_half*B00 + half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    
    # Continue with b8 to b47
    b8 = half*B00 + neg_half*B02 + neg_half*B03 + half*B10 + neg_half*B12 + neg_half*B13 + half*B21 + neg_half_j*B31
    b9 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + half*B31 + half*B32 + half*B33
    b10 = half_j*B01 + half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + neg_half_j*B21 + neg_half_j*B23 + neg_half*B31 + neg_half*B33
    b11 = neg_half_j*B00 + half_j*B03 + neg_half_j*B10 + half_j*B13 + half_j*B21 + half_j*B22 + neg_half*B31 + neg_half*B32
    b12 = neg_half*B00 + half*B02 + half*B03 + neg_half*B10 + half*B12 + half*B13 + half*B20 + neg_half*B22 + neg_half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b13 = half_j*B00 + neg_half_j*B02 + neg_half_j*B10 + half_j*B12 + half_j*B20 + neg_half_j*B22 + neg_half*B30 + half*B32
    b14 = neg_half*B01 + neg_half*B10 + half*B20 + half_j*B31
    b15 = half_j*B00 + neg_half_j*B03 + half_j*B10 + neg_half_j*B13 + neg_half_j*B20 + half_j*B23 + half*B30 + neg_half*B33
    b16 = half*B01 + half*B02 + half*B10 + neg_half*B12 + half*B20 + neg_half*B22 + neg_half_j*B31 + neg_half_j*B32
    b17 = neg_half_j*B00 + half_j*B02 + neg_half_j*B10 + half_j*B12 + neg_half_j*B20 + half_j*B22 + half*B30 + neg_half*B32
    b18 = neg_half_j*B01 + neg_half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + neg_half_j*B20 + half_j*B22 + half*B30 + neg_half*B32
    b19 = neg_half_j*B00 + half_j*B02 + half_j*B10 + neg_half_j*B12 + half_j*B20 + neg_half_j*B22 + half*B30 + neg_half*B32
    b20 = neg_half_j*B01 + neg_half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + half_j*B21 + half_j*B23 + half*B31 + half*B33
    
    # Complete b21 to b47
    b21 = neg_half*B01 + neg_half*B02 + half*B11 + half*B12 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b22 = neg_half_j*B00 + half_j*B02 + half_j*B03 + neg_half_j*B10 + half_j*B12 + half_j*B13 + neg_half_j*B20 + half_j*B22 + half_j*B23 + half*B30 + neg_half*B32 + neg_half*B33
    b23 = neg_half*B00 + half*B02 + half*B03 + neg_half*B10 + half*B12 + half*B13 + neg_half*B20 + half*B22 + half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b24 = half_j*B01 + neg_half_j*B11 + neg_half_j*B20 + half_j*B22 + half_j*B23 + half*B30 + neg_half*B32 + neg_half*B33
    b25 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b26 = half*B01 + half*B02 + neg_half*B11 + neg_half*B12 + neg_half*B21 + neg_half*B22 + neg_half_j*B31 + neg_half_j*B32
    b27 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B20 + neg_half*B30
    b28 = half*B01 + half*B11 + half*B21 + neg_half_j*B31
    b29 = half_j*B01 + half_j*B02 + neg_half_j*B11 + neg_half_j*B12 + half_j*B21 + half_j*B22 + neg_half*B31 + neg_half*B32
    b30 = neg_half*B00 + half*B03 + neg_half*B10 + half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b31 = half*B00 + neg_half*B02 + neg_half*B10 + half*B12 + neg_half*B21 + neg_half*B23 + half_j*B31 + half_j*B33
    b32 = half_j*B01 + neg_half_j*B11 + neg_half_j*B21 + half*B31
    b33 = neg_half*B01 + neg_half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + neg_half_j*B31 + neg_half_j*B33
    b34 = half_j*B00 + neg_half_j*B10 + half_j*B21 + half_j*B22 + half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b35 = neg_half_j*B01 + neg_half_j*B02 + half_j*B11 + half_j*B12 + neg_half_j*B21 + neg_half_j*B22 + neg_half*B31 + neg_half*B32
    b36 = neg_half*B01 + neg_half*B02 + neg_half*B03 + neg_half*B11 + neg_half*B12 + neg_half*B13 + neg_half*B21 + neg_half*B22 + neg_half*B23 + neg_half_j*B31 + neg_half_j*B32 + neg_half_j*B33
    b37 = half_j*B01 + half_j*B02 + half_j*B03 + neg_half_j*B10 + half_j*B12 + half_j*B13 + neg_half_j*B20 + half_j*B22 + half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b38 = half_j*B00 + neg_half_j*B10 + neg_half_j*B20 + neg_half*B30
    b39 = neg_half_j*B00 + half_j*B03 + half_j*B11 + half_j*B13 + half_j*B21 + half_j*B23 + neg_half*B30 + half*B33
    b40 = half_j*B01 + half_j*B02 + half_j*B11 + half_j*B12 + neg_half_j*B21 + neg_half_j*B22 + half*B31 + half*B32
    b41 = half*B00 + neg_half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b42 = half_j*B00 + neg_half_j*B10 + half_j*B20 + half*B30
    b43 = half*B00 + neg_half*B02 + neg_half*B03 + neg_half*B11 + neg_half*B12 + neg_half*B13 + half*B21 + half*B22 + half*B23 + neg_half_j*B30 + half_j*B32 + half_j*B33
    b44 = neg_half_j*B00 + half_j*B10 + neg_half_j*B20 + half*B30
    b45 = neg_half_j*B01 + neg_half_j*B02 + neg_half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + half*B31 + half*B32 + half*B33
    b46 = neg_half*B00 + half*B02 + half*B10 + neg_half*B12 + half*B20 + neg_half*B22 + half_j*B30 + neg_half_j*B32
    b47 = half*B00 + half*B11 + half*B21 + half_j*B30
    
    # Perform the 48 multiplications efficiently
    m = np.zeros(48, dtype=out_dtype)
    
    # We can directly compute these multiplications
    # Numbering is maintained for clarity
    m[0] = a0 * b0
    m[1] = a1 * b1
    m[2] = a2 * b2
    m[3] = a3 * b3
    m[4] = a4 * b4
    m[5] = a5 * b5
    m[6] = a6 * b6
    m[7] = a7 * b7
    m[8] = a8 * b8
    m[9] = a9 * b9
    m[10] = a10 * b10
    m[11] = a11 * b11
    m[12] = a12 * b12
    m[13] = a13 * b13
    m[14] = a14 * b14
    m[15] = a15 * b15
    m[16] = a16 * b16
    m[17] = a17 * b17
    m[18] = a18 * b18
    m[19] = a19 * b19
    m[20] = a20 * b20
    m[21] = a21 * b21
    m[22] = a22 * b22
    m[23] = a23 * b23
    m[24] = a24 * b24
    m[25] = a25 * b25
    m[26] = a26 * b26
    m[27] = a27 * b27
    m[28] = a28 * b28
    m[29] = a29 * b29
    m[30] = a30 * b30
    m[31] = a31 * b31
    m[32] = a32 * b32
    m[33] = a33 * b33
    m[34] = a34 * b34
    m[35] = a35 * b35
    m[36] = a36 * b36
    m[37] = a37 * b37
    m[38] = a38 * b38
    m[39] = a39 * b39
    m[40] = a40 * b40
    m[41] = a41 * b41
    m[42] = a42 * b42
    m[43] = a43 * b43
    m[44] = a44 * b44
    m[45] = a45 * b45
    m[46] = a46 * b46
    m[47] = a47 * b47
    
    # Construct the result matrix efficiently
    # For C[0,0]
    C[0,0] = half_j*m[0] + neg_half_j*m[1] + neg_half*m[5] + half*m[8] + half_j*m[9] + \
             (neg_half+half_j)*m[11] + half*m[14] + neg_half_j*m[15] + (neg_half+neg_half_j)*m[16] + \
             half_j*m[17] + (neg_half+neg_half_j)*m[18] + neg_half_j*m[24] + half_j*m[26] + \
             half_j*m[27] + half*m[28] + half_j*m[30] + neg_half_j*m[32] + half*m[34] + \
             half*m[36] + neg_half_j*m[37] + neg_half*m[38] + (half+neg_half_j)*m[39] + \
             neg_half_j*m[40] + neg_half*m[42] + neg_half*m[43] + neg_half*m[44] + \
             neg_half_j*m[46] + half*m[47]
    
    # For C[0,1]
    C[0,1] = neg_half_j*m[0] + half*m[2] + (neg_half+neg_half_j)*m[3] + half*m[5] + \
             half*m[6] + neg_half*m[8] + (half+neg_half_j)*m[11] + neg_half*m[12] + \
             half_j*m[13] + half_j*m[14] + half_j*m[15] + neg_half_j*m[17] + \
             (half+half_j)*m[18] + half*m[20] + neg_half*m[22] + half_j*m[24] + \
             neg_half_j*m[27] + neg_half*m[28] + neg_half_j*m[29] + half_j*m[32] + \
             (neg_half+neg_half_j)*m[33] + neg_half*m[34] + neg_half*m[37] + half_j*m[40] + \
             half_j*m[41] + neg_half_j*m[43] + half*m[44] + neg_half_j*m[47]
    
    # For C[0,2]
    C[0,2] = neg_half*m[2] + half*m[3] + neg_half*m[5] + neg_half_j*m[8] + half_j*m[11] + \
             half*m[12] + neg_half_j*m[13] + neg_half_j*m[14] + neg_half_j*m[15] + \
             neg_half*m[16] + neg_half*m[18] + half_j*m[19] + neg_half*m[20] + half_j*m[21] + \
             neg_half*m[23] + neg_half_j*m[24] + neg_half*m[25] + half_j*m[26] + half*m[27] + \
             half_j*m[30] + neg_half*m[31] + neg_half_j*m[32] + half*m[33] + half*m[34] + \
             half_j*m[35] + half*m[36] + neg_half_j*m[37] + neg_half*m[38] + neg_half_j*m[39] + \
             half_j*m[43] + neg_half*m[44] + half*m[47]
    
    # For C[0,3]
    C[0,3] = half_j*m[0] + neg_half_j*m[1] + half_j*m[3] + neg_half_j*m[4] + neg_half*m[6] + \
             half*m[7] + half*m[8] + half_j*m[9] + neg_half*m[10] + neg_half*m[11] + half*m[14] + \
             neg_half_j*m[16] + half_j*m[17] + neg_half_j*m[18] + neg_half*m[21] + half*m[22] + \
             half*m[24] + half_j*m[27] + half*m[28] + half_j*m[29] + neg_half_j*m[31] + \
             half_j*m[33] + half_j*m[34] + half*m[37] + half*m[39] + neg_half_j*m[40] + \
             neg_half_j*m[41] + neg_half*m[42] + neg_half*m[43] + neg_half_j*m[45] + \
             neg_half_j*m[46] + half_j*m[47]
    
    # For C[1,0]
    C[1,0] = neg_half*m[0] + neg_half*m[1] + neg_half*m[5] + neg_half_j*m[8] + neg_half_j*m[9] + \
             (half+neg_half_j)*m[11] + neg_half_j*m[14] + half_j*m[15] + (neg_half+half_j)*m[16] + \
             half_j*m[17] + (neg_half+neg_half_j)*m[18] + neg_half*m[24] + half*m[26] + \
             neg_half*m[27] + neg_half_j*m[28] + half*m[30] + neg_half*m[32] + half_j*m[34] + \
             half*m[36] + neg_half*m[37] + neg_half*m[38] + (neg_half+neg_half_j)*m[39] + \
             half_j*m[40] + half*m[42] + half_j*m[43] + neg_half_j*m[44] + neg_half*m[46] + \
             neg_half_j*m[47]
    
    # For C[1,1]
    C[1,1] = half*m[0] + neg_half*m[2] + (half+neg_half_j)*m[3] + half*m[5] + half*m[6] + \
             half_j*m[8] + (neg_half+half_j)*m[11] + half*m[12] + neg_half*m[13] + \
             neg_half*m[14] + neg_half_j*m[15] + neg_half_j*m[17] + (half+half_j)*m[18] + \
             half_j*m[20] + neg_half*m[22] + half*m[24] + half*m[27] + half_j*m[28] + \
             half*m[29] + half*m[32] + (half+neg_half_j)*m[33] + neg_half_j*m[34] + \
             neg_half_j*m[37] + neg_half_j*m[40] + neg_half*m[41] + half*m[43] + \
             half_j*m[44] + half*m[47]
    
    # For C[1,2]
    C[1,2] = half*m[2] + neg_half*m[3] + neg_half*m[5] + neg_half*m[8] + neg_half_j*m[11] + \
             neg_half*m[12] + half*m[13] + half*m[14] + half_j*m[15] + neg_half*m[16] + \
             neg_half*m[18] + half_j*m[19] + neg_half_j*m[20] + neg_half_j*m[21] + half_j*m[23] + \
             neg_half*m[24] + neg_half_j*m[25] + half*m[26] + half_j*m[27] + half*m[30] + \
             neg_half*m[31] + neg_half*m[32] + neg_half*m[33] + half_j*m[34] + neg_half_j*m[35] + \
             half*m[36] + neg_half*m[37] + neg_half*m[38] + neg_half_j*m[39] + neg_half*m[43] + \
             neg_half_j*m[44] + neg_half_j*m[47]
    
    # For C[1,3]
    C[1,3] = neg_half*m[0] + neg_half*m[1] + half_j*m[3] + neg_half*m[4] + neg_half*m[6] + \
             neg_half*m[7] + neg_half_j*m[8] + neg_half_j*m[9] + neg_half*m[10] + half*m[11] + \
             neg_half_j*m[14] + half_j*m[16] + half_j*m[17] + neg_half_j*m[18] + half*m[21] + \
             half*m[22] + neg_half_j*m[24] + neg_half*m[27] + neg_half_j*m[28] + neg_half*m[29] + \
             neg_half_j*m[31] + half_j*m[33] + neg_half*m[34] + half_j*m[37] + neg_half*m[39] + \
             half_j*m[40] + half*m[41] + half*m[42] + half_j*m[43] + half*m[45] + \
             neg_half*m[46] + neg_half*m[47]
    
    # For C[2,0]
    C[2,0] = neg_half_j*m[0] + half_j*m[1] + half_j*m[5] + neg_half_j*m[8] + half*m[9] + \
             (half+half_j)*m[11] + half_j*m[14] + neg_half*m[15] + (neg_half+neg_half_j)*m[16] + \
             half*m[17] + (neg_half+half_j)*m[18] + neg_half*m[24] + half_j*m[26] + half*m[27] + \
             neg_half*m[28] + neg_half_j*m[30] + neg_half_j*m[32] + neg_half_j*m[34] + \
             neg_half_j*m[36] + neg_half*m[37] + neg_half_j*m[38] + (neg_half+half_j)*m[39] + \
             neg_half*m[40] + neg_half_j*m[42] + half_j*m[43] + neg_half*m[44] + \
             neg_half_j*m[46] + half_j*m[47]
    
    # For C[2,1]
    C[2,1] = half_j*m[0] + half_j*m[2] + (neg_half+neg_half_j)*m[3] + neg_half_j*m[5] + \
             half_j*m[6] + half_j*m[8] + (neg_half+neg_half_j)*m[11] + half_j*m[12] + \
             half_j*m[13] + neg_half*m[14] + half*m[15] + neg_half*m[17] + (half+neg_half_j)*m[18] + \
             neg_half*m[20] + half_j*m[22] + half*m[24] + neg_half*m[27] + half*m[28] + \
             neg_half_j*m[29] + half_j*m[32] + (half+half_j)*m[33] + half_j*m[34] + half_j*m[37] + \
             half*m[40] + neg_half_j*m[41] + neg_half*m[43] + half*m[44] + half*m[47]
    
    # For C[2,2]
    C[2,2] = neg_half_j*m[2] + half*m[3] + half_j*m[5] + half*m[8] + half_j*m[11] + neg_half_j*m[12] + \
             neg_half_j*m[13] + half*m[14] + neg_half*m[15] + neg_half*m[16] + neg_half*m[18] + \
             neg_half*m[19] + half*m[20] + neg_half_j*m[21] + half*m[23] + neg_half*m[24] + \
             half*m[25] + half_j*m[26] + half_j*m[27] + neg_half_j*m[30] + half*m[31] + \
             neg_half_j*m[32] + neg_half*m[33] + neg_half_j*m[34] + neg_half*m[35] + \
             neg_half_j*m[36] + neg_half*m[37] + neg_half_j*m[38] + half_j*m[39] + half*m[43] + \
             neg_half*m[44] + half_j*m[47]
    
    # For C[2,3]
    C[2,3] = neg_half_j*m[0] + half_j*m[1] + half_j*m[3] + neg_half_j*m[4] + neg_half_j*m[6] + \
             half_j*m[7] + neg_half_j*m[8] + half*m[9] + neg_half_j*m[10] + half*m[11] + \
             half_j*m[14] + neg_half_j*m[16] + half*m[17] + half_j*m[18] + neg_half*m[21] + \
             neg_half_j*m[22] + half_j*m[24] + half*m[27] + neg_half*m[28] + half_j*m[29] + \
             neg_half_j*m[31] + neg_half_j*m[33] + neg_half*m[34] + neg_half_j*m[37] + \
             neg_half*m[39] + neg_half*m[40] + half_j*m[41] + neg_half_j*m[42] + half_j*m[43] + \
             neg_half_j*m[45] + neg_half_j*m[46] + neg_half*m[47]
    
    # For C[3,0]
    C[3,0] = neg_half_j*m[0] + neg_half_j*m[1] + half*m[5] + half_j*m[8] + half_j*m[9] + \
             (neg_half+half_j)*m[11] + neg_half_j*m[14] + neg_half_j*m[15] + (half+half_j)*m[16] + \
             neg_half_j*m[17] + (half+half_j)*m[18] + half*m[24] + neg_half_j*m[26] + half*m[27] + \
             half*m[28] + half_j*m[30] + half_j*m[32] + neg_half_j*m[34] + neg_half*m[36] + \
             half*m[37] + neg_half*m[38] + (half+neg_half_j)*m[39] + neg_half_j*m[40] + \
             half*m[42] + neg_half_j*m[43] + neg_half*m[44] + half_j*m[46] + neg_half_j*m[47]
    
    # For C[3,1]
    C[3,1] = half_j*m[0] + neg_half*m[2] + (neg_half+neg_half_j)*m[3] + neg_half*m[5] + half*m[6] + \
             neg_half_j*m[8] + (half+neg_half_j)*m[11] + neg_half*m[12] + half_j*m[13] + \
             neg_half*m[14] + half_j*m[15] + half_j*m[17] + (neg_half+neg_half_j)*m[18] + \
             neg_half*m[20] + half*m[22] + neg_half*m[24] + neg_half*m[27] + neg_half*m[28] + \
             neg_half_j*m[29] + neg_half_j*m[32] + (half+half_j)*m[33] + half_j*m[34] + \
             half_j*m[37] + half_j*m[40] + neg_half_j*m[41] + neg_half*m[43] + half*m[44] + \
             half*m[47]
    
    # For C[3,2]
    C[3,2] = half*m[2] + half_j*m[3] + half*m[5] + neg_half*m[8] + neg_half*m[11] + half*m[12] + \
             neg_half_j*m[13] + half*m[14] + neg_half_j*m[15] + half_j*m[16] + half_j*m[18] + \
             half_j*m[19] + half*m[20] + half*m[21] + neg_half*m[23] + half*m[24] + half*m[25] + \
             neg_half_j*m[26] + half_j*m[27] + half_j*m[30] + neg_half_j*m[31] + half_j*m[32] + \
             neg_half_j*m[33] + neg_half_j*m[34] + neg_half_j*m[35] + neg_half*m[36] + half*m[37] + \
             neg_half*m[38] + half*m[39] + half*m[43] + neg_half*m[44] + neg_half_j*m[47]
    
    # For C[3,3]
    C[3,3] = neg_half_j*m[0] + neg_half_j*m[1] + half*m[3] + half_j*m[4] + neg_half*m[6] + \
             neg_half*m[7] + half_j*m[8] + half_j*m[9] + neg_half*m[10] + half_j*m[11] + \
             neg_half_j*m[14] + half*m[16] + neg_half_j*m[17] + half*m[18] + neg_half_j*m[21] + \
             neg_half*m[22] + neg_half_j*m[24] + half*m[27] + half*m[28] + half_j*m[29] + \
             neg_half*m[31] + neg_half*m[33] + neg_half*m[34] + neg_half_j*m[37] + \
             neg_half_j*m[39] + neg_half_j*m[40] + half_j*m[41] + half*m[42] + neg_half_j*m[43] + \
             neg_half_j*m[45] + half_j*m[46] + neg_half*m[47]
    
    # If input was real numeric (not object), ensure output is real
    if is_real_input:
        for i in range(4):
            for j in range(4):
                C[i,j] = C[i,j].real
    
    return C

# ----------------------------
# Hybrid 16x16 = (rank-48 block) x (46-mult base)
# ----------------------------
def hybrid_16x16(A16, B16):
    """
    Multiply 16x16 matrices using:
      - top level: rank-48 4x4 block multiplication
      - base level: 46-mult Rosowski 4x4 inside each block product
    """
    A16 = np.array(A16, dtype=object)
    B16 = np.array(B16, dtype=object)
    assert A16.shape == (16, 16) and B16.shape == (16, 16)

    Ab = np.empty((4, 4), dtype=object)
    Bb = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            Ab[i, j] = Block(A16[i*4:(i+1)*4, j*4:(j+1)*4])
            Bb[i, j] = Block(B16[i*4:(i+1)*4, j*4:(j+1)*4])

    Cb = alphaevolve_rank48_4x4(Ab, Bb)  # 4x4 of Blocks

    C16 = np.empty((16, 16), dtype=object)
    for i in range(4):
        for j in range(4):
            C16[i*4:(i+1)*4, j*4:(j+1)*4] = Cb[i, j].to_array()
    return C16


# ----------------------------
# Verification
# ----------------------------
def verify_all(trials=20, seed=0):
    global MULT_COUNT
    rng = np.random.default_rng(seed)

    # 1) Rosowski 4x4: correctness + 46 mults
    for _ in range(trials):
        A0 = rng.integers(-5, 6, (4, 4))
        B0 = rng.integers(-5, 6, (4, 4))
        A = np.array([[MulCounter(int(x)) for x in row] for row in A0], dtype=object)
        B = np.array([[MulCounter(int(x)) for x in row] for row in B0], dtype=object)

        MULT_COUNT = 0
        C = rosowski_4x4_commutative(A, B)
        assert MULT_COUNT == 46, f"Rosowski mult count {MULT_COUNT} != 46"

        Cval = np.array([[int(c.val) for c in row] for row in C], dtype=int)
        assert np.array_equal(Cval, A0 @ B0), "Rosowski incorrect!"

    # 2) Rank-48 4x4: correctness + 48 mults
    for _ in range(trials):
        A0 = rng.integers(-5, 6, (4, 4))
        B0 = rng.integers(-5, 6, (4, 4))
        A = np.array([[MulCounter(int(x)) for x in row] for row in A0], dtype=object)
        B = np.array([[MulCounter(int(x)) for x in row] for row in B0], dtype=object)

        MULT_COUNT = 0
        C = alphaevolve_rank48_4x4(A, B)
        assert MULT_COUNT == 48, f"Rank-48 mult count {MULT_COUNT} != 48"

        # Extract values - may be complex with negligible imaginary part for real inputs
        def extract_val(c):
            v = c.val
            if isinstance(v, complex):
                return int(round(v.real))
            return int(v)
        
        Cval = np.array([[extract_val(c) for c in row] for row in C], dtype=int)
        assert np.array_equal(Cval, A0 @ B0), "Rank-48 incorrect!"

    # 3) Hybrid 16x16: correctness + 48*46 = 2208 mults
    for _ in range(trials):
        A0 = rng.integers(-3, 4, (16, 16))
        B0 = rng.integers(-3, 4, (16, 16))
        A = np.array([[MulCounter(int(x)) for x in row] for row in A0], dtype=object)
        B = np.array([[MulCounter(int(x)) for x in row] for row in B0], dtype=object)

        MULT_COUNT = 0
        C = hybrid_16x16(A, B)
        assert MULT_COUNT == 2208, f"Hybrid mult count {MULT_COUNT} != 2208"

        # Extract values - may be complex with negligible imaginary part for real inputs
        def extract_val(c):
            v = c.val
            if isinstance(v, complex):
                return int(round(v.real))
            return int(v)
        
        Cval = np.array([[extract_val(c) for c in row] for row in C], dtype=int)
        assert np.array_equal(Cval, A0 @ B0), "Hybrid incorrect!"

    print("✅ All tests passed.")
    print("Rosowski(4x4) multiplications:", 46)
    print("Rank-48(4x4) multiplications:", 48)
    print("Hybrid(16x16) multiplications:", 2208)


if __name__ == "__main__":
    verify_all(trials=20, seed=0)

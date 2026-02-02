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
    Rational-coefficient rank-48 4×4 algorithm (SLP from helper).
    Uses exactly 48 bilinear multiplications.
    """
    A = np.asarray(A, dtype=object)
    B = np.asarray(B, dtype=object)
    assert A.shape == (4, 4) and B.shape == (4, 4)

    # Input aliases (1-indexed names from the SLP)
    A11, A12, A13, A14 = A[0, 0], A[0, 1], A[0, 2], A[0, 3]
    A21, A22, A23, A24 = A[1, 0], A[1, 1], A[1, 2], A[1, 3]
    A31, A32, A33, A34 = A[2, 0], A[2, 1], A[2, 2], A[2, 3]
    A41, A42, A43, A44 = A[3, 0], A[3, 1], A[3, 2], A[3, 3]

    B11, B12, B13, B14 = B[0, 0], B[0, 1], B[0, 2], B[0, 3]
    B21, B22, B23, B24 = B[1, 0], B[1, 1], B[1, 2], B[1, 3]
    B31, B32, B33, B34 = B[2, 0], B[2, 1], B[2, 2], B[2, 3]
    B41, B42, B43, B44 = B[3, 0], B[3, 1], B[3, 2], B[3, 3]

    # L: linear forms in A
    x16 = A13 + A24
    x17 = A14 + A23
    x18 = A12 - A21
    x19 = A31 - A42
    x20 = A33 + A44
    x21 = A34 + A43
    x22 = A22 - A11
    x23 = A32 - A41
    x24 = A13 - A23
    x25 = A32 - A42
    x26 = A33 + A43
    x27 = A31 - A41
    x28 = A34 + A44
    x29 = A12 + A22
    x30 = A11 + A21
    x31 = A14 - A24
    x32 = x23 - x19
    x33 = x16 + x17
    x34 = x20 - x21
    x35 = x22 - x18
    x36 = x20 + x21
    x37 = x18 + x22
    x38 = x16 - x17
    x39 = x19 + x23
    x40 = x29 + x30
    x41 = x25 - x27
    x42 = x26 - x28
    x43 = x24 + x31
    l8 = x32 - x43
    x45 = A33 - A43
    x46 = A31 + A41
    x47 = A13 + A23
    l34 = x34 + x40
    l27 = x33 - x41
    x50 = A32 + A42
    x51 = A12 - A22
    x52 = A14 + A24
    l24 = x42 - x35
    x54 = A34 - A44
    x55 = A11 - A21
    x56 = x17 + x18
    x57 = x34 - x35
    x59 = x37 + x32
    x60 = x38 + x46 + x50
    l38 = x29 - x25
    x63 = x36 + x33
    l2 = x26 - x24
    x66 = x36 - x33
    l6 = x28 - x31
    l36 = x25 + x29
    l12 = x24 + x26
    x71 = x34 + x35
    x72 = x39 - x38
    x73 = x16 - x22
    l22 = x28 + x31
    x75 = x52 - x39 - x47
    x76 = x38 + x39
    x77 = x37 - x32
    x78 = x55 + x36 - x51
    l9 = x27 + x30
    x80 = x45 + x54 - x37
    l42 = x30 - x27
    x82 = x19 + x20
    x83 = x21 + x23
    l0 = l27 - x80
    l1 = x27 - x55
    l3 = x42 - x33
    l4 = l24 + x60
    l5 = x57 - x76
    l7 = x57 + x76
    l10 = x71 + x72
    l11 = x56 + x83
    l13 = x47 - x26
    l14 = l42 - l2
    l15 = x72 - x71
    l16 = x40 + x32
    l17 = x77 - x66
    l18 = x56 - x83
    l19 = x66 + x77
    l20 = x24 - x45
    l21 = x73 + x82
    l23 = x78 - l8
    l25 = l27 + x80
    l26 = x29 + x50
    l28 = x78 + l8
    l29 = x28 + x52
    l30 = x25 + x51
    l31 = x73 - x82
    l32 = x60 - l24
    l33 = x34 - x43
    l35 = x63 - x59
    l37 = l36 + l22
    l39 = x35 - x41
    l40 = x59 + x63
    l41 = x31 + x54
    l43 = l12 + l9
    l44 = l34 + x75
    l45 = l34 - x75
    l46 = x46 - x30
    l47 = l6 - l38

    # R: linear forms in B
    y16 = B21 - B23
    y17 = B32 + B33
    y18 = B42 + B44
    y19 = B11 - B14
    y20 = B41 - B43
    y21 = B31 - B34
    y22 = B22 + B24
    y23 = B12 + B13
    r15 = y19 - y21
    r17 = y16 + y20
    r40 = y23 - y17
    y27 = y18 + B43
    r39 = y22 + y19
    r3 = y17 + y20
    y30 = B13 - y19
    r5 = y18 + y22
    y32 = B24 - r17
    y33 = y17 + B34
    r33 = y18 - y21
    y35 = B24 - y16
    r16 = y16 - y23
    y37 = r15 - B13
    r32 = B22 + B12
    y39 = r39 - r33
    y40 = y17 + y37
    y41 = y18 + y32
    y42 = r17 - r40
    r28 = B22 - B12
    r0 = B31 + B41
    y45 = B34 - r40
    y46 = r15 + r5
    y47 = r3 - r16
    r11 = y42 - y46
    r44 = B41 - B31
    r29 = B42 + B43 - y17
    r45 = y33 - y27
    r6 = B22 - B42
    r1 = B12 + B14 - y22
    r19 = y20 - y16
    r31 = y39 - y47
    r42 = B11 + B31
    r36 = B23 + y22 + y27
    r37 = y27 + y35
    r9 = B14 - y45
    r18 = y42 + y46
    r2 = B12 + B32
    r24 = y40 - y41 + r32
    r14 = B12 - B31
    r26 = B22 + B23 + y23
    r47 = B22 + B41
    r46 = B11 - B13 + y16
    r38 = B41 - B21
    r10 = y22 - y18
    r27 = y27 + r39 + r0 + y45 - y16
    r43 = y33 - y30
    r4 = y30 + y35
    r23 = y35 - y30
    r8 = y40 + y41 - r28
    r13 = B31 - B33 - y20
    r21 = y39 + y47
    r7 = y19 + y21
    r22 = B44 + y32
    r30 = B24 + y19 - B21
    r12 = B33 + y37
    r41 = B41 - B44 + y21
    r20 = B32 + B34 + y18
    r25 = y27 + y33
    r35 = y17 + y23
    r34 = (r44 + r32) * 2 - r24

    L = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11,
        l12, l13, l14, l15, l16, l17, l18, l19, l20, l21,
        l22, l23, l24, l25, l26, l27, l28, l29, l30, l31,
        l32, l33, l34, l35, l36, l37, l38, l39, l40, l41,
        l42, l43, l44, l45, l46, l47,
    ]
    R = [
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11,
        r12, r13, r14, r15, r16, r17, r18, r19, r20, r21,
        r22, r23, r24, r25, r26, r27, r28, r29, r30, r31,
        r32, r33, r34, r35, r36, r37, r38, r39, r40, r41,
        r42, r43, r44, r45, r46, r47,
    ]

    p = [L[i] * R[i] for i in range(48)]

    # P: output linear combinations of products (solved from L/R)
    P_COEFFS = [
        [0, Fraction(-1, 2), 0, 0, 0, 0, 0, 0, Fraction(1, 8), Fraction(1, 2), 0, Fraction(1, 4), 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, 0, 0, 0, 0, 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 8), Fraction(1, 4), 0, Fraction(-1, 2), 0, 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 4), Fraction(-1, 2), Fraction(-1, 2), Fraction(1, 4), 0, 0, Fraction(1, 4), Fraction(1, 4), 0, 0, Fraction(-1, 4)],
        [0, 0, 0, Fraction(-1, 2), 0, 0, Fraction(1, 2), 0, Fraction(-1, 8), 0, 0, Fraction(-1, 4), Fraction(-1, 2), Fraction(1, 2), Fraction(1, 4), Fraction(-1, 4), 0, 0, 0, 0, 0, 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 8), Fraction(-1, 4), Fraction(1, 2), 0, 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, 0, Fraction(-1, 4), 0, 0, Fraction(1, 4), Fraction(-1, 4), 0, 0, Fraction(-1, 4)],
        [0, 0, 0, Fraction(1, 4), 0, 0, 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 8), Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 4), Fraction(1, 4), Fraction(-1, 4), 0, Fraction(-1, 8), Fraction(1, 4), 0, Fraction(1, 8), 0, Fraction(-1, 4), Fraction(-1, 8), 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 2), Fraction(-1, 8), 0, Fraction(1, 4), Fraction(-1, 8), 0, 0, Fraction(1, 4), Fraction(-1, 2), Fraction(-1, 4), 0, 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, Fraction(-1, 4)],
        [0, Fraction(-1, 2), 0, Fraction(1, 4), 0, 0, Fraction(-1, 2), 0, Fraction(1, 8), Fraction(1, 2), Fraction(-1, 4), Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, Fraction(1, 4), 0, Fraction(1, 8), 0, 0, Fraction(-1, 8), 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 8), Fraction(1, 4), Fraction(-1, 2), 0, Fraction(-1, 8), 0, Fraction(-1, 4), Fraction(1, 8), 0, 0, Fraction(1, 4), 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, Fraction(1, 4), 0, Fraction(1, 4), 0, Fraction(1, 4)],
        [0, Fraction(1, 2), 0, 0, 0, Fraction(1, 4), 0, 0, Fraction(-1, 8), 0, 0, 0, 0, 0, Fraction(1, 4), 0, 0, Fraction(1, 4), Fraction(-1, 4), 0, 0, 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 8), Fraction(-1, 4), 0, Fraction(1, 2), 0, 0, 0, Fraction(-1, 8), 0, Fraction(1, 2), Fraction(-1, 4), 0, Fraction(1, 2), 0, 0, Fraction(1, 2), Fraction(-1, 4), Fraction(1, 4), 0, 0, Fraction(1, 4)],
        [0, 0, Fraction(1, 2), Fraction(-1, 2), 0, Fraction(-1, 4), 0, 0, Fraction(1, 8), 0, 0, 0, 0, Fraction(1, 2), Fraction(1, 4), 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, 0, Fraction(1, 2), 0, Fraction(-1, 8), 0, 0, Fraction(-1, 8), Fraction(1, 4), Fraction(1, 2), 0, 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, 0, 0, 0, 0, Fraction(1, 4), Fraction(-1, 4), 0, 0, Fraction(-1, 4)],
        [0, 0, Fraction(-1, 2), Fraction(1, 4), 0, Fraction(1, 4), 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 8), 0, Fraction(-1, 2), Fraction(-1, 4), 0, Fraction(-1, 4), 0, Fraction(-1, 8), 0, 0, Fraction(-1, 8), 0, Fraction(1, 4), Fraction(1, 8), 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 2), Fraction(1, 8), 0, Fraction(-1, 4), Fraction(-1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(-1, 4), 0, Fraction(1, 4), 0, 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, Fraction(1, 4)],
        [0, Fraction(1, 2), 0, Fraction(1, 4), 0, 0, 0, Fraction(1, 4), Fraction(-1, 8), 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 4), 0, Fraction(1, 4), Fraction(1, 4), Fraction(-1, 8), 0, 0, Fraction(-1, 8), Fraction(-1, 2), 0, Fraction(1, 8), 0, 0, Fraction(1, 8), Fraction(-1, 4), Fraction(-1, 2), 0, Fraction(-1, 8), 0, Fraction(1, 4), Fraction(1, 8), 0, 0, Fraction(1, 4), 0, Fraction(1, 4), 0, 0, Fraction(1, 2), Fraction(-1, 4), 0, Fraction(1, 4), 0, Fraction(1, 4)],
        [Fraction(-1, 4), 0, 0, 0, 0, 0, 0, 0, Fraction(1, 8), Fraction(1, 2), 0, Fraction(1, 4), 0, 0, Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), 0, 0, 0, 0, 0, 0, 0, Fraction(-1, 8), 0, Fraction(1, 2), Fraction(1, 8), 0, 0, 0, 0, Fraction(-1, 4), 0, Fraction(-1, 8), 0, 0, Fraction(1, 4), Fraction(1, 2), 0, Fraction(1, 4), 0, 0, Fraction(1, 4), 0, 0, Fraction(1, 2), Fraction(1, 4)],
        [Fraction(1, 4), 0, 0, 0, 0, 0, Fraction(-1, 2), 0, Fraction(-1, 8), 0, 0, Fraction(-1, 4), Fraction(-1, 2), 0, Fraction(-1, 4), Fraction(-1, 4), 0, 0, 0, 0, Fraction(-1, 2), 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 8), 0, 0, 0, 0, Fraction(1, 4), Fraction(-1, 2), Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, 0, Fraction(-1, 4), Fraction(1, 2), 0, Fraction(1, 4), 0, 0, 0, Fraction(1, 4)],
        [0, 0, 0, Fraction(1, 4), 0, 0, 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 8), Fraction(1, 2), 0, Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), 0, Fraction(1, 8), Fraction(-1, 4), Fraction(1, 2), Fraction(1, 8), 0, 0, Fraction(-1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(-1, 8), 0, 0, 0, Fraction(1, 8), Fraction(-1, 4), Fraction(1, 4), Fraction(-1, 8), 0, 0, Fraction(1, 4), Fraction(1, 2), Fraction(1, 4), 0, 0, 0, Fraction(-1, 4), 0, 0, 0, Fraction(1, 4)],
        [Fraction(-1, 4), 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, Fraction(1, 2), 0, Fraction(1, 8), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), 0, 0, Fraction(1, 4), 0, Fraction(1, 4), 0, Fraction(-1, 8), 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 8), 0, 0, 0, Fraction(1, 8), 0, Fraction(1, 4), Fraction(-1, 8), 0, 0, Fraction(1, 4), 0, Fraction(-1, 4), Fraction(1, 4), Fraction(-1, 2), 0, Fraction(1, 4), 0, 0, Fraction(1, 2), Fraction(-1, 4)],
        [Fraction(1, 4), 0, 0, 0, 0, Fraction(-1, 4), 0, 0, Fraction(1, 8), 0, 0, 0, 0, 0, Fraction(1, 4), 0, Fraction(1, 2), Fraction(-1, 4), Fraction(1, 4), 0, 0, 0, 0, 0, Fraction(-1, 8), 0, Fraction(1, 2), Fraction(-1, 8), 0, 0, 0, 0, Fraction(-1, 4), 0, Fraction(1, 8), 0, Fraction(-1, 2), Fraction(1, 4), 0, 0, 0, 0, Fraction(1, 2), Fraction(1, 4), 0, 0, Fraction(1, 2), Fraction(1, 4)],
        [Fraction(-1, 4), 0, Fraction(1, 2), 0, 0, Fraction(1, 4), 0, 0, Fraction(-1, 8), 0, 0, 0, 0, 0, Fraction(1, 4), 0, 0, Fraction(1, 4), Fraction(-1, 4), 0, Fraction(1, 2), 0, Fraction(-1, 2), 0, Fraction(1, 8), 0, 0, Fraction(1, 8), 0, 0, 0, 0, Fraction(1, 4), Fraction(1, 2), Fraction(-1, 8), 0, 0, Fraction(1, 4), 0, 0, 0, Fraction(-1, 2), 0, Fraction(-1, 4), 0, 0, 0, Fraction(-1, 4)],
        [0, 0, Fraction(-1, 2), Fraction(1, 4), 0, Fraction(-1, 4), 0, 0, Fraction(1, 8), 0, 0, Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, Fraction(1, 4), 0, Fraction(1, 8), 0, Fraction(-1, 2), Fraction(-1, 8), 0, 0, Fraction(-1, 8), Fraction(-1, 4), Fraction(1, 2), Fraction(1, 8), 0, 0, 0, Fraction(-1, 8), Fraction(-1, 4), Fraction(-1, 4), Fraction(1, 8), Fraction(1, 4), Fraction(-1, 2), Fraction(1, 4), 0, Fraction(-1, 4), 0, 0, 0, Fraction(1, 4), 0, 0, 0, Fraction(1, 4)],
        [Fraction(1, 4), 0, 0, Fraction(-1, 4), Fraction(1, 4), 0, 0, Fraction(1, 4), Fraction(1, 8), 0, 0, Fraction(-1, 8), 0, 0, Fraction(1, 4), 0, Fraction(1, 4), Fraction(-1, 4), Fraction(1, 8), 0, 0, Fraction(-1, 8), Fraction(1, 2), 0, Fraction(1, 8), 0, 0, Fraction(-1, 8), 0, 0, 0, Fraction(1, 8), 0, Fraction(-1, 4), Fraction(1, 8), 0, 0, Fraction(-1, 4), 0, Fraction(1, 4), 0, Fraction(1, 2), Fraction(1, 2), Fraction(1, 4), 0, 0, Fraction(1, 2), Fraction(1, 4)],
    ]

    C_flat = []
    for coeffs in P_COEFFS:
        acc = 0
        for coeff, pi in zip(coeffs, p):
            if coeff:
                acc += coeff * pi
        C_flat.append(acc)

    return np.array(C_flat, dtype=object).reshape(4, 4)

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

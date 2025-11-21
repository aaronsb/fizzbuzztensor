# TensorFizzBuzz: A Signal Processing Approach to a Classic Problem

**Abstract**: We present a novel tensor-based approach to FizzBuzz that reveals its underlying mathematical structure as a periodic signal with well-defined spectral properties. By representing FizzBuzz as a pattern vector of period 15, we reduce the problem from algorithmic conditionals to simple modular indexing. We demonstrate that FizzBuzz can be understood as a discrete signal with a fundamental frequency of 1/15 cycles per sample, and explore connections to trigonometric and Fourier-based solutions.

---

## 1. Introduction

FizzBuzz is traditionally presented as a control-flow problem:

```
For n = 1 to N:
  if n divisible by 15: print "FizzBuzz"
  else if n divisible by 3: print "Fizz"
  else if n divisible by 5: print "Buzz"
  else: print n
```

However, this algorithmic perspective obscures the problem's fundamental mathematical structure. Susam Pal's elegant work "Fizz Buzz With Cosines" [1] demonstrated that FizzBuzz can be solved using trigonometric functions, exploiting the periodic nature of divisibility. This raised a natural question: **if FizzBuzz is fundamentally a periodic function, why not represent it as a first-class tensor?**

We propose viewing FizzBuzz not as a sequence of conditionals, but as a **periodic signal** that can be represented as a rank-1 tensor - the pattern vector. This representation makes the periodicity explicit and enables efficient vectorized computation.

## 2. The Pattern Vector

### 2.1 Periodicity and the LCM

The key insight is that FizzBuzz has period $P = \text{lcm}(3, 5) = 15$. After every 15 positions, the pattern repeats exactly. This allows us to represent the entire infinite sequence with a single vector:

$$
\mathbf{p} = [0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 3]
$$

where the encoding is:
- 0 → print number
- 1 → print "Fizz"
- 2 → print "Buzz"
- 3 → print "FizzBuzz"

### 2.2 Computation via Modular Indexing

For any position $n \geq 1$, the FizzBuzz category is:

$$
c(n) = \mathbf{p}[(n-1) \bmod 15]
$$

This is an $O(1)$ operation with constant space complexity, requiring only 15 elements regardless of sequence length.

### 2.3 Tensor Construction

The pattern vector can be computed directly from divisibility checks:

$$
\mathbf{p}_i = \begin{cases}
1 & \text{if } i \equiv 0 \pmod{3} \text{ and } i \not\equiv 0 \pmod{5} \\
2 & \text{if } i \not\equiv 0 \pmod{3} \text{ and } i \equiv 0 \pmod{5} \\
3 & \text{if } i \equiv 0 \pmod{3} \text{ and } i \equiv 0 \pmod{5} \\
0 & \text{otherwise}
\end{cases}
$$

Or more compactly using indicator functions:

$$
\mathbf{p}_i = \mathbb{1}_{3|i} \cdot 1 + \mathbb{1}_{5|i} \cdot 2
$$

where $\mathbb{1}_{d|i}$ is 1 if $d$ divides $i$, else 0.

## 3. Vectorized Implementation

### 3.1 NumPy Pattern Construction

```python
import numpy as np

# Create pattern for one period
nums = np.arange(1, 16)[:, None]        # (15, 1)
divisors = np.array([3, 5])[None, :]    # (1, 2)

# Divisibility matrix via broadcasting
div_matrix = (nums % divisors == 0).astype(int)  # (15, 2)

# Encode as single vector: [div_by_3, div_by_5] → category
PATTERN = div_matrix @ np.array([1, 2])  # (15,)
```

### 3.2 Lookup Function

```python
def fizzbuzz(n):
    """FizzBuzz for numbers 1 to n using pattern vector."""
    nums = np.arange(1, n + 1)
    categories = PATTERN[(nums - 1) % 15]

    # Decode categories to strings
    decoder = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)
    result = decoder[categories].copy()

    # Fill in numbers where category is 0
    result[categories == 0] = nums[categories == 0].astype(str)
    return result
```

## 4. Signal Analysis

### 4.1 FizzBuzz as a Discrete Signal

Viewing the category values as a discrete signal $s[n] = c(n)$, we can analyze its properties:

![FizzBuzz Waveform](images/fizzbuzz_waveform.png)

**Figure 1**: Top - The FizzBuzz pattern signal over 5 periods. Middle - Component signals showing divisibility by 3 (blue) and 5 (orange). Bottom - Binary divisibility matrix.

The waveform clearly shows:
- Periodic structure with period $P = 15$
- Two component frequencies (1/3 and 1/5) that interfere to create the pattern
- The binary matrix reveals the underlying divisibility checks

### 4.2 Frequency Domain Analysis

Applying the Discrete Fourier Transform:

$$
S[k] = \sum_{n=0}^{N-1} s[n] e^{-i 2\pi k n / N}
$$

![Frequency Spectrum](images/fizzbuzz_fft.png)

**Figure 2**: Time domain signal (top) and frequency spectrum (bottom). The fundamental frequency at $f_0 = 1/15 \approx 0.0667$ is clearly visible.

The spectrum reveals:
- Strong DC component (average value)
- Fundamental frequency at $1/P = 1/15$
- Harmonics at integer multiples of the fundamental
- Clean spectral lines due to perfect periodicity

## 5. Trigonometric Representation

### 5.1 Cosine-Based Solution

Given the periodic nature, FizzBuzz can be represented using trigonometric functions. The divisibility checks can be encoded as:

$$
\text{div\_by\_3}(n) = \frac{1 + \cos\left(\frac{2\pi n}{3}\right)}{2} \cdot \frac{1 + \cos\left(\frac{4\pi n}{3}\right)}{2}
$$

$$
\text{div\_by\_5}(n) = \frac{1 + \cos\left(\frac{2\pi n}{5}\right)}{2} \cdot \frac{1 + \cos\left(\frac{4\pi n}{5}\right)}{2} \cdot \frac{1 + \cos\left(\frac{6\pi n}{5}\right)}{2} \cdot \frac{1 + \cos\left(\frac{8\pi n}{5}\right)}{2}
$$

These expressions equal 1 when $n$ is divisible by the respective number, and 0 otherwise.

### 5.2 Fourier Series Representation

The pattern can be expressed as a Fourier series:

$$
c(t) = a_0 + \sum_{k=1}^{\infty} \left[ a_k \cos\left(\frac{2\pi k t}{15}\right) + b_k \sin\left(\frac{2\pi k t}{15}\right) \right]
$$

For our discrete signal, only a finite number of harmonics are needed (at most 15/2 = 7 for a period-15 signal).

## 6. Higher-Dimensional Representations

### 6.1 2D Visualization

Arranging FizzBuzz in a 2D grid reveals spatial patterns:

![2D Heatmap](images/fizzbuzz_2d.png)

**Figure 3**: FizzBuzz as a 20×20 texture. Diagonal patterns emerge where multiples of 15 align.

### 6.2 Rank-2 Tensor: The Divisibility Matrix

The most fundamental representation is the $(N, D)$ divisibility matrix, where $N$ is sequence length and $D$ is the number of divisors:

$$
\mathbf{M} = \begin{bmatrix}
\mathbb{1}_{3|1} & \mathbb{1}_{5|1} \\
\mathbb{1}_{3|2} & \mathbb{1}_{5|2} \\
\vdots & \vdots \\
\mathbb{1}_{3|N} & \mathbb{1}_{5|N}
\end{bmatrix} = \begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
\vdots & \vdots
\end{bmatrix}
$$

This can be computed via broadcasting:

```python
nums = np.arange(1, N+1)[:, None]      # (N, 1)
divisors = np.array([3, 5])[None, :]   # (1, 2)
M = (nums % divisors == 0)              # (N, 2) via broadcasting
```

### 6.3 Rank-1 Compression

The rank-2 matrix compresses to rank-1 via encoding:

$$
\mathbf{c} = \mathbf{M} \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \text{cat}_1 + 2 \cdot \text{cat}_2
$$

This maps the four states $\\{(0,0), (1,0), (0,1), (1,1)\\}$ to $\\{0, 1, 2, 3\\}$.

## 7. Generalization

### 7.1 Arbitrary Divisor Sets

For divisors $d_1, d_2, \ldots, d_k$, the period is:

$$
P = \text{lcm}(d_1, d_2, \ldots, d_k)
$$

The pattern vector has length $P$ and can encode $2^k$ possible states (each subset of divisors).

### 7.2 Example: FizzBuzzBazz

For divisors $\\{3, 5, 7\\}$:
- Period: $\text{lcm}(3, 5, 7) = 105$
- States: $2^3 = 8$ categories
- Pattern vector: 105 elements

```python
def create_pattern(divisors):
    """Create pattern vector for arbitrary divisors."""
    div_values = [d for d, _ in divisors]
    period = np.lcm.reduce(div_values)

    nums = np.arange(1, period + 1)[:, None]
    div_array = np.array(div_values)[None, :]
    div_matrix = (nums % div_array == 0).astype(int)

    # Binary encoding: each divisor gets a bit
    powers = 2 ** np.arange(len(divisors))
    pattern = div_matrix @ powers

    return pattern
```

## 8. Computational Complexity

### 8.1 Space Complexity

- **Traditional approach**: $O(1)$ (no storage, compute on-the-fly)
- **Pattern vector**: $O(\text{lcm}(d_1, \ldots, d_k))$ (constant for fixed divisors)
- **For FizzBuzz**: $O(15) = O(1)$

### 8.2 Time Complexity

Per element:
- **Traditional**: $O(k)$ modulo operations and comparisons
- **Pattern vector**: $O(1)$ modulo and lookup

For sequence of length $N$:
- **Traditional**: $O(Nk)$
- **Pattern vector**: $O(N)$ (vectorized, highly parallelizable)

## 9. Philosophical Implications

The pattern vector representation reveals that FizzBuzz is not fundamentally a programming problem, but a **mathematical object** - a periodic function with:

1. **Well-defined period** determined by the LCM of divisors
2. **Spectral decomposition** into component frequencies
3. **Minimal representation** as a finite lookup table
4. **Elegant generalization** to arbitrary divisor sets

This transforms FizzBuzz from an interview screening question into a case study in:
- Periodic functions and Fourier analysis
- Tensor operations and broadcasting
- The duality between algorithmic and declarative representations
- Signal processing perspectives on discrete mathematics

## 10. Conclusion

We have demonstrated that FizzBuzz, when viewed through the lens of tensor operations and signal processing, reveals a rich mathematical structure. The pattern vector representation:

- Reduces infinite sequences to finite lookup tables
- Enables $O(1)$ per-element computation
- Exposes the periodic and spectral nature of the problem
- Generalizes naturally to arbitrary divisor sets
- Connects to trigonometric and Fourier-based solutions

This approach transforms a simple programming exercise into an exploration of periodicity, signal analysis, and the expressive power of tensor operations.

---

## References

[1] Pal, Susam. "Fizz Buzz With Cosines." *susam.net*, https://susam.net/fizz-buzz-with-cosines.html. Accessed 21 Nov. 2025.

### Code

- Implementation: `fizzbuzz.py`
- Visualizations: `visualize.py`
- Pattern vector: 15-element rank-1 tensor encoding complete solution
- Fundamental frequency: $f_0 = 1/15 \approx 0.0667$ cycles/sample

## Acknowledgments

This work was directly inspired by Susam Pal's "Fizz Buzz With Cosines" [1], which elegantly demonstrated that FizzBuzz can be solved using trigonometric functions. Upon seeing that periodic functions could solve FizzBuzz, the natural question arose: "Why not tensors?" This paper explores the answer to that question, showing that representing FizzBuzz as a first-class tensor - the pattern vector - reveals the problem's fundamental mathematical structure as a discrete periodic signal.

---

**Repository**: `/home/aaron/Projects/ai/tensorfizzbuzz/`

*"The most elemental solution to FizzBuzz is not an algorithm, but a number: 15."*

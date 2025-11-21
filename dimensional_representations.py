"""
Different dimensional representations of FizzBuzz

Demonstrates how different tensor shapes solve different problems:
- 2×2 binary matrix: maximum compression (4 elements)
- 3×5 modular matrix: explicit structure (15 elements)
- 15-element vector: fast sequential lookup
- Batched 3D: parallel computation across multiple sequences
"""

import numpy as np


# ============================================================================
# Representation 1: Binary Divisibility Matrix (2×2 - most compressed)
# ============================================================================

PATTERN_BINARY = np.array([
    [0, 2],  # not div by 3: [number, Buzz]
    [1, 3]   # div by 3:     [Fizz, FizzBuzz]
])

def fizzbuzz_binary(n):
    """
    Most compressed representation: 4 elements.

    Uses binary divisibility checks to index into 2×2 matrix.
    More computation per lookup, but minimal storage.
    """
    nums = np.arange(1, n + 1)
    div_by_3 = (nums % 3 == 0).astype(int)
    div_by_5 = (nums % 5 == 0).astype(int)

    # Index into 2×2 matrix
    categories = PATTERN_BINARY[div_by_3, div_by_5]

    # Decode
    decoder = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)
    result = decoder[categories].copy()
    result[categories == 0] = nums[categories == 0].astype(str)

    return result


# ============================================================================
# Representation 2: Modular Matrix (3×5 - explicit structure)
# ============================================================================

PATTERN_MODULAR = np.array([
    [3, 1, 1, 1, 1],  # n%3==0 (divisible by 3)
    [2, 0, 0, 0, 0],  # n%3==1
    [2, 0, 0, 0, 0]   # n%3==2
])

def fizzbuzz_modular(n):
    """
    Explicit 2D structure: 15 elements as 3×5 matrix.

    Each dimension corresponds to one divisor's modular cycle.
    Shows the Cartesian product structure clearly.
    """
    nums = np.arange(1, n + 1)
    categories = PATTERN_MODULAR[nums % 3, nums % 5]

    decoder = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)
    result = decoder[categories].copy()
    result[categories == 0] = nums[categories == 0].astype(str)

    return result


# ============================================================================
# Representation 3: Pattern Vector (15-element - from main implementation)
# ============================================================================

PATTERN_VECTOR = np.array([0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 3])

def fizzbuzz_vector(n):
    """
    Sequential pattern: 15 elements.

    Single modulo operation, direct array indexing.
    Fastest for sequential access.
    """
    nums = np.arange(1, n + 1)
    categories = PATTERN_VECTOR[(nums - 1) % 15]

    decoder = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)
    result = decoder[categories].copy()
    result[categories == 0] = nums[categories == 0].astype(str)

    return result


# ============================================================================
# Representation 4: Batched 3D Tensor (batch × sequence × divisors)
# ============================================================================

def fizzbuzz_batched(batch_size, sequence_length):
    """
    3D tensor for parallel computation: (batch, sequence, divisors)

    Computes multiple FizzBuzz sequences simultaneously.
    Useful for processing different ranges or parameters in parallel.
    """
    # Create batched sequences
    # Batch 0: [1, 2, ..., sequence_length]
    # Batch 1: [sequence_length+1, ..., 2*sequence_length]
    # etc.

    offsets = np.arange(batch_size)[:, None]  # (batch, 1)
    positions = np.arange(sequence_length)[None, :]  # (1, sequence)
    nums = offsets * sequence_length + positions + 1  # (batch, sequence)

    # Create divisibility tensor
    divisors = np.array([3, 5])[None, None, :]  # (1, 1, 2)
    div_matrix = (nums[:, :, None] % divisors == 0).astype(int)  # (batch, sequence, 2)

    # Encode categories
    powers = np.array([1, 2])[None, None, :]  # (1, 1, 2)
    categories = (div_matrix * powers).sum(axis=2)  # (batch, sequence)

    # Decode to strings
    decoder = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)
    result = np.empty((batch_size, sequence_length), dtype=object)

    for i in range(batch_size):
        for j in range(sequence_length):
            cat = categories[i, j]
            if cat == 0:
                result[i, j] = str(nums[i, j])
            else:
                result[i, j] = decoder[cat]

    return result, div_matrix


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_representations():
    """Compare the different representations."""

    print("FizzBuzz Dimensional Representations")
    print("=" * 70)
    print()

    # Test that all produce same results
    n = 30
    r1 = fizzbuzz_binary(n)
    r2 = fizzbuzz_modular(n)
    r3 = fizzbuzz_vector(n)

    assert np.array_equal(r1, r2), "Binary ≠ Modular"
    assert np.array_equal(r2, r3), "Modular ≠ Vector"

    print("✓ All representations produce identical results")
    print()

    # Compare properties
    print("Representation Properties:")
    print("-" * 70)
    print(f"{'Representation':<20} {'Shape':<15} {'Elements':<10} {'Best For'}")
    print("-" * 70)
    print(f"{'Binary Matrix':<20} {'(2, 2)':<15} {4:<10} {'Minimal storage'}")
    print(f"{'Modular Matrix':<20} {'(3, 5)':<15} {15:<10} {'Explicit structure'}")
    print(f"{'Pattern Vector':<20} {'(15,)':<15} {15:<10} {'Fast sequential'}")
    print(f"{'Batched Tensor':<20} {'(B, N, 2)':<15} {'B×N×2':<10} {'Parallel batches'}")
    print()

    # Show the compact representations
    print("Binary Matrix (2×2 - most compact):")
    print(PATTERN_BINARY)
    print("  Index: [div_by_3][div_by_5]")
    print()

    print("Modular Matrix (3×5 - explicit structure):")
    print(PATTERN_MODULAR)
    print("  Index: [n % 3][n % 5]")
    print()

    print("Pattern Vector (15 - sequential):")
    print(PATTERN_VECTOR)
    print("  Index: (n-1) % 15")
    print()

    # Demonstrate batched computation
    print("Batched 3D Tensor Example:")
    batch_size, seq_len = 3, 10
    batched_result, div_tensor = fizzbuzz_batched(batch_size, seq_len)

    print(f"  Shape: {div_tensor.shape} (batch × sequence × divisors)")
    print(f"  Batch 0 (1-10):  {' '.join(batched_result[0])}")
    print(f"  Batch 1 (11-20): {' '.join(batched_result[1])}")
    print(f"  Batch 2 (21-30): {' '.join(batched_result[2])}")
    print()

    # Space complexity analysis
    print("Space Complexity Analysis:")
    print("-" * 70)
    print("Binary (2×2):      4 elements + 2 modulo ops per lookup")
    print("Modular (3×5):    15 elements + 2 modulo ops per lookup")
    print("Vector (15):      15 elements + 1 modulo op per lookup")
    print("Batched (B×N×2): 2BN elements, all computed in parallel")
    print()

    print("Trade-offs:")
    print("-" * 70)
    print("• Binary matrix: smallest storage, but more computation")
    print("• Modular matrix: shows Cartesian product structure explicitly")
    print("• Pattern vector: simplest indexing, good for sequential access")
    print("• Batched tensor: enables parallel computation of multiple sequences")


def visualize_dimensions():
    """Show how dimensions relate to problem structure."""

    print("\nDimensional Structure of FizzBuzz:")
    print("=" * 70)
    print()

    print("Rank-0 (scalar): The concept/rule itself")
    print("  'divisibility by 3 and 5'")
    print()

    print("Rank-1 (vector): Complete period")
    print("  15 elements = LCM(3,5)")
    print("  [0,0,1,0,2,1,0,0,1,2,0,1,0,0,3]")
    print()

    print("Rank-2 (matrix): Factored structure")
    print("  3×5 (modular): position in each cycle")
    print("  2×2 (binary):  divisibility checks")
    print()

    print("Rank-3 (tensor): Parallel computation")
    print("  (batch, sequence, divisors)")
    print("  Process multiple sequences simultaneously")
    print()

    print("Higher ranks: Generalizations")
    print("  For divisors {3,5,7}: 3×5×7 cube or 2×2×2 binary")
    print("  For divisors {3,5,7,11}: 3×5×7×11 hypercube or 2×2×2×2 binary")


if __name__ == "__main__":
    compare_representations()
    visualize_dimensions()

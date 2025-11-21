"""
FizzBuzz: Compact Binary Matrix Approach

Maximum compression: 2×2 matrix (4 elements) indexed by binary divisibility.
"""

import numpy as np


# The compact 2×2 lookup table
# Index: [divisible by 3?][divisible by 5?]
PATTERN_COMPACT = np.array([
    [0, 2],  # not div by 3: [number, Buzz]
    [1, 3]   # div by 3:     [Fizz, FizzBuzz]
])

# Decoder: category → output
DECODER = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)


def fizzbuzz(n):
    """
    Compute FizzBuzz for numbers 1 to n using 2×2 binary matrix.

    This is the most compact representation: only 4 elements.

    Args:
        n: Upper limit (inclusive)

    Returns:
        numpy array of strings
    """
    nums = np.arange(1, n + 1)

    # Binary divisibility checks
    div_by_3 = (nums % 3 == 0).astype(int)
    div_by_5 = (nums % 5 == 0).astype(int)

    # Index into 2×2 matrix
    categories = PATTERN_COMPACT[div_by_3, div_by_5]

    # Decode categories to strings
    result = DECODER[categories].copy()

    # Fill in numbers where category is 0
    number_mask = (categories == 0)
    result[number_mask] = nums[number_mask].astype(str)

    return result


def print_fizzbuzz(n):
    """Print FizzBuzz from 1 to n."""
    result = fizzbuzz(n)
    for line in result:
        print(line)


def show_matrix():
    """Visualize the compact 2×2 matrix."""
    print("Compact Binary Matrix (2×2):")
    print("=" * 50)
    print(f"Matrix:\n{PATTERN_COMPACT}")
    print()
    print("Index: [divisible by 3?][divisible by 5?]")
    print()
    print("Lookup table:")
    print("  [False, False] → 0 (number)")
    print("  [False, True]  → 2 (Buzz)")
    print("  [True,  False] → 1 (Fizz)")
    print("  [True,  True]  → 3 (FizzBuzz)")
    print()
    print("Space: 4 elements (vs 15 for pattern vector)")
    print("Time:  2 modulo ops + 1 lookup per position")


if __name__ == "__main__":
    print("FizzBuzz: Compact Binary Matrix Approach")
    print("=" * 50)
    print()

    show_matrix()

    print("\nFizzBuzz 1-30:")
    print("-" * 50)
    print_fizzbuzz(30)

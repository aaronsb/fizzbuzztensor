"""
TensorFizzBuzz: The Pattern Vector Approach

FizzBuzz reduced to its mathematical essence - a repeating pattern of period 15.
"""

import numpy as np


# The fundamental pattern: the complete solution to FizzBuzz
# Period = LCM(3, 5) = 15
PATTERN = np.array([0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 3])

# Decoder: category → output
DECODER = np.array(["{}", "Fizz", "Buzz", "FizzBuzz"], dtype=object)


def fizzbuzz(n):
    """
    Compute FizzBuzz for numbers 1 to n using the pattern vector.

    The pattern vector encodes the complete solution:
    - 0: print the number
    - 1: print "Fizz"
    - 2: print "Buzz"
    - 3: print "FizzBuzz"

    Returns:
        numpy array of strings
    """
    nums = np.arange(1, n + 1)

    # Map each number to its position in the pattern
    categories = PATTERN[(nums - 1) % 15]

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


def show_pattern():
    """Visualize the fundamental pattern."""
    print("The Pattern Vector (period = 15):")
    print("=" * 50)
    print(f"Pattern: {PATTERN}")
    print(f"Decoded: {fizzbuzz(15)}")
    print()
    print("Position:  ", " ".join(f"{i:>4}" for i in range(1, 16)))
    print("Category:  ", " ".join(f"{c:>4}" for c in PATTERN))
    print("Output:    ", " ".join(f"{s:>4}" for s in fizzbuzz(15)))
    print()
    print("This pattern repeats forever!")


def create_pattern(divisors):
    """
    Create a pattern vector for arbitrary divisors.

    Args:
        divisors: list of (divisor, label) tuples

    Returns:
        pattern vector and decoder
    """
    # Period is LCM of all divisors
    div_values = [d for d, _ in divisors]
    period = np.lcm.reduce(div_values)

    # Create pattern for one period
    nums = np.arange(1, period + 1)[:, None]
    div_array = np.array(div_values)[None, :]

    # Divisibility matrix: (period, n_divisors)
    div_matrix = (nums % div_array == 0).astype(int)

    # Encode: each combination gets a unique category
    # Use powers of 2 for binary encoding
    powers = 2 ** np.arange(len(divisors))
    pattern = div_matrix @ powers

    # Create decoder
    n_categories = 2 ** len(divisors)
    decoder = np.empty(n_categories, dtype=object)

    for category in range(n_categories):
        # Decode which divisors are active
        active = [(category >> i) & 1 for i in range(len(divisors))]
        labels = [divisors[i][1] for i, a in enumerate(active) if a]
        decoder[category] = "".join(labels) if labels else "{}"

    return pattern, decoder


if __name__ == "__main__":
    print("TensorFizzBuzz: The Pattern Vector Approach")
    print("=" * 50)
    print()

    show_pattern()

    print("\nFizzBuzz 1-30:")
    print("-" * 50)
    print_fizzbuzz(30)

    print("\n\nGeneralizing: FizzBuzzBazz (divisors 3, 5, 7)")
    print("=" * 50)
    pattern_357, decoder_357 = create_pattern([(3, "Fizz"), (5, "Buzz"), (7, "Bazz")])
    print(f"Pattern length (LCM): {len(pattern_357)}")
    print(f"First 35 values:")

    nums = np.arange(1, 36)
    categories = pattern_357[(nums - 1) % len(pattern_357)]
    result = decoder_357[categories].copy()
    number_mask = (result == "{}")
    result[number_mask] = nums[number_mask].astype(str)

    for i, val in enumerate(result, 1):
        print(f"{i:3}: {val}")

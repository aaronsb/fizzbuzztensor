"""
FizzBuzz: Batched 3D Tensor Approach

Parallel computation across multiple sequences using 3D tensors.
Shape: (batch_size, sequence_length, n_divisors)
"""

import numpy as np


def fizzbuzz_batched(batch_size, sequence_length, offset=0):
    """
    Compute multiple FizzBuzz sequences in parallel using 3D tensors.

    Args:
        batch_size: Number of sequences to compute
        sequence_length: Length of each sequence
        offset: Starting offset for batch 0 (default: 0 for starting at 1)

    Returns:
        result: (batch, sequence) array of strings
        div_tensor: (batch, sequence, 2) divisibility tensor
    """
    # Create batched sequences
    # Batch i starts at: offset + i * sequence_length + 1
    batch_offsets = np.arange(batch_size)[:, None]  # (batch, 1)
    positions = np.arange(sequence_length)[None, :]  # (1, sequence)
    nums = offset + batch_offsets * sequence_length + positions + 1  # (batch, sequence)

    # Create 3D divisibility tensor
    divisors = np.array([3, 5])[None, None, :]  # (1, 1, 2)
    div_tensor = (nums[:, :, None] % divisors == 0).astype(int)  # (batch, sequence, 2)

    # Encode categories from divisibility
    powers = np.array([1, 2])[None, None, :]  # (1, 1, 2)
    categories = (div_tensor * powers).sum(axis=2)  # (batch, sequence)

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

    return result, div_tensor, nums


def print_batched(result, start_nums):
    """Pretty print batched results."""
    batch_size = result.shape[0]
    for i in range(batch_size):
        start = start_nums[i, 0]
        end = start_nums[i, -1]
        print(f"  Batch {i} ({start:3d}-{end:3d}): {' '.join(result[i])}")


def show_tensor_structure():
    """Visualize the 3D tensor structure."""
    print("Batched 3D Tensor Structure:")
    print("=" * 70)
    print()
    print("Shape: (batch_size, sequence_length, n_divisors)")
    print()
    print("Dimensions:")
    print("  Axis 0 (batch):    Which sequence we're computing")
    print("  Axis 1 (sequence): Position within the sequence")
    print("  Axis 2 (divisors): Divisibility checks [div_by_3, div_by_5]")
    print()
    print("Example: tensor[2, 5, 1] = divisibility by 5 at position 5 of batch 2")
    print()


if __name__ == "__main__":
    print("FizzBuzz: Batched 3D Tensor Approach")
    print("=" * 70)
    print()

    show_tensor_structure()

    # Demonstrate with small batches
    print("Example: 3 batches of 10 numbers each")
    print("-" * 70)
    result, div_tensor, nums = fizzbuzz_batched(batch_size=3, sequence_length=10)

    print(f"Tensor shape: {div_tensor.shape}")
    print()
    print_batched(result, nums)
    print()

    # Show the divisibility tensor for first batch, first 5 positions
    print("Divisibility tensor for Batch 0, positions 0-4:")
    print("  [div_by_3, div_by_5]")
    for j in range(5):
        print(f"  Position {j} (n={nums[0, j]}): {div_tensor[0, j]}")
    print()

    # Demonstrate parallel computation advantage
    print("Parallel Computation Example:")
    print("-" * 70)
    print("Computing FizzBuzz for ranges [1-100], [101-200], [201-300]")
    result, _, nums = fizzbuzz_batched(batch_size=3, sequence_length=100)

    print(f"Computed {result.size} values in parallel")
    print(f"Tensor shape: {result.shape}")
    print()
    print("First 15 of each batch:")
    for i in range(3):
        print(f"  Batch {i}: {' '.join(result[i, :15])}")

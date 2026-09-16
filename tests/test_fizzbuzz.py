"""
Every tensor representation must agree with the naive reference on every input.
"""

import numpy as np
import pytest

import dimensional_representations as dims
from fizzbuzz import DECODER, PATTERN, create_pattern, fizzbuzz
from fizzbuzz_batched import fizzbuzz_batched
from fizzbuzz_compact import PATTERN_COMPACT
from fizzbuzz_compact import fizzbuzz as fizzbuzz_compact


def reference(n):
    """The FizzBuzz everyone writes in the interview."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


N_VALUES = [1, 2, 14, 15, 16, 30, 100, 1000]


# --- Pattern vector -----------------------------------------------------------


def test_pattern_has_period_15():
    assert PATTERN.shape == (15,)
    assert PATTERN[-1] == 3  # 15 is the only FizzBuzz in one period


def test_pattern_matches_generated_pattern():
    pattern, decoder = create_pattern([(3, "Fizz"), (5, "Buzz")])
    np.testing.assert_array_equal(pattern, PATTERN)
    assert list(decoder) == list(DECODER)


@pytest.mark.parametrize("n", N_VALUES)
def test_pattern_vector_matches_reference(n):
    assert list(fizzbuzz(n)) == reference(n)


def test_fizzbuzz_zero_is_empty():
    assert len(fizzbuzz(0)) == 0
    assert len(fizzbuzz_compact(0)) == 0
    assert len(dims.fizzbuzz_binary(0)) == 0
    assert len(dims.fizzbuzz_modular(0)) == 0
    assert len(dims.fizzbuzz_vector(0)) == 0


# --- Compact 2x2 matrix -------------------------------------------------------


def test_compact_matrix_is_2x2():
    assert PATTERN_COMPACT.shape == (2, 2)
    # Rows index divisibility by 3, columns by 5.
    assert PATTERN_COMPACT[0, 0] == 0  # neither -> number
    assert PATTERN_COMPACT[1, 0] == 1  # by 3 -> Fizz
    assert PATTERN_COMPACT[0, 1] == 2  # by 5 -> Buzz
    assert PATTERN_COMPACT[1, 1] == 3  # both -> FizzBuzz


@pytest.mark.parametrize("n", N_VALUES)
def test_compact_matrix_matches_reference(n):
    assert list(fizzbuzz_compact(n)) == reference(n)


# --- Batched 3D tensor --------------------------------------------------------


@pytest.mark.parametrize(("batch", "length"), [(1, 15), (3, 10), (10, 100), (7, 13)])
def test_batched_tensor_matches_reference(batch, length):
    result, div_tensor, nums = fizzbuzz_batched(batch, length)
    assert result.shape == (batch, length)
    assert div_tensor.shape == (batch, length, 2)
    assert nums[0, 0] == 1
    assert nums[-1, -1] == batch * length
    assert list(result.ravel()) == reference(batch * length)


def test_batched_offset_shifts_range():
    result, _, nums = fizzbuzz_batched(2, 5, offset=100)
    assert nums[0, 0] == 101
    assert list(result.ravel()) == reference(110)[100:]


def test_divisibility_tensor_encodes_divisors():
    _, div_tensor, nums = fizzbuzz_batched(2, 15)
    np.testing.assert_array_equal(div_tensor[..., 0], (nums % 3 == 0).astype(int))
    np.testing.assert_array_equal(div_tensor[..., 1], (nums % 5 == 0).astype(int))


# --- dimensional_representations agrees with the standalone modules ----------


@pytest.mark.parametrize("n", N_VALUES)
def test_dimensional_representations_agree(n):
    expected = reference(n)
    assert list(dims.fizzbuzz_binary(n)) == expected
    assert list(dims.fizzbuzz_modular(n)) == expected
    assert list(dims.fizzbuzz_vector(n)) == expected


def test_dimensional_batched_agrees():
    result, div_matrix = dims.fizzbuzz_batched(3, 20)
    assert div_matrix.shape == (3, 20, 2)
    assert list(result.ravel()) == reference(60)


# --- Generalization -----------------------------------------------------------


def test_create_pattern_period_is_lcm():
    pattern, decoder = create_pattern([(3, "Fizz"), (5, "Buzz"), (7, "Bazz")])
    assert len(pattern) == 105
    assert len(decoder) == 8
    assert decoder[7] == "FizzBuzzBazz"
    assert decoder[0] == "{}"


def test_create_pattern_period_is_lcm_not_product():
    pattern, _ = create_pattern([(4, "Four"), (6, "Six")])
    assert len(pattern) == 12  # lcm(4, 6), not 24


def test_create_pattern_single_divisor():
    pattern, decoder = create_pattern([(7, "Bazz")])
    assert len(pattern) == 7
    assert list(pattern) == [0, 0, 0, 0, 0, 0, 1]
    assert list(decoder) == ["{}", "Bazz"]


def test_create_pattern_label_order_follows_divisor_order():
    _, decoder = create_pattern([(5, "Buzz"), (3, "Fizz")])
    assert decoder[3] == "BuzzFizz"


def test_create_pattern_generalized_output():
    pattern, decoder = create_pattern([(3, "Fizz"), (5, "Buzz"), (7, "Bazz")])
    nums = np.arange(1, 106)
    categories = pattern[(nums - 1) % len(pattern)]
    result = decoder[categories]

    for n, label in zip(nums, result, strict=True):
        expected = "".join(
            name for d, name in [(3, "Fizz"), (5, "Buzz"), (7, "Bazz")] if n % d == 0
        )
        assert label == (expected or "{}")

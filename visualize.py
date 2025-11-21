"""
Visualize FizzBuzz as a waveform - because why not?
"""

import numpy as np
import matplotlib.pyplot as plt
from fizzbuzz import PATTERN, fizzbuzz


def plot_pattern_waveform(periods=5, save_path="fizzbuzz_waveform.png"):
    """Plot the FizzBuzz pattern as a repeating waveform."""

    # Extend pattern for multiple periods
    extended_pattern = np.tile(PATTERN, periods)
    x = np.arange(len(extended_pattern)) + 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: The raw category signal
    ax = axes[0]
    ax.plot(x, extended_pattern, linewidth=2, color='steelblue')
    ax.scatter(x, extended_pattern, s=50, c=extended_pattern, cmap='viridis',
               edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('FizzBuzz Pattern Waveform (Categories)', fontsize=14, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Number', 'Fizz', 'Buzz', 'FizzBuzz'])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add period markers
    for i in range(periods + 1):
        ax.axvline(x=i * 15 + 1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        if i < periods:
            ax.text(i * 15 + 8, 3.2, f'Period {i+1}', ha='center',
                   fontsize=10, color='red', fontweight='bold')

    # Plot 2: Individual divisibility signals
    ax = axes[1]

    # Create divisibility signals
    div_by_3 = ((extended_pattern == 1) | (extended_pattern == 3)).astype(int)
    div_by_5 = ((extended_pattern == 2) | (extended_pattern == 3)).astype(int)

    ax.fill_between(x, 0, div_by_3, alpha=0.5, color='blue', label='Divisible by 3 (Fizz)')
    ax.fill_between(x, 0, -div_by_5, alpha=0.5, color='orange', label='Divisible by 5 (Buzz)')
    ax.plot(x, div_by_3, linewidth=2, color='blue')
    ax.plot(x, -div_by_5, linewidth=2, color='orange')

    ax.set_ylabel('Divisibility', fontsize=12, fontweight='bold')
    ax.set_title('Component Signals: Divisibility by 3 and 5', fontsize=14, fontweight='bold')
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Div by 5', '0', 'Div by 3'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Add period markers
    for i in range(periods + 1):
        ax.axvline(x=i * 15 + 1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Plot 3: Binary representation (as a spectrogram-style viz)
    ax = axes[2]

    # Create binary matrix: each row is a bit
    binary_matrix = np.array([
        div_by_3,
        div_by_5
    ])

    im = ax.imshow(binary_matrix, aspect='auto', cmap='hot', interpolation='nearest',
                   extent=[0.5, len(extended_pattern) + 0.5, -0.5, 1.5])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Div by 3', 'Div by 5'])
    ax.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Divisor Check', fontsize=12, fontweight='bold')
    ax.set_title('Binary Divisibility Matrix', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['False', 'True'])

    # Add period markers
    for i in range(periods + 1):
        ax.axvline(x=i * 15 + 0.5, color='cyan', linestyle='--', alpha=0.7, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Waveform saved to {save_path}")

    return fig


def plot_frequency_spectrum(n=1000, save_path="fizzbuzz_fft.png"):
    """
    Analyze the frequency spectrum of FizzBuzz.
    Because this is getting ridiculous.
    """

    # Generate long sequence
    categories = PATTERN[np.arange(n) % 15]

    # Compute FFT
    fft = np.fft.fft(categories)
    freqs = np.fft.fftfreq(n)
    magnitude = np.abs(fft)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain
    x = np.arange(n)
    ax1.plot(x[:150], categories[:150], linewidth=1.5, color='steelblue')
    ax1.scatter(x[:150], categories[:150], s=20, c=categories[:150],
               cmap='viridis', edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Category', fontsize=11, fontweight='bold')
    ax1.set_title('Time Domain: First 150 Values', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Frequency domain
    # Only plot positive frequencies
    positive_freqs = freqs[:n//2]
    positive_magnitude = magnitude[:n//2]

    ax2.stem(positive_freqs[:50], positive_magnitude[:50], linefmt='steelblue',
            markerfmt='o', basefmt='gray')
    ax2.set_xlabel('Frequency (cycles per sample)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
    ax2.set_title('Frequency Spectrum (FFT)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight the fundamental frequency (1/15)
    fundamental = 1/15
    ax2.axvline(x=fundamental, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Fundamental (1/15 ≈ {fundamental:.4f})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Frequency spectrum saved to {save_path}")

    return fig


def plot_2d_heatmap(size=15, save_path="fizzbuzz_2d.png"):
    """
    Visualize FizzBuzz as a 2D heatmap.
    What if FizzBuzz was a texture?
    """

    # Create 2D grid
    total = size * size
    categories = PATTERN[np.arange(total) % 15]
    grid = categories.reshape(size, size)

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(grid, cmap='viridis', interpolation='nearest')

    # Add text annotations
    for i in range(size):
        for j in range(size):
            n = i * size + j + 1
            cat = grid[i, j]

            # Get the output string
            if cat == 0:
                text = str(n)
            elif cat == 1:
                text = "Fizz"
            elif cat == 2:
                text = "Buzz"
            else:
                text = "FB"

            color = 'white' if cat >= 2 else 'black'
            ax.text(j, i, text, ha="center", va="center",
                   color=color, fontsize=8, fontweight='bold')

    ax.set_title(f'FizzBuzz as a {size}×{size} Texture',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(1, size + 1))
    ax.set_yticklabels(range(1, size + 1))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Number', 'Fizz', 'Buzz', 'FizzBuzz'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"2D heatmap saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Generating FizzBuzz visualizations...")
    print("=" * 50)

    plot_pattern_waveform(periods=5)
    plot_frequency_spectrum(n=1000)
    plot_2d_heatmap(size=20)

    print("\nDone! This was absolutely ridiculous and I loved it.")

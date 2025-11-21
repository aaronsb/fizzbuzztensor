"""
Visualization for the Batched 3D Tensor Approach
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fizzbuzz_batched import fizzbuzz_batched


def plot_3d_tensor_structure(save_path="docs/images/fizzbuzz_3d_structure.png"):
    """
    Visualize the 3D tensor structure for batched computation.
    """
    # Generate small batch for visualization
    batch_size, seq_len = 5, 20
    result, div_tensor, nums = fizzbuzz_batched(batch_size, seq_len)

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 3D scatter showing the tensor dimensions
    ax1 = fig.add_subplot(221, projection='3d')

    # Create coordinates for scatter plot
    batches, sequences, divisors = [], [], []
    colors = []

    for b in range(batch_size):
        for s in range(seq_len):
            for d in range(2):
                batches.append(b)
                sequences.append(s)
                divisors.append(d)
                # Color by divisibility
                if div_tensor[b, s, d]:
                    colors.append('red' if d == 0 else 'blue')
                else:
                    colors.append('lightgray')

    ax1.scatter(batches, sequences, divisors, c=colors, alpha=0.6, s=20)
    ax1.set_xlabel('Batch', fontweight='bold')
    ax1.set_ylabel('Sequence Position', fontweight='bold')
    ax1.set_zlabel('Divisor (0=3, 1=5)', fontweight='bold')
    ax1.set_title('3D Tensor Structure\n(batch × sequence × divisors)',
                  fontweight='bold', fontsize=12)

    # Plot 2: Heatmap of categories per batch
    ax2 = fig.add_subplot(222)

    # Create category matrix
    categories = np.zeros((batch_size, seq_len))
    for b in range(batch_size):
        for s in range(seq_len):
            div_3 = div_tensor[b, s, 0]
            div_5 = div_tensor[b, s, 1]
            categories[b, s] = div_3 * 1 + div_5 * 2

    im = ax2.imshow(categories, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Sequence Position', fontweight='bold')
    ax2.set_ylabel('Batch', fontweight='bold')
    ax2.set_title('Category Heatmap Across Batches', fontweight='bold', fontsize=12)

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Number', 'Fizz', 'Buzz', 'FizzBuzz'])

    # Plot 3: Divisibility by 3 heatmap
    ax3 = fig.add_subplot(223)

    im3 = ax3.imshow(div_tensor[:, :, 0], aspect='auto', cmap='Reds',
                     interpolation='nearest', vmin=0, vmax=1)
    ax3.set_xlabel('Sequence Position', fontweight='bold')
    ax3.set_ylabel('Batch', fontweight='bold')
    ax3.set_title('Divisibility by 3 (Tensor[:,:,0])', fontweight='bold', fontsize=12)

    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(['False', 'True'])

    # Plot 4: Divisibility by 5 heatmap
    ax4 = fig.add_subplot(224)

    im4 = ax4.imshow(div_tensor[:, :, 1], aspect='auto', cmap='Blues',
                     interpolation='nearest', vmin=0, vmax=1)
    ax4.set_xlabel('Sequence Position', fontweight='bold')
    ax4.set_ylabel('Batch', fontweight='bold')
    ax4.set_title('Divisibility by 5 (Tensor[:,:,1])', fontweight='bold', fontsize=12)

    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_ticks([0, 1])
    cbar4.set_ticklabels(['False', 'True'])

    plt.suptitle(f'Batched 3D Tensor: Shape {div_tensor.shape}',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"3D tensor structure visualization saved to {save_path}")

    return fig


def plot_parallel_computation(save_path="docs/images/fizzbuzz_parallel.png"):
    """
    Visualize the parallel computation aspect of batched approach.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Generate 3 batches
    batch_size, seq_len = 3, 30
    result, div_tensor, nums = fizzbuzz_batched(batch_size, seq_len)

    for idx, ax in enumerate(axes):
        # Get categories for this batch
        categories = div_tensor[idx, :, 0] * 1 + div_tensor[idx, :, 1] * 2
        positions = np.arange(seq_len)

        # Plot as bar chart
        colors_map = {0: '#E0E0E0', 1: '#FFA726', 2: '#66BB6A', 3: '#FFEB3B'}
        bar_colors = [colors_map[int(c)] for c in categories]

        bars = ax.bar(positions, np.ones(seq_len), color=bar_colors,
                     edgecolor='black', linewidth=0.5)

        # Add text labels
        for i, (pos, cat) in enumerate(zip(positions, categories)):
            if i % 2 == 0:  # Label every other position to avoid crowding
                ax.text(pos, 0.5, result[idx, i], ha='center', va='center',
                       fontsize=8, fontweight='bold')

        start_num = nums[idx, 0]
        end_num = nums[idx, -1]
        ax.set_ylabel(f'Batch {idx}\n({start_num}-{end_num})',
                     fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_xticks(positions[::5])
        ax.set_xticklabels([str(nums[idx, i]) for i in range(0, seq_len, 5)])
        ax.set_yticks([])

        if idx == 0:
            ax.set_title('Parallel Batch Computation', fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Position (number value)', fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E0E0E0', edgecolor='black', label='Number'),
        Patch(facecolor='#FFA726', edgecolor='black', label='Fizz'),
        Patch(facecolor='#66BB6A', edgecolor='black', label='Buzz'),
        Patch(facecolor='#FFEB3B', edgecolor='black', label='FizzBuzz')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Parallel computation visualization saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Generating batched 3D tensor visualizations...")
    print("=" * 50)

    plot_3d_tensor_structure()
    plot_parallel_computation()

    print("\nDone!")

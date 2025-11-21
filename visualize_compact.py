"""
Visualization for the Compact 2×2 Binary Matrix Approach
"""

import numpy as np
import matplotlib.pyplot as plt
from fizzbuzz_compact import PATTERN_COMPACT, DECODER


def plot_compact_matrix(save_path="docs/images/fizzbuzz_compact.png"):
    """
    Visualize the 2×2 compact binary matrix.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: The 2×2 matrix heatmap
    im = ax1.imshow(PATTERN_COMPACT, cmap='viridis', vmin=0, vmax=3,
                    interpolation='nearest')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = PATTERN_COMPACT[i, j]
            output = DECODER[value] if value > 0 else "Number"
            ax1.text(j, i, f'{value}\n({output})',
                    ha="center", va="center",
                    color="white" if value >= 2 else "black",
                    fontsize=14, fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['False', 'True'], fontsize=11)
    ax1.set_yticklabels(['False', 'True'], fontsize=11)
    ax1.set_xlabel('Divisible by 5?', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Divisible by 3?', fontsize=12, fontweight='bold')
    ax1.set_title('Compact Binary Matrix (2×2)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Number', 'Fizz', 'Buzz', 'FizzBuzz'])

    # Right plot: Comparison of storage requirements
    representations = ['Binary\nMatrix', '15-Element\nVector']
    elements = [4, 15]
    colors = ['#2E7D32', '#1565C0']

    bars = ax2.bar(representations, elements, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, elements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val} elements',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Storage (# of elements)', fontsize=12, fontweight='bold')
    ax2.set_title('Storage Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 18)
    ax2.grid(axis='y', alpha=0.3)

    # Add annotation
    ax2.text(0.5, 16, '73% reduction in storage',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Compact matrix visualization saved to {save_path}")

    return fig


def plot_decision_tree(save_path="docs/images/fizzbuzz_decision_tree.png"):
    """
    Visualize the binary decision tree for the compact approach.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Decision tree structure
    # Root
    ax.text(0.5, 0.9, 'n mod 3 == 0?', ha='center', fontsize=14,
           bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))

    # Level 1
    ax.text(0.25, 0.65, 'True', ha='center', fontsize=11, style='italic')
    ax.text(0.75, 0.65, 'False', ha='center', fontsize=11, style='italic')

    ax.text(0.25, 0.6, 'n mod 5 == 0?', ha='center', fontsize=13,
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(0.75, 0.6, 'n mod 5 == 0?', ha='center', fontsize=13,
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))

    # Level 2 - Outputs
    # Left branch (div by 3)
    ax.text(0.125, 0.35, 'False', ha='center', fontsize=10, style='italic')
    ax.text(0.375, 0.35, 'True', ha='center', fontsize=10, style='italic')

    ax.text(0.125, 0.3, 'Fizz', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFA726', edgecolor='black', linewidth=2))
    ax.text(0.375, 0.3, 'FizzBuzz', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFEB3B', edgecolor='black', linewidth=2))

    # Right branch (not div by 3)
    ax.text(0.625, 0.35, 'False', ha='center', fontsize=10, style='italic')
    ax.text(0.875, 0.35, 'True', ha='center', fontsize=10, style='italic')

    ax.text(0.625, 0.3, 'Number', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E0E0E0', edgecolor='black', linewidth=2))
    ax.text(0.875, 0.3, 'Buzz', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#66BB6A', edgecolor='black', linewidth=2))

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    # Root to level 1
    ax.annotate('', xy=(0.25, 0.63), xytext=(0.5, 0.87), arrowprops=arrow_props)
    ax.annotate('', xy=(0.75, 0.63), xytext=(0.5, 0.87), arrowprops=arrow_props)

    # Level 1 to level 2 (left)
    ax.annotate('', xy=(0.125, 0.33), xytext=(0.25, 0.57), arrowprops=arrow_props)
    ax.annotate('', xy=(0.375, 0.33), xytext=(0.25, 0.57), arrowprops=arrow_props)

    # Level 1 to level 2 (right)
    ax.annotate('', xy=(0.625, 0.33), xytext=(0.75, 0.57), arrowprops=arrow_props)
    ax.annotate('', xy=(0.875, 0.33), xytext=(0.75, 0.57), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)

    ax.set_title('Binary Decision Tree for Compact FizzBuzz', fontsize=16, fontweight='bold', pad=20)

    # Add matrix representation
    ax.text(0.5, 0.1, 'Matrix Representation: PATTERN[div_by_3][div_by_5]',
           ha='center', fontsize=12, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Decision tree visualization saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Generating compact 2×2 matrix visualizations...")
    print("=" * 50)

    plot_compact_matrix()
    plot_decision_tree()

    print("\nDone!")

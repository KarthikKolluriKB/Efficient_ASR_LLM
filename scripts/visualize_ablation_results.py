"""
Ablation Study Visualization Script
====================================
Compares WER vs Parameter Reduction for different encoder configurations.

Models compared:
- Whisper-small (12L) - baseline
- Whisper-base (6L) - comparison
- Ablation 10L, 8L, 6L - reduced layer versions
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# RESULTS DICTIONARY - UPDATE WITH TRUE VALUES
# =============================================================================
# Format: "model_name": {"layers": int, "test_wer": float, "params_millions": float}
#
# NOTE: We use ENCODER-ONLY params since we're using Whisper encoder + LLM architecture
# Full Whisper-small = 244M, but encoder only = ~88M
# Full Whisper-base = 74M, but encoder only = ~39M
#
# Whisper-small encoder breakdown:
#   - Conv layers: ~1.96M
#   - Positional embedding: ~1.15M  
#   - LayerNorm: ~0.002M
#   - Per transformer block: ~7.09M
#   - Total 12L: 1.96 + 1.15 + 0.002 + (12 * 7.09) = ~88.15M

RESULTS = {
    # Baseline models (encoder-only params)
    "whisper-small (12L)": {
        "layers": 12,
        "test_wer": 42.18,  # Actual test WER
        "params_millions": 88.15,  # Encoder only: 12 blocks
    },
    "whisper-base (6L)": {
        "layers": 6,
        "test_wer": 64.72,  # Actual test WER
        "params_millions": 39.0,  # Encoder only: 6 blocks (smaller dim)
    },
    # Ablation study models (whisper-small encoder with reduced layers)
    # Base params (conv + pos_emb + ln) = ~3.11M, each block = ~7.09M
    "ablation-10L": {
        "layers": 10,
        "test_wer": 48.46,  # Actual test WER
        "params_millions": 74.0,  # 3.11 + (10 * 7.09) = ~74M
    },
    "ablation-8L": {
        "layers": 8,
        "test_wer": 52.83,  # Actual test WER
        "params_millions": 59.8,  # 3.11 + (8 * 7.09) = ~60M
    },
    "ablation-6L": {
        "layers": 6,
        "test_wer": 56.74,  # Actual test WER
        "params_millions": 45.6,  # 3.11 + (6 * 7.09) = ~46M
    },
}

# Reference baselines (not trained by us, for comparison only)
REFERENCE_BASELINES = {
    "Zero-shot Whisper-small": {"wer": 38.9, "description": "Vanilla Whisper-small on Danish CV"},
    "Fine-tuned Whisper-small": {"wer": 32.3, "description": "Fine-tuned with robust data"},
}

# Baseline for comparison (whisper-small 12L)
BASELINE_MODEL = "whisper-small (12L)"


def calculate_param_reduction(results: dict, baseline: str) -> dict:
    """Calculate parameter reduction percentage relative to baseline."""
    baseline_params = results[baseline]["params_millions"]
    
    reductions = {}
    for model, data in results.items():
        reduction_pct = (1 - data["params_millions"] / baseline_params) * 100
        reductions[model] = reduction_pct
    
    return reductions


def create_wer_vs_param_reduction_plot(results: dict, baseline: str, save_path: str = None):
    """
    Create a scatter plot of WER vs Parameter Reduction.
    
    Args:
        results: Dictionary with model results
        baseline: Name of the baseline model
        save_path: Optional path to save the figure
    """
    param_reductions = calculate_param_reduction(results, baseline)
    
    # Prepare data
    models = list(results.keys())
    wers = [results[m]["test_wer"] for m in models]
    reductions = [param_reductions[m] for m in models]
    layers = [results[m]["layers"] for m in models]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color scheme
    colors = {
        "whisper-small (12L)": "#2ecc71",  # Green - baseline
        "whisper-base (6L)": "#3498db",     # Blue - comparison
        "ablation-10L": "#e74c3c",          # Red - ablation
        "ablation-8L": "#e67e22",           # Orange - ablation
        "ablation-6L": "#9b59b6",           # Purple - ablation
    }
    
    # ==========================================================================
    # Plot 1: WER vs Parameter Reduction
    # ==========================================================================
    ax1 = axes[0]
    
    for model in models:
        color = colors.get(model, "#95a5a6")
        marker = "s" if "baseline" in model.lower() or "whisper" in model.lower() else "o"
        size = 200 if model == baseline else 150
        
        ax1.scatter(
            param_reductions[model],
            results[model]["test_wer"],
            c=color,
            s=size,
            marker=marker,
            label=model,
            edgecolors="black",
            linewidths=1.5,
            zorder=3
        )
        
        # Add annotation
        ax1.annotate(
            f'{results[model]["test_wer"]:.1f}%',
            (param_reductions[model], results[model]["test_wer"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add reference line from baseline
    baseline_wer = results[baseline]["test_wer"]
    ax1.axhline(y=baseline_wer, color='gray', linestyle='--', alpha=0.5, label=f'Our Baseline ({baseline_wer:.1f}%)')
    
    # Add reference line for zero-shot Whisper-small
    ax1.axhline(y=38.9, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Zero-shot Whisper-small (38.9%)')
    
    ax1.set_xlabel("Parameter Reduction (%)", fontsize=12)
    ax1.set_ylabel("Test WER (%)", fontsize=12)
    ax1.set_title("WER vs Parameter Reduction\n(Ablation Study)", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, max(reductions) + 10)
    
    # ==========================================================================
    # Plot 2: WER vs Number of Encoder Layers
    # ==========================================================================
    ax2 = axes[1]
    
    for model in models:
        color = colors.get(model, "#95a5a6")
        marker = "s" if "whisper" in model.lower() else "o"
        size = 200 if model == baseline else 150
        
        ax2.scatter(
            results[model]["layers"],
            results[model]["test_wer"],
            c=color,
            s=size,
            marker=marker,
            label=model,
            edgecolors="black",
            linewidths=1.5,
            zorder=3
        )
        
        # Add annotation
        ax2.annotate(
            f'{results[model]["test_wer"]:.1f}%',
            (results[model]["layers"], results[model]["test_wer"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Connect ablation points with a trend line
    ablation_models = [m for m in models if "ablation" in m.lower()]
    if ablation_models:
        ablation_layers = [results[m]["layers"] for m in ablation_models]
        ablation_wers = [results[m]["test_wer"] for m in ablation_models]
        # Sort by layers
        sorted_pairs = sorted(zip(ablation_layers, ablation_wers))
        sorted_layers, sorted_wers = zip(*sorted_pairs)
        ax2.plot(sorted_layers, sorted_wers, 'r--', alpha=0.5, linewidth=2, label='Ablation Trend')
    
    # Add reference line for zero-shot Whisper-small
    ax2.axhline(y=38.9, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Zero-shot Whisper-small (38.9%)')
    
    ax2.set_xlabel("Number of Encoder Layers", fontsize=12)
    ax2.set_ylabel("Test WER (%)", fontsize=12)
    ax2.set_title("WER vs Encoder Layers\n(Layer Reduction Impact)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(4, 14, 2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def create_bar_comparison_plot(results: dict, baseline: str, save_path: str = None):
    """
    Create a bar chart comparing WER and parameter counts.
    
    Args:
        results: Dictionary with model results
        baseline: Name of the baseline model
        save_path: Optional path to save the figure
    """
    param_reductions = calculate_param_reduction(results, baseline)
    
    models = list(results.keys())
    wers = [results[m]["test_wer"] for m in models]
    params = [results[m]["params_millions"] for m in models]
    reductions = [param_reductions[m] for m in models]
    
    # Sort by number of layers (descending)
    sorted_indices = np.argsort([results[m]["layers"] for m in models])[::-1]
    models = [models[i] for i in sorted_indices]
    wers = [wers[i] for i in sorted_indices]
    params = [params[i] for i in sorted_indices]
    reductions = [reductions[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#2ecc71' if m == baseline else '#3498db' if 'base' in m else '#e74c3c' for m in models]
    
    # ==========================================================================
    # Plot 1: WER Comparison
    # ==========================================================================
    ax1 = axes[0]
    bars1 = ax1.barh(models, wers, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, wer in zip(bars1, wers):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{wer:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel("Test WER (%)", fontsize=12)
    ax1.set_title("Test WER Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(wers) * 1.2)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ==========================================================================
    # Plot 2: Parameter Count & Reduction
    # ==========================================================================
    ax2 = axes[1]
    bars2 = ax2.barh(models, params, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels with reduction percentage
    for bar, param, red in zip(bars2, params, reductions):
        label = f'{param:.0f}M'
        if red > 0:
            label += f' (-{red:.0f}%)'
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel("Parameters (Millions)", fontsize=12)
    ax2.set_title("Parameter Count Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(params) * 1.3)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def print_summary_table(results: dict, baseline: str):
    """Print a formatted summary table of results."""
    param_reductions = calculate_param_reduction(results, baseline)
    baseline_wer = results[baseline]["test_wer"]
    zero_shot_wer = 38.9  # Zero-shot Whisper-small on Danish CV
    
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*90)
    print(f"\nReference: Zero-shot Whisper-small on Danish Common Voice = {zero_shot_wer}% WER")
    print(f"Reference: Fine-tuned Whisper-small (robust data) = 32.3% WER\n")
    print("-"*90)
    print(f"{'Model':<25} {'Layers':>8} {'Params (M)':>12} {'Reduction':>12} {'Test WER':>10} {'vs Zero-shot':>12}")
    print("-"*90)
    
    # Sort by layers descending
    sorted_models = sorted(results.keys(), key=lambda m: results[m]["layers"], reverse=True)
    
    for model in sorted_models:
        data = results[model]
        reduction = param_reductions[model]
        wer_delta_vs_zeroshot = data["test_wer"] - zero_shot_wer
        
        delta_str = f"+{wer_delta_vs_zeroshot:.1f}%" if wer_delta_vs_zeroshot > 0 else f"{wer_delta_vs_zeroshot:.1f}%"
        
        print(f"{model:<25} {data['layers']:>8} {data['params_millions']:>12.1f} "
              f"{reduction:>11.1f}% {data['test_wer']:>9.1f}% {delta_str:>12}")
    
    print("="*90)
    
    # Print efficiency analysis
    print("\n" + "="*90)
    print("EFFICIENCY ANALYSIS")
    print("="*90)
    
    baseline_params = results[baseline]["params_millions"]
    for model in sorted_models:
        if model == baseline:
            continue
        data = results[model]
        reduction = param_reductions[model]
        wer_increase = data["test_wer"] - baseline_wer
        
        if reduction > 0:
            efficiency_ratio = wer_increase / reduction
            print(f"{model}: {reduction:.1f}% param reduction → {wer_increase:.1f}% WER increase "
                  f"(ratio: {efficiency_ratio:.2f}% WER per 1% params)")
    
    print("="*90 + "\n")


def main():
    """Main function to generate all visualizations."""
    print("Generating Ablation Study Visualizations...")
    print(f"Baseline model: {BASELINE_MODEL}")
    
    # Print summary table
    print_summary_table(RESULTS, BASELINE_MODEL)
    
    # Create scatter plot (WER vs Param Reduction)
    create_wer_vs_param_reduction_plot(
        RESULTS, 
        BASELINE_MODEL,
        save_path="outputs/ablation_wer_vs_reduction.png"
    )
    
    # Create bar comparison plot
    create_bar_comparison_plot(
        RESULTS,
        BASELINE_MODEL,
        save_path="outputs/ablation_bar_comparison.png"
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

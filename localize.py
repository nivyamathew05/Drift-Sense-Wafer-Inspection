import argparse
import os
import json
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.localizer import DriftSenseLocalizer

def run_evaluation(manifest_path, output_summary_path="results/summary_metrics.json", plot_path="results/evaluation_plots.png", failure_plot_path="results/failure_analysis.png"):
    df = pd.read_csv(manifest_path)
    localizer = DriftSenseLocalizer()

    pred_xs, pred_ys = [], []
    errors, runtimes, confidences = [], [], []
    pass_5px, pass_4px, pass_2px, pass_1px, pass_subpixel = [], [], [], [], []

    print(f"Running localization evaluation on {len(df)} pairs...", flush=True)

    for idx, row in df.iterrows():
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            print(f"Error loading pair {row['pair_id']}", flush=True)
            continue

        px, py, conf, rt = localizer.localize(ref_img, search_img)

        tx, ty = float(row["true_x"]), float(row["true_y"])
        err = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)

        if (idx + 1) % 5 == 0 or (idx + 1) == 1:
            print(f"Evaluated pair {idx+1}/{len(df)}: Error = {err:.2f}px, Runtime = {rt*1000:.1f}ms", flush=True)

        pred_xs.append(round(px, 4))
        pred_ys.append(round(py, 4))
        errors.append(round(err, 4))
        runtimes.append(round(rt, 5))
        confidences.append(round(conf, 4))

        pass_5px.append(int(err <= 5.0))
        pass_4px.append(int(err <= 4.0))
        pass_2px.append(int(err <= 2.0))
        pass_1px.append(int(err <= 1.0))
        pass_subpixel.append(int(err < 0.5))

    df["pred_x"] = pred_xs
    df["pred_y"] = pred_ys
    df["euclidean_error_px"] = errors
    df["confidence"] = confidences
    df["runtime_sec"] = runtimes
    df["pass_5px"] = pass_5px
    df["pass_4px"] = pass_4px
    df["pass_2px"] = pass_2px
    df["pass_1px"] = pass_1px
    df["pass_subpixel"] = pass_subpixel

    df.to_csv(manifest_path, index=False)

    # Compute aggregate statistics
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    worst_err = float(np.max(errors))
    mean_runtime = float(np.mean(runtimes))

    rate_5px = float(np.mean(pass_5px) * 100.0)
    rate_4px = float(np.mean(pass_4px) * 100.0)
    rate_2px = float(np.mean(pass_2px) * 100.0)
    rate_1px = float(np.mean(pass_1px) * 100.0)
    rate_subpixel = float(np.mean(pass_subpixel) * 100.0)

    summary = {
        "total_pairs_evaluated": len(df),
        "mean_euclidean_error_px": round(mean_err, 4),
        "median_euclidean_error_px": round(median_err, 4),
        "worst_case_error_px": round(worst_err, 4),
        "pass_rate_5px_percent": round(rate_5px, 2),
        "pass_rate_4px_percent": round(rate_4px, 2),
        "pass_rate_2px_percent": round(rate_2px, 2),
        "pass_rate_1px_percent": round(rate_1px, 2),
        "pass_rate_subpixel_percent": round(rate_subpixel, 2),
        "mean_runtime_seconds_per_pair": round(mean_runtime, 5)
    }

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print("====================================================\n", flush=True)

    generate_plots(df, plot_path)
    generate_failure_visual(df, failure_plot_path)

    return summary

def generate_plots(df, plot_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Drift-Sense AI Navigation Error Recovery Benchmark Results", fontsize=14, fontweight='bold')

    thresholds = np.linspace(0, 10, 100)
    pass_rates = [np.mean(df["euclidean_error_px"] <= t) * 100 for t in thresholds]
    
    axes[0].plot(thresholds, pass_rates, color='#1f77b4', linewidth=2.5)
    axes[0].axvline(1.0, color='r', linestyle='--', label='1px Threshold')
    axes[0].axvline(5.0, color='g', linestyle='--', label='5px Threshold')
    axes[0].set_title("Localization Pass Rate vs Tolerance")
    axes[0].set_xlabel("Euclidean Error Tolerance (pixels)")
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].hist(df["euclidean_error_px"], bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(df["euclidean_error_px"]), color='red', linestyle='-', linewidth=2, label=f'Mean Error ({np.mean(df["euclidean_error_px"]):.2f}px)')
    axes[1].set_title("Euclidean Error Distribution")
    axes[1].set_xlabel("Euclidean Error (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].bar(range(len(df)), df["runtime_sec"] * 1000.0, color='#ff7f0e', alpha=0.8)
    axes[2].axhline(np.mean(df["runtime_sec"]) * 1000.0, color='blue', linestyle='--', label=f'Mean: {np.mean(df["runtime_sec"])*1000:.1f}ms')
    axes[2].set_title("Inference Computation Time Per Image Pair")
    axes[2].set_xlabel("Pair Index")
    axes[2].set_ylabel("Runtime (milliseconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Evaluation plots saved to: {plot_path}", flush=True)

def generate_failure_visual(df, failure_plot_path):
    worst_row = df.loc[df["euclidean_error_px"].idxmax()]

    ref_img = cv2.imread(worst_row["reference_path"])
    search_img = cv2.imread(worst_row["search_path"])

    tx, ty = int(round(worst_row["true_x"])), int(round(worst_row["true_y"]))
    px, py = int(round(worst_row["pred_x"])), int(round(worst_row["pred_y"]))

    vis_search = search_img.copy()
    cv2.circle(vis_search, (tx, ty), 16, (0, 255, 0), 3)
    cv2.drawMarker(vis_search, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(vis_search, "True Target", (tx + 20, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.circle(vis_search, (px, py), 16, (0, 0, 255), 3)
    cv2.drawMarker(vis_search, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.putText(vis_search, f"Pred ({worst_row['euclidean_error_px']:.2f}px err)", (px + 20, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Failure Case Analysis - {worst_row['pair_id']} ({worst_row['architecture']})", fontsize=14, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"100x Reference Pattern (1000x1000)")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(vis_search, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"10x Search Image with Ground Truth vs Prediction")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(failure_plot_path, dpi=200)
    plt.close()
    print(f"Failure visual saved to: {failure_plot_path}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Drift-Sense Localization on Benchmark Dataset.")
    parser.add_argument("--manifest", type=str, default="results/dataset_manifest.csv", help="Path to manifest CSV.")
    parser.add_argument("--ref", type=str, default=None, help="Optional single reference image path.")
    parser.add_argument("--search", type=str, default=None, help="Optional single search image path.")
    args = parser.parse_args()

    if args.ref and args.search:
        localizer = DriftSenseLocalizer()
        ref_img = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
        px, py, conf, rt = localizer.localize(ref_img, search_img)
        print(f"\nSingle Pair Result: Predicted Center = ({px:.2f}, {py:.2f}), Confidence = {conf:.4f}, Runtime = {rt*1000:.2f} ms\n", flush=True)
    else:
        run_evaluation(args.manifest)

if __name__ == "__main__":
    main()

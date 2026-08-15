import argparse
import os
import json
import pandas as pd
import cv2
import numpy as np
from src.sem_generator import SEMDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic SEM Image Pairs for Drift-Sense Benchmark.")
    parser.add_argument("--num_pairs", type=int, default=35, help="Number of benchmark image pairs to generate (default: 35).")
    parser.add_argument("--output_dir", type=str, default="results/dataset", help="Directory to save pair images.")
    parser.add_argument("--manifest", type=str, default="results/dataset_manifest.csv", help="Path to output CSV manifest.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    generator = SEMDataGenerator()

    manifest_rows = []

    print(f"Generating {args.num_pairs} synthetic SEM benchmark pairs...", flush=True)

    for i in range(1, args.num_pairs + 1):
        seed = 1000 + i
        architecture = "DRAM" if i % 2 == 1 else "FinFET"
        
        # Varied noise severity, scale ratio, and rotation
        noise_severity = round(0.5 + (i % 5) * 0.25, 2)
        scale_ratio = round(9.0 + (i % 9) * 0.25, 2)
        rotation_deg = round(-2.0 + (i % 5) * 1.0, 2)

        ref_img, search_img, meta = generator.generate_pair(
            seed=seed,
            architecture=architecture,
            noise_severity=noise_severity,
            scale_ratio=scale_ratio,
            rotation_deg=rotation_deg
        )

        ref_rel_path = os.path.join(args.output_dir, f"pair_{i:03d}_ref.png")
        search_rel_path = os.path.join(args.output_dir, f"pair_{i:03d}_search.png")
        meta_rel_path = os.path.join(args.output_dir, f"pair_{i:03d}_meta.json")

        if i % 5 == 0 or i == 1 or i == args.num_pairs:
            print(f"Generated pair {i}/{args.num_pairs} ({architecture}, noise={noise_severity}, scale={scale_ratio})", flush=True)

        cv2.imwrite(ref_rel_path, ref_img)
        cv2.imwrite(search_rel_path, search_img)

        with open(meta_rel_path, "w") as f:
            json.dump(meta, f, indent=2)

        manifest_rows.append({
            "pair_id": f"pair_{i:03d}",
            "architecture": architecture,
            "reference_path": ref_rel_path,
            "search_path": search_rel_path,
            "metadata_path": meta_rel_path,
            "true_x": meta["true_x"],
            "true_y": meta["true_y"],
            "scale_ratio": meta["scale_ratio"],
            "rotation_deg": meta["rotation_deg"],
            "noise_severity": meta["noise_severity"],
            "seed": seed
        })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(args.manifest, index=False)
    print(f"Dataset generation complete! Manifest saved to: {args.manifest}", flush=True)

if __name__ == "__main__":
    main()

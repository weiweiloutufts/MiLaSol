"""
Convert FASTA sequences to three embedding files (ESM2, RayGun, ProtT5).

This script reads a FASTA file and generates three CSV files containing
embeddings for each sequence using different models:
- esm2.csv: ESM2 embeddings
- raygun.csv: RayGun embeddings derived from ESM2
- prott5.csv: ProtT5 embeddings
- seqs.txt: Original sequences

Usage:
    python fasta_to_embeddings.py <input_fasta> <output_dir>
    
Example:
    python fasta_to_embeddings.py data/input.fasta embeddings_output/
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from milasol.data.seq_to_embeddings_new import (
    _esm_perres_and_pooled,
    _load_sequences,
    _prott5_pooled,
    _raygun_from_esm,
    _resolve_cache_dir,
    transfer_to_three_embedding_csvs,
)


def fasta_to_embeddings(
    input_file: str,
    output_dir: str,
    cache_dir: Optional[str] = None,
    esm_batch_size: int = 8,
    prot_batch_size: int = 4,
    models: str = "all",
    verbose: bool = True,
) -> None:
    """
    Convert FASTA file to three embedding CSV files.
    
    Args:
        input_file: Path to input FASTA file
        output_dir: Path to output directory
        cache_dir: Optional model cache directory
        esm_batch_size: Batch size for ESM2 model
        prot_batch_size: Batch size for ProtT5 model
        verbose: Whether to print progress information
    
    Raises:
        FileNotFoundError: If input file does not exist
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if verbose:
        print(f"[Info] Reading FASTA file: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)

    if models == "all":
        artifacts = transfer_to_three_embedding_csvs(
            source=str(input_path),
            seq_col="sequence",  # Not used for FASTA input
            out_dir=output_dir,
            cache_dir=cache_dir,
            esm_batch_size=esm_batch_size,
            prot_batch_size=prot_batch_size,
            write_to_disk=True,
        )

        if verbose:
            print(f"\n[Success] Embedding generation complete!")
            print(f"[Info] Generated {len(artifacts.sequences)} sequence embeddings")
            print(f"[Info] Output files:")
            print(f"  - {output_path / 'seqs.txt'}")
            print(f"  - {output_path / 'esm2.csv'}")
            print(f"  - {output_path / 'raygun.csv'}")
            print(f"  - {output_path / 'prott5.csv'}")
        return

    if models not in {"prott5", "raygun", "esm2"}:
        raise ValueError(f"Unsupported models option: {models}")

    seqs = _load_sequences(str(input_path), seq_col="sequence")
    resolved_cache_dir = _resolve_cache_dir(cache_dir)

    seq_path = output_path / "seqs.txt"
    if verbose:
        print(f"[Info] Writing {len(seqs)} sequences to {seq_path}")
    with open(seq_path, "w") as f:
        for seq in seqs:
            f.write(seq + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if models == "prott5":
        prot_path = output_path / "prott5.csv"
        prot_mat = _prott5_pooled(
            seqs, device=device, cache_dir=resolved_cache_dir, batch_size=prot_batch_size
        )
        df_prott5 = pd.DataFrame(
            prot_mat, columns=[f"e{i}" for i in range(prot_mat.shape[1])]
        )
        if verbose:
            print(f"[Info] Writing ProtT5 embeddings with shape {df_prott5.shape} to {prot_path}")
        df_prott5.to_csv(prot_path, index=False, header=True)

        if verbose:
            print("\n[Success] ProtT5 embedding generation complete!")
            print(f"[Info] Generated {len(seqs)} ProtT5 embeddings")
            print(f"[Info] Output files:")
            print(f"  - {seq_path}")
            print(f"  - {prot_path}")
        return

    per_res_list, esm_pooled = _esm_perres_and_pooled(
        seqs, device=device, batch_size=esm_batch_size
    )

    if models == "esm2":
        esm_path = output_path / "esm2.csv"
        df_esm = pd.DataFrame(
            esm_pooled, columns=[f"e{i}" for i in range(esm_pooled.shape[1])]
        )
        if verbose:
            print(f"[Info] Writing ESM embeddings with shape {df_esm.shape} to {esm_path}")
        df_esm.to_csv(esm_path, index=False, header=True)

        if verbose:
            print("\n[Success] ESM2 embedding generation complete!")
            print(f"[Info] Generated {len(seqs)} ESM2 embeddings")
            print(f"[Info] Output files:")
            print(f"  - {seq_path}")
            print(f"  - {esm_path}")
        return

    ray_path = output_path / "raygun.csv"
    ray_mat = _raygun_from_esm(per_res_list, device=device, cache_dir=resolved_cache_dir)
    df_raygun = pd.DataFrame(
        ray_mat, columns=[f"e{i}" for i in range(ray_mat.shape[1])]
    )
    if verbose:
        print(f"[Info] Writing RayGun embeddings with shape {df_raygun.shape} to {ray_path}")
    df_raygun.to_csv(ray_path, index=False, header=True)

    if verbose:
        print("\n[Success] RayGun embedding generation complete!")
        print(f"[Info] Generated {len(seqs)} RayGun embeddings")
        print(f"[Info] Output files:")
        print(f"  - {seq_path}")
        print(f"  - {ray_path}")


def main():
    """Command-line interface for FASTA to embeddings conversion."""
    parser = argparse.ArgumentParser(
        description="Convert FASTA sequences to three embedding files (ESM2, RayGun, ProtT5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fasta_to_embeddings.py input.fasta output_dir/
  python fasta_to_embeddings.py data/seqs.fasta embeddings/ --cache-dir /path/to/cache
        """,
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "output_dir",
        help="Path to output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--cache-dir",
        default="/cluster/tufts/cowenlab/wlou01/modelcache/",
        help="Optional directory for caching model weights",
    )
    parser.add_argument(
        "--esm-batch-size",
        type=int,
        default=8,
        help="Batch size for ESM2 model (default: 8)",
    )
    parser.add_argument(
        "--prot-batch-size",
        type=int,
        default=4,
        help="Batch size for ProtT5 model (default: 4)",
    )
    parser.add_argument(
        "--models",
        choices=["all", "prott5", "raygun", "esm2"],
        default="all",
        help="Which embeddings to generate (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    
    args = parser.parse_args()
    
    try:
        fasta_to_embeddings(
            input_file=args.input_file,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            esm_batch_size=args.esm_batch_size,
            prot_batch_size=args.prot_batch_size,
            models=args.models,
            verbose=not args.quiet,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

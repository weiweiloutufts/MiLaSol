from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import esm
from transformers import T5EncoderModel, T5Tokenizer

# ----------------- Public API response container -----------------
# have options for both in-memory tensors and on-disk paths


@dataclass(frozen=True)
class EmbeddingArtifacts:
    sequences: List[str]
    esm_embeddings: torch.Tensor
    raygun_embeddings: torch.Tensor
    prott5_embeddings: torch.Tensor
    esm_per_residue: List[torch.Tensor]
    sequence_path: Optional[str]
    esm_path: Optional[str]
    raygun_path: Optional[str]
    prott5_path: Optional[str]


# ----------------- Sequence loading -----------------


import pandas as pd
import os
from pathlib import Path
from typing import List, Sequence, Union


def _load_sequences(
    source: Union[str, Sequence[str]], seq_col: str = "sequence"
) -> List[str]:
    if isinstance(source, (list, tuple)):
        seqs = [str(s) for s in source]

    elif isinstance(source, str):
        print("[Info] Loading sequences from source...")
        s = source.strip()
        p = Path(s)

        # Treat recognized file extensions as paths without pre-checking filesystem visibility.
        try:
            looks_like_path = len(s) < 240 and (
                p.suffix.lower() in {".csv", ".tsv", ".txt", ".fa", ".fasta", ".faa", ".fas"}
            )
        except OSError:
            looks_like_path = False

        if looks_like_path:
            ext = p.suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(p)
                if seq_col not in df.columns:
                    raise ValueError(
                        f"CSV missing column '{seq_col}'. Columns: {list(df.columns)}"
                    )
                seqs = df[seq_col].astype(str).tolist()

            elif ext == ".tsv":
                df = pd.read_csv(p, sep="	")
                if seq_col not in df.columns:
                    raise ValueError(
                        f"TSV missing column '{seq_col}'. Columns: {list(df.columns)}"
                    )
                seqs = df[seq_col].astype(str).tolist()

            elif ext in {".txt", ".fa", ".fasta", ".faa", ".fas"}:
                print(f"[Info] Reading sequences from file: {p}")
                raw_lines = p.read_text().splitlines()
                has_header = any(ln.strip().startswith(">") for ln in raw_lines)

                if ext == ".txt" and not has_header:
                    # Plain .txt with no FASTA headers: treat each non-empty line as one sequence
                    seqs = [ln.strip() for ln in raw_lines if ln.strip()]
                    if seqs:
                        print(f"[Info] Parsed {len(seqs)} line-delimited sequence(s).")
                else:
                    # FASTA-style parse (handles multi-line records and inline single-line FASTA)
                    seqs, cur = [], []
                    for line in raw_lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(">"):
                            if cur:
                                seqs.append("".join(cur))
                            # Support inline sequence on the header line (e.g., ">id SEQUENCE")
                            header_rest = line[1:].strip()
                            inline_seq = ""
                            if header_rest:
                                parts = header_rest.split(maxsplit=1)
                                if len(parts) == 2:
                                    inline_seq = parts[1].replace(" ", "")
                            cur = [inline_seq] if inline_seq else []
                        else:
                            cur.append(line)

                    if cur:
                        seqs.append("".join(cur))
                    if seqs:
                        print(
                            f"[Info] Parsed {len(seqs)} FASTA sequence(s); last length {len(seqs[-1])}"
                        )

                # .txt fallback no longer needed; covered above
            else:
                seqs = [p.read_text().strip()]

        else:
            # Definitely treat as a raw sequence string
            seqs = [s]

    else:
        raise TypeError(
            "source must be a CSV/TSV/FASTA/TXT path, list[str], or a single sequence string"
        )

    # Normalize
    seqs = [s.replace(" ", "").upper() for s in seqs if s and s.strip()]
    if not seqs:
        raise ValueError("No sequences found.")
    return seqs


# ----------------- ESM 2.0.0 -----------------


@torch.no_grad()
def _esm_perres_and_pooled(
    seqs: List[str], device: str, batch_size: int = 8
) -> Tuple[List[torch.Tensor], np.ndarray]:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    per_res_list: List[torch.Tensor] = []
    pooled_chunks: List[np.ndarray] = []

    allowed_residues = {tok for tok in alphabet.tok_to_idx if len(tok) == 1}
    data = [(f"seq_{i}", s) for i, s in enumerate(seqs)]
    invalid_examples = []
    sanitized_data = []
    invalid_count = 0
    for name, seq in data:
        invalid_chars = sorted({ch for ch in seq if ch not in allowed_residues})
        if invalid_chars:
            invalid_count += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(f"{name}={''.join(invalid_chars)}")
            seq = ''.join(ch if ch in allowed_residues else 'X' for ch in seq)
        sanitized_data.append((name, seq))
    data = sanitized_data
    if invalid_count:
        preview = ", ".join(invalid_examples)
        print(
            f"[ESM] WARNING: replaced invalid residues with X in {invalid_count} sequences. "
            f"Examples: {preview}"
        )

    # ---- length guard (ESM2 limit) ----
    MAX_LEN = 2000
    too_long = [(name, len(seq)) for name, seq in data if len(seq) > MAX_LEN]
    if too_long:
        preview = ", ".join([f"{n}={L}" for n, L in too_long[:10]])
        print(
            f"[ESM] WARNING: truncating {len(too_long)} sequences to max length {MAX_LEN}. "
            f"Examples: {preview}"
        )
        data = [
            (name, seq[:MAX_LEN] if len(seq) > MAX_LEN else seq)
            for name, seq in data
        ]
    # -------------------------------

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)

        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        reps = out["representations"][model.num_layers]  # (B, L, D)
        reps = reps[:, 1:-1]

        for b in range(reps.size(0)):
            per_tok = reps[b]
            per_res_list.append(per_tok.detach().cpu())
            pooled = per_tok.mean(dim=0)
            pooled_chunks.append(pooled.detach().cpu().float().numpy())

    pooled_np = np.stack(pooled_chunks, axis=0)
    return per_res_list, pooled_np


# ----------------- ProtT5 -----------------


def _prep_prott5(seqs: Iterable[str]) -> List[str]:
    valid = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    return [" ".join([ch if ch in valid else "X" for ch in s]) for s in seqs]


@torch.no_grad()
def _prott5_pooled(
    seqs: List[str], device: str, cache_dir: Optional[str], batch_size: int = 4
) -> np.ndarray:
    model_id = "Rostlab/prot_t5_xl_uniref50"
    tok = T5Tokenizer.from_pretrained(
        model_id, do_lower_case=False, cache_dir=cache_dir
    )
    mdl = (
        T5EncoderModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
    )

    texts = _prep_prott5(seqs)
    chunks: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).bool()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        chunks.append(pooled.cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


# ----------------- RayGun (ESM -> Ray encoder) -----------------


@torch.no_grad()
def _raygun_from_esm(
    per_res_list: List[torch.Tensor], device: str, cache_dir: str
) -> np.ndarray:
    localurl = str(Path(cache_dir) / "rohitsinghlab_raygun_main")
    raymodel, esmdecoder, _ = torch.hub.load(
        localurl, "pretrained_uniref50_4_4mil_800M", source="local"
    )
    raymodel = raymodel.model.to(device).eval()

    outs = []
    min_residues = 50
    short_examples = []
    short_count = 0
    for idx, per_tok in enumerate(per_res_list):
        x = per_tok.to(device)
        if x.ndim == 2:
            if x.size(0) < min_residues:
                short_count += 1
                if len(short_examples) < 10:
                    short_examples.append(f"seq_{idx}={x.size(0)}")
                # RayGun's reduction layer divides by zero when seq_len < 50.
                x = x.transpose(0, 1).unsqueeze(0)
                x = F.interpolate(
                    x, size=min_residues, mode="linear", align_corners=False
                )
                x = x.transpose(1, 2).contiguous()
            else:
                x = x.unsqueeze(0)

        out = raymodel(x, return_logits_and_seqs=False)
        z = out["fixed_length_embedding"].mean(dim=1)
        outs.append(z.squeeze(0).detach().cpu().numpy())

    if short_count:
        preview = ", ".join(short_examples)
        print(
            f"[RayGun] WARNING: upsampled {short_count} sequences shorter than "
            f"{min_residues} residues before RayGun encoding. Examples: {preview}"
        )

    return np.stack(outs, axis=0)


# ----------------- all three -----------------


def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
    """
    Priority:
      1) explicit cache_dir arg
      2) $HF_HOME (or $TRANSFORMERS_CACHE if HF_HOME unset)
      3) ~/.cache/protein_models
    Also sets env vars so esm/transformers/torch.hub all use it.
    """
    if cache_dir:
        cd = Path(cache_dir).expanduser()
    else:
        cd = Path(
            os.getenv("HF_HOME")
            or os.getenv("TRANSFORMERS_CACHE")
            or "~/.cache/protein_models"
        ).expanduser()

    cd.mkdir(parents=True, exist_ok=True)

    # Make sure all libraries respect it
    os.environ.setdefault("HF_HOME", str(cd))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cd))

    
    
    try:
        torch.hub.set_dir(cd)
    except Exception:
        pass
    return str(cd)


def transfer_to_three_embedding_csvs(
    source: Union[str, Sequence[str]],
    *,
    seq_col: str = "sequence",
    out_dir: Optional[str] = "embeddings_out_csv",
    cache_dir: Optional[str] = None,
    esm_batch_size: int = 8,
    prot_batch_size: int = 4,
    write_to_disk: bool = True,
) -> EmbeddingArtifacts:
    """Generate ESM, RayGun, and ProtT5 embeddings for ``source``.

    Args:
        source: Sequence source (path or collection of strings).
        seq_col: Column name to read from CSV/TSV inputs.
        out_dir: Directory for CSV outputs; ignored when ``write_to_disk`` is False.
        cache_dir: Model cache location; defaults to env variables or ``~/.cache`` fallback.
        esm_batch_size: Batch size for ESM forward pass.
        prot_batch_size: Batch size for ProtT5 forward pass.
        write_to_disk: When False, skips writing text/CSV artifacts and returns in-memory data only.

    Returns:
        ``EmbeddingArtifacts`` containing both in-memory tensors and optional on-disk paths.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = _load_sequences(source, seq_col=seq_col)
    # print(f"[Info] Generating embeddings for {len(seqs)} sequences on device: {device}")

    if write_to_disk and out_dir is None:
        raise ValueError("out_dir must be provided when write_to_disk=True")

    cache_dir = _resolve_cache_dir(cache_dir)

    out_path = Path(out_dir) if write_to_disk else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
    
    sequence_path = str(out_path / "seqs.txt") if out_path else None
    esm_path = str(out_path / "esm2.csv") if out_path else None
    ray_path = str(out_path / "raygun.csv") if out_path else None
    prot_path = str(out_path / "prott5.csv") if out_path else None

    # Ensure each item is a string sequence (join lists/tuples)
    seqs = [("".join(s) if isinstance(s, (list, tuple)) else str(s)) for s in seqs]

    if sequence_path:
        print(f"[Info] Writing {len(seqs)} sequences to {sequence_path}")
        with open(sequence_path, "w") as f:
            for s in seqs:
                f.write(s + "\n")

    per_res_list, esm_pooled = _esm_perres_and_pooled(
        seqs, device, batch_size=esm_batch_size
    )
    df_esm = pd.DataFrame(
        esm_pooled, columns=[f"e{i}" for i in range(esm_pooled.shape[1])]
    )
    if esm_path:
        print(f"[Info] Writing ESM embeddings with shape {df_esm.shape} to {esm_path}")
        df_esm.to_csv(esm_path, index=False, header=True)
    esm_emb = torch.tensor(df_esm.to_numpy(), dtype=torch.float32)

    ray_mat = _raygun_from_esm(per_res_list, device=device, cache_dir=cache_dir)
    df_raygun = pd.DataFrame(
        ray_mat, columns=[f"e{i}" for i in range(ray_mat.shape[1])]
    )
    if ray_path:
        print(
            f"[Info] Writing RayGun embeddings with shape {df_raygun.shape} to {ray_path}"
        )
        df_raygun.to_csv(ray_path, index=False, header=True)
    raygun_emb = torch.tensor(df_raygun.to_numpy(), dtype=torch.float32)

    prot_mat = _prott5_pooled(
        seqs, device, cache_dir=cache_dir, batch_size=prot_batch_size
    )
    df_prott5 = pd.DataFrame(
        prot_mat, columns=[f"e{i}" for i in range(prot_mat.shape[1])]
    )
    if prot_path:
        print(
            f"[Info] Writing ProtT5 embeddings with shape {df_prott5.shape} to {prot_path}"
        )
        df_prott5.to_csv(prot_path, index=False, header=True)
    prott5_emb = torch.tensor(df_prott5.to_numpy(), dtype=torch.float32)

    assert len(seqs) == len(df_esm) == len(df_raygun) == len(df_prott5)
    assert (
        esm_emb.size(0) == raygun_emb.size(0) == prott5_emb.size(0) == len(per_res_list)
    )

    return EmbeddingArtifacts(
        sequences=seqs,
        esm_embeddings=esm_emb,
        raygun_embeddings=raygun_emb,
        prott5_embeddings=prott5_emb,
        esm_per_residue=per_res_list,
        sequence_path=None,
        esm_path=None,
        raygun_path=None,
        prott5_path=None,
    )

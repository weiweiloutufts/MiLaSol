# utils_cache_and_embeddings.py
from __future__ import annotations
import os,torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import esm                      # fair-esm 2.0.0+
from transformers import T5Tokenizer, T5EncoderModel




# ----------------- Sequence loading -----------------
def _load_sequences(source: Union[str, List[str]], seq_col: str = "sequence") -> List[str]:
    if isinstance(source, (list, tuple)):
        seqs = [str(s) for s in source]

    elif isinstance(source, str):
        s = source.strip()
        p = Path(s)

        # Safely decide if it's a path
        try:
            looks_like_path = (
                len(s) < 240 and (  # avoid long AA strings
                    p.suffix.lower() in {".csv", ".tsv", ".txt", ".fa", ".fasta", ".faa", ".fas"} or
                    p.exists()
                )
            )
        except OSError:
            looks_like_path = False

        if looks_like_path and p.exists():
            ext = p.suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(p)
                if seq_col not in df.columns:
                    raise ValueError(f"CSV missing column '{seq_col}'. Columns: {list(df.columns)}")
                seqs = df[seq_col].astype(str).tolist()

            elif ext == ".tsv":
                df = pd.read_csv(p, sep="\t")
                if seq_col not in df.columns:
                    raise ValueError(f"TSV missing column '{seq_col}'. Columns: {list(df.columns)}")
                seqs = df[seq_col].astype(str).tolist()

            elif ext in {".txt", ".fa", ".fasta", ".faa", ".fas"}:
                # multi-line FASTA-safe reader
                seqs, cur = [], []
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if cur:
                            seqs.append("".join(cur))
                            cur = []
                    else:
                        cur.append(line)
                if cur:
                    seqs.append("".join(cur))
                # .txt fallback: if no FASTA records, treat non-empty lines as sequences
                if not seqs and ext == ".txt":
                    seqs = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

            else:
                seqs = [p.read_text().strip()]

        else:
            # Definitely treat as a raw sequence string
            seqs = [s]
    else:
        raise TypeError("source must be a CSV/TSV/FASTA/TXT path, list[str], or a single sequence string")

    # Normalize
    seqs = [s.replace(" ", "").upper() for s in seqs if s and s.strip()]
    if not seqs:
        raise ValueError("No sequences found.")
    return seqs


# ----------------- ESM 2.0.0 -----------------
@torch.no_grad()
def _esm_perres_and_pooled(seqs: List[str], device: str, batch_size: int = 8) -> tuple[list[torch.Tensor], np.ndarray]:
    
    #print(f"Info: Extracting ESM embeddings for {len(seqs)} sequences using batch size {batch_size} on device {device}...")
   
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    per_res_list: list[torch.Tensor] = []
    pooled_chunks: list[np.ndarray] = []

    data = [(f"seq_{i}", s) for i, s in enumerate(seqs)]
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)

        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        reps = out["representations"][model.num_layers]      # (B, L, D)
        reps = reps[:, 1:-1]                                 

        for b in range(reps.size(0)):
            per_tok = reps[b]                                # (L_b, D)
            per_res_list.append(per_tok.detach().cpu())
            pooled = per_tok.mean(dim=0)
            pooled_chunks.append(pooled.detach().cpu().float().numpy())
    #print(f"Info: Extracted ESM embeddings for {len(per_res_list)} sequences.")
    #print(f"Info: First Extracted ESM embeddings shape is  {per_res_list[0].shape} sequences.")
    
    pooled_np = np.stack(pooled_chunks, axis=0)
    return per_res_list, pooled_np


# ----------------- ProtT5 -----------------
def _prep_prott5(seqs: List[str]) -> List[str]:
    valid = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    return [" ".join([ch if ch in valid else "X" for ch in s]) for s in seqs]

@torch.no_grad()
def _prott5_pooled(seqs: List[str], device: str, cache_dir: str,batch_size: int = 4) -> np.ndarray:
    model_id = "Rostlab/prot_t5_xl_uniref50"
    tok = T5Tokenizer.from_pretrained(model_id, do_lower_case=False)
    mdl = T5EncoderModel.from_pretrained(model_id,cache_dir=cache_dir).to(device).eval()

    texts = _prep_prott5(seqs)
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).bool()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        chunks.append(pooled.cpu().float().numpy())
    return np.concatenate(chunks, axis=0)



# ----------------- RayGun (ESM -> Ray encoder) -----------------


@torch.no_grad()
def _raygun_from_esm(
    per_res_list: List[torch.Tensor],  
    device: str,
) -> np.ndarray:
    """
    Run RayGun encoder over per-residue ESM embeddings and mean-pool.
    Returns (N, D_raygun) numpy array.
    """

    # Build RayGun
  
    #raymodel = raygun_4_4mil_800M()  # returns the model
    localurl="/cluster/tufts/cowenlab/wlou01/modelcache/rohitsinghlab_raygun_main"
    raymodel, esmdecoder, _ = torch.hub.load(localurl, "pretrained_uniref50_4_4mil_800M", source = "local")
    raymodel = raymodel.model.to(device).eval()
   

    outs = []
    for per_tok in per_res_list:
        x = per_tok.to(device)                      # (L, D)
        if x.ndim == 2:
            x = x.unsqueeze(0)                      # -> (1, L, D) for 

        # Call the model; do NOT use model.encoder(...)
        out = raymodel(x, return_logits_and_seqs=False)
        # out["fixed_length_embedding"]: (1, 50, 1280)
        z = out["fixed_length_embedding"].mean(dim=1)  # (1, 1280), simple pool over 50 slots
        outs.append(z.squeeze(0).detach().cpu().numpy())

    return np.stack(outs, axis=0) # (N, D_r)

# ----------------- all three -----------------
def transfer_to_three_embedding_csvs(
    source: Union[str, List[str]],
    *,
    seq_col: str = "sequence",
    out_dir: str = "embeddings_out_csv",
    cache_dir: Optional[str] = None,
    esm_batch_size: int = 8,
    prot_batch_size: int = 4,
) -> Tuple[str, str, str, str,torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    Accepts CSV/TXT/FASTA path, list[str], or single sequence string.
    Resolves cache dir (arg -> HF_HOME/TRANSFORMERS_CACHE -> ~/.cache/protein_models),
    sets env vars so esm/HF/torch.hub use it, creates dir if needed.
    Writes:
      <base>.txt, <base>.esm2.csv, <base>.raygun.csv, <base>.prott5.csv
    Returns (source_txt, esm_csv, ray_csv, prot_csv).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = _load_sequences(source, seq_col=seq_col)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
   

    source_txt = str(out / f"seqs.txt")
    esm_csv    = str(out / f"esm2.csv")
    ray_csv    = str(out / f"raygun.csv")
    prot_csv   = str(out / f"prott5.csv")

    # Ensure each item is a string sequence (join lists/tuples)
    seqs = [
        ("".join(s) if isinstance(s, (list, tuple)) else str(s))
        for s in seqs
    ]
    # Save sequences (one per line)
    with open(source_txt, "w") as f:
        for s in seqs: f.write(s + "\n")

    # ESM2 (per-residue for RayGun + pooled for CSV)
    per_res_list, esm_pooled = _esm_perres_and_pooled(seqs,device,batch_size=esm_batch_size)
    df_esm =pd.DataFrame(esm_pooled, columns=[f"e{i}" for i in range(esm_pooled.shape[1])])
    df_esm.to_csv(esm_csv, index=False)
    esm_emb=torch.tensor(df_esm.to_numpy(), dtype=torch.float32)

    # RayGun (consumes per-residue ESM)
    ray_mat = _raygun_from_esm(
        per_res_list,
        device=device)
    df_raygun = pd.DataFrame(ray_mat, columns=[f"e{i}" for i in range(ray_mat.shape[1])])
    df_raygun.to_csv(ray_csv, index=False)
    raygun_emb=torch.tensor(df_raygun.to_numpy(), dtype=torch.float32)

    # ProtT5 (masked mean)
    prot_mat = _prott5_pooled(seqs, device, cache_dir,batch_size=prot_batch_size)
    df_prott5 = pd.DataFrame(prot_mat, columns=[f"e{i}" for i in range(prot_mat.shape[1])])
    df_prott5.to_csv(prot_csv, index=False)
    prott5_emb=torch.tensor(df_prott5.to_numpy(), dtype=torch.float32)
    
    assert len(seqs) == len(df_esm) == len(df_raygun) == len(df_prott5) == esm_emb.size(0) == raygun_emb.size(0) == prott5_emb.size(0) == len(per_res_list)

    return source_txt, esm_csv, ray_csv, prot_csv,per_res_list

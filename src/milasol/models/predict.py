import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from typing import Union
from milasol.data.datasets import ProteinDataset, collate_fn
from milasol.models.base import ProteinClassifier
from milasol.data.seq_to_embeddings import transfer_to_three_embedding_csvs


@torch.no_grad()
def get_pred(model, loader, device):
    model.eval()
    all_probs, all_preds, all_index = [], [], []

    for seqs, esm_embs, prot_embs, raygun_embs, feats, labels, idx in loader:
        # feats/labels may be sentinel tensors; be defensive
        seqs = seqs.to(device)
        esm_embs = esm_embs.to(device)
        prot_embs = prot_embs.to(device)
        raygun_embs = raygun_embs.to(device)

        logits, _, _, _ = model(seqs, esm_embs, prot_embs, raygun_embs)  # (B,) or (B,1)

        probs = torch.sigmoid(logits).view(-1)  # (B,)
        preds = (probs > 0.5).to(torch.int64).view(-1)

        if isinstance(idx, int):
            all_index.append(idx)
        else:
            all_index.extend(list(idx))
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return all_index, all_probs, all_preds


def _normalize_device(dev: Union[str, torch.device]) -> torch.device:
    if isinstance(dev, str):
        if dev.startswith("cuda") and not torch.cuda.is_available():
            # fall back gracefully if someone passes "cuda" but no CUDA present
            return torch.device("cpu")
        return torch.device(dev)
    return dev


def init_model(modelname: str, device: torch.device | str):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    #  instantiate the architecture (keep these hyperparams in sync with training)
    model = ProteinClassifier(
        vocab_size=22,
        embed_dim=128,
        num_filters=128,
        kernel_size=6,
        lstm_hidden_dim=128,
        num_lstm_layers=1,
        esm_dim=1280,
        prot_dim=1024,
        output_dim=1,
        latent_dim=64,
    ).to(device)

    #  load weights
    state_dict = torch.load(modelname, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # print(f"[INFO] Loaded model weights from {ckpt_path}")
    return model


def prediction(
    model: torch.nn.Module,
    source_data: str,
    device: Union[str, torch.device] = "cpu",
    out_dir: str = "outputs/",
    cache_dir: str | None = None,
):

    device = _normalize_device(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # build (or reuse) embeddings
    (
        source_data_txt,
        esm_csv,
        ray_csv,
        prott5_csv,
        per_res_list,
    ) = transfer_to_three_embedding_csvs(
        source_data,
        seq_col="sequence",  # only matters for CSV input
        out_dir=out_dir,  # will be created if missing
        cache_dir=cache_dir,  # optional; defaults inside the embedder if None
    )

    # dataset & loader
    data = ProteinDataset(
        source_data_txt,
        esm_csv,
        prott5_csv,
        ray_csv,
        feats_file=None,
        label_file=None,
        augment=False,
    )
    loader = DataLoader(data, batch_size=32, collate_fn=collate_fn)

    # predict
    all_index, all_probs, all_preds = get_pred(model, loader, device)
    # print("[INFO] Prediction completed.")

    # save alongside the source CSV

    #out_csv = str(Path(out_dir) / "predictions.csv")
    # pd.DataFrame({"idxs": all_index,"predicted_label": all_preds, "predicted_prob": all_probs}).to_csv(out_csv, index=False)

    # print(f"[INFO] Predictions saved to {out_csv}")
    return all_probs, all_preds, per_res_list


def main():
    ap = argparse.ArgumentParser(description="Run protein solubility prediction")
    ap.add_argument("--modelname", required=True, help="Checkpoint file")
    ap.add_argument(
        "--source_data",
        required=True,
        nargs="+",  # <== allows list of items
        help="Input: (1) CSV/TXT file, (2) a single sequence string, "
        "or (3) multiple sequences separated by space",
    )
    ap.add_argument("--out_dir", default="outputs/")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    # if they passed exactly one thing, treat it as str
    if len(args.source_data) == 1:
        source = args.source_data[0]
    else:
        # treat multiple inputs as list of sequences
        source = args.source_data

    prediction(
        args.modelname,
        source,  # can be str (file or single seq) OR list[str]
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()

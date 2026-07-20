import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


import argparse
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_optimizer as optim

from milasol.data.datasets_new import ProteinDataset, collate_fn

from milasol.models.base import ProteinClassifier


def _require_input_file(path_value, arg_name):
    if not path_value:
        raise ValueError(f"Missing required input argument: --{arg_name}")
    return str(path_value)


def weighted_supervised_contrastive_loss(
    features, labels, temperature=0.05, class_weights={0: 1.0, 1: 1.0}
):
    # Normalize embeddings to unit length
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0]

    # Compute cosine similarity matrix [B, B]
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # labels: [B] -> [B, 1] for broadcasting
    labels = labels.contiguous().view(-1, 1)

    # mask[i, j] = 1 if labels[i] == labels[j], else 0
    mask = torch.eq(labels, labels.T).float().to(features.device)

    # Create mask to remove self-comparisons (diagonal entries)
    logits_mask = torch.eye(batch_size, device=features.device).bool()
    mask = mask.masked_fill(logits_mask, 0)  # zero out diagonal

    # Apply per-class weights to the mask
    weight_matrix = torch.zeros_like(mask)
    for i in range(batch_size):
        class_weight = class_weights[labels[i].item()]
        weight_matrix[i] = mask[i] * class_weight

    # Compute log-softmax of similarities excluding self-pairs
    exp_logits = torch.exp(similarity_matrix) * (~logits_mask)
    log_prob = similarity_matrix - torch.log(
        exp_logits.sum(dim=1, keepdim=True) + 1e-12
    )

    # Compute weighted log-probabilities for positive pairs
    mean_log_prob_pos = (weight_matrix * log_prob).sum(1) / (
        weight_matrix.sum(1) + 1e-12
    )

    # Final loss is negative mean over samples
    loss = -mean_log_prob_pos.mean()

    return loss


def evaluate(model, loader, device, mode="Validation"):
    """Runs validation or testing phase to compute loss and classification metrics."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for seqs, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx in loader:
            seqs, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx = (
                seqs.to(device),
                esm_embs.to(device),
                prot_embs.to(device),
                raygun_embs.to(device),
                solu_feats.to(device),
                labels.to(device),
                idx.to(device),
            )

            logits, _, _, _ = model(
                seqs, esm_embs, prot_embs, raygun_embs, solu_feats
            )  # Only one pass
            loss = criterion(logits.view(-1), labels.view(-1).float())
            total_loss += loss.item()

            probs = torch.sigmoid(logits).squeeze()
            preds = torch.sigmoid(logits) > 0.5
            labels = labels.view(-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute classification metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="binary", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Probabilities needed for AUROC and AUPRC
    if all_probs is not None:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auroc = auprc = 0.0  # or handle None as needed

    # Confusion matrix to get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # Sensitivity (Insoluble) = TN / (TN + FP)
    sensitivity_insoluble = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision (Insoluble) = TN / (TN + FN)
    precision_insoluble = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Gain (Soluble) = Recall / (Proportion of soluble in dataset)
    total = tp + tn + fp + fn
    n_soluble = tp + fn
    n_insoluble = tn + fp
    gain_soluble = (recall / (n_soluble / total)) if n_soluble > 0 else 0.0
    gain_insoluble = (
        (sensitivity_insoluble / (n_insoluble / total)) if n_insoluble > 0 else 0.0
    )

    best_thresh, best_acc = find_best_threshold(all_probs, all_labels)
    print(
        f"🔍 Best threshold for accuracy is {best_thresh:.2f}, Accuracy is {best_acc:.4f}"
    )
    # Print all metrics
    print(
        f"{mode} - Loss: {avg_loss:.4f}, ACC: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, MCC: {mcc:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, "
        f"Sens(Insoluble): {sensitivity_insoluble:.4f}, Prec(Insoluble): {precision_insoluble:.4f}, "
        f"Gain(Sol): {gain_soluble:.2f}, Gain(Ins): {gain_insoluble:.2f}"
    )
    return accuracy, mcc


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max", min_epochs=10):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(
        self, score, model, modelname, current_epoch, train_acc=None, val_acc=None
    ):

        if self.best_score is None:
            self.best_score = score

            torch.save(model.state_dict(), modelname)
            print(
                f" ✅ Best Model saved at epoch {current_epoch} with score: {score:.4f}"
            )

        elif (self.mode == "min" and score > self.best_score - self.min_delta) or (
            self.mode == "max" and score < self.best_score + self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience and current_epoch >= self.min_epochs:
                self.early_stop = True

        else:
            self.best_score = score

            torch.save(model.state_dict(), modelname)
            print(
                f"✅ Best Model saved at epoch {current_epoch} with score: {score:.4f}"
            )

            self.counter = 0


class LRRestarter:
    def __init__(
        self, optimizer, initial_T_max, restart_T_max, patience, start_restart_epoch
    ):
        self.optimizer = optimizer
        self.initial_T_max = initial_T_max
        self.restart_T_max = restart_T_max
        self.patience = patience
        self.start_restart_epoch = start_restart_epoch

        self.scheduler = CosineAnnealingLR(
            optimizer, T_max=self.initial_T_max, eta_min=5e-5
        )
        self.best_score = None
        self.epochs_since_improvement = 0
        self.last_restart_epoch = 0

    def step(self):
        self.scheduler.step()

    def update(self, val_score, epoch):
        if self.best_score is None or val_score > self.best_score + 0.001:
            self.best_score = val_score
            self.epochs_since_improvement = 0
            return False  # no restart
        else:
            self.epochs_since_improvement += 1

            if (
                epoch >= self.start_restart_epoch
                and self.epochs_since_improvement > self.patience
            ):
                print(f"🔁 Restarting LR at epoch {epoch+1} due to no improvement")
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=self.restart_T_max, eta_min=5e-5
                )
                self.epochs_since_improvement = 0
                self.last_restart_epoch = epoch
                return True  # did restart

        return False  # no restart


def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.05, device="cpu"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.device = device

        # Register as buffer (non-trainable, not a Parameter)
        self.register_buffer("centers", torch.zeros(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        labels = labels.view(-1).long()
        batch_size = features.size(0)
        loss = 0.0

        # Compute loss only; update in post_step
        for j in range(self.num_classes):
            mask = labels == j
            if mask.sum() == 0:
                continue

            z_j = features[mask]
            center_j = self.centers[j].detach()  # Detach from graph
            loss += F.mse_loss(z_j, center_j.expand_as(z_j))

        return loss / batch_size

    def update_centers(self, features, labels):
        """Call this manually after loss.backward() and optimizer.step()"""
        labels = labels.view(-1).long()
        with torch.no_grad():
            for j in range(self.num_classes):
                mask = labels == j
                if mask.sum() == 0:
                    continue
                z_j = features[mask]
                mean_j = z_j.mean(dim=0)
                self.centers[j] = self.centers[j] - self.alpha * (
                    self.centers[j] - mean_j
                )


def batch_hard_triplet_loss_cosine(embeddings, labels, margin=0.2, device="cuda"):
    """
    Batch-hard triplet loss using cosine distance:
    - embeddings: Tensor [batch_size, feat_dim]
    - labels: Tensor [batch_size]
    """
    # Normalize embeddings to unit norm (important for cosine!)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Cosine similarity: [B, B]
    sim_matrix = torch.matmul(embeddings, embeddings.T)

    # Cosine distance: 1 - cosine similarity
    dist_matrix = 1.0 - sim_matrix

    labels = labels.view(-1, 1)
    label_mask = labels.eq(labels.T).to(device)  # [B, B]
    pos_mask = label_mask.clone()
    neg_mask = ~label_mask

    # Remove self-comparisons
    diag_mask = torch.eye(label_mask.size(0), dtype=torch.bool).to(device)
    pos_mask[diag_mask] = False

    # Hardest positive: maximum cosine distance (i.e., least similar)
    hardest_pos_dist, _ = (dist_matrix * pos_mask.float()).max(dim=1)

    # Hardest negative: minimum cosine distance (i.e., most similar among wrong labels)
    dist_matrix_neg = dist_matrix + 1e5 * (~neg_mask)
    hardest_neg_dist, _ = dist_matrix_neg.min(dim=1)

    # Triplet loss with cosine distance
    loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
    return loss.mean()


def find_best_threshold(probs, labels):

    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = -1

    for t in thresholds:
        preds = (probs > t).astype(int)

        score = accuracy_score(labels, preds)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    center_loss_fn,
    device,
    *,
    rec_loss_weight: float,
    con_weight: float,
    ent_weight: float,
    triplet_wt: float,
    proto_weight: float,
    center_weight: float = 0.1,
):
    model.train()
    total_loss = total_cls_loss = total_con_loss = total_recon_loss = total_entropy = (
        total_proto
    ) = total_center = 0.0
    all_preds, all_labels, all_probs = [], [], []

    proto_neg_mean = None  # init lazily once we see z

    for seqs, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx in train_loader:
        seqs, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx = (
            seqs.to(device),
            esm_embs.to(device),
            prot_embs.to(device),
            raygun_embs.to(device),
            solu_feats.to(device),
            labels.to(device),
            idx.to(device),
        )
        # print("Training, seqs,esm_embs, biophys_feats, labels",seqs.shape,esm_embs.shape, biophys_feats.shape, labels.shape)
        optimizer.zero_grad()

        logits, z, aug_z, rec_loss = model(seqs, esm_embs, prot_embs, raygun_embs)

        # lazy init using current D
        if proto_neg_mean is None:
            proto_neg_mean = torch.zeros(aug_z.size(1), device=device)

        smoothed_labels = smooth_labels(labels, smoothing=0.1)
        cls_loss = criterion(logits.view(-1), smoothed_labels.view(-1).float())

        # Normalize embeddings first
        z_normal = F.normalize(aug_z, dim=1)

        con_loss = weighted_supervised_contrastive_loss(
            z_normal, labels.view(-1).long(), class_weights={0: 1, 1: 2}
        )

        probs = torch.sigmoid(logits).squeeze()
        entropy = -(
            probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)
        ).mean()

        # Triplet Loss (cosine)
        triplet_loss = batch_hard_triplet_loss_cosine(
            z_normal, labels, margin=0.2, device=z.device
        )

        # Center Loss
        center_loss = center_loss_fn(z_normal, labels)

        # ✅ Get class-0 prototype

        # Negative class prototype pull
        neg_mask = (labels == 0).view(-1)
        if neg_mask.any():
            z_neg = z_normal[neg_mask]  # [N, D]
            proto_neg_mean = z_neg.mean(dim=0).detach() * 0.75 + proto_neg_mean * 0.25
            proto = proto_neg_mean.unsqueeze(0)  # [1, D]
            proto_loss_neg = F.cosine_embedding_loss(
                z_neg,
                proto.expand(z_neg.size(0), -1),
                target=torch.ones(z_neg.size(0), device=z.device),
            )
        else:
            proto_loss_neg = torch.tensor(0.0, device=z.device)

        # Combine with weights
        proto_loss = 1.0 * proto_loss_neg

        # ✅ Compute full loss (now includes prototype every batch)
        loss = (
            cls_loss
            + con_weight * con_loss
            + rec_loss_weight * rec_loss
            + ent_weight * entropy
            + triplet_wt * triplet_loss
            + proto_weight * proto_loss
            + center_weight * center_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # scheduler.step()

        # ✅ Update centers separately AFTER optimizer.step
        center_loss_fn.update_centers(z.detach(), labels.detach())

        total_loss += loss.item()
        total_cls_loss += cls_loss.detach().item()
        total_con_loss += con_loss.detach().item()
        total_recon_loss += rec_loss.detach().item()
        total_entropy += entropy.detach().item()
        total_proto += proto_loss.detach().item()
        total_center += center_loss.detach().item()

        probs = torch.sigmoid(logits).squeeze()
        preds = probs > 0.5
        labels = labels.view(-1)

        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # Averages
    total_len = len(train_loader)
    avg_train_loss = total_loss / total_len
    avg_cls_loss = total_cls_loss / total_len
    avg_con_loss = total_con_loss / total_len
    avg_recon_loss = total_recon_loss / total_len
    avg_entropy = total_entropy / total_len
    avg_proto = total_proto / total_len
    avg_center = total_center / total_len

    # Classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="binary", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auroc, auprc = float("nan"), float("nan")

    # Return logs + a compact dict for convenience
    logs = {
        "loss": avg_train_loss,
        "cls": avg_cls_loss,
        "con": avg_con_loss,
        "rec": avg_recon_loss,
        "ent": avg_entropy,
        "proto": avg_proto,
        "center": avg_center,
        "acc": accuracy,
        "prec": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
    }
    return logs


def train(
    batch_size,
    kernel_size,
    num_filters,
    lstm_hidden_dim,
    num_lstm_layers,
    learning_rate,
    latent_dim,
    contrastive_weight,
    rec_loss_weight,
    entropy_weight,
    triplet_weight,
    proto_weight,
    pos_rate=2.0,
    model_dir=None,
    train_sequence_file=None,
    train_esm_file=None,
    train_prot_file=None,
    train_raygun_file=None,
    train_label_file=None,
    val_sequence_file=None,
    val_esm_file=None,
    val_prot_file=None,
    val_raygun_file=None,
    val_label_file=None,
    test_sequence_file=None,
    test_esm_file=None,
    test_prot_file=None,
    test_raygun_file=None,
    test_label_file=None,
):

    # ----- Model/data hyperparams -----
    vocab_size = 22
    embed_dim = 128
    output_dim = 1
    esm_dim = 1280
    prot_dim = 1024

    entropy_weight = 0.1

    print("Model parameters")
    print("batch_size: ", batch_size)
    print("kernel_size: ", kernel_size)
    print("num_filters: ", num_filters)
    print("lstm_hidden_dim: ", lstm_hidden_dim)
    print("num_lstm_layers: ", num_lstm_layers)
    print("learning_rate: ", learning_rate)
    print("latent_dim: ", latent_dim)
    print("contrastive_weight: ", contrastive_weight)
    print("rec_loss_weight: ", rec_loss_weight)
    print("entropy_weight: ", entropy_weight)
    print("triplet_weight: ", triplet_weight)
    print("proto_weight: ", proto_weight)
    print("pos_rate: ", pos_rate)
    print("lam range: 0.75-0.95")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_dir = Path(model_dir or ".")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_sequence_file = _require_input_file(train_sequence_file, "train_sequence_file")
    train_esm_file = _require_input_file(train_esm_file, "train_esm_file")
    train_raygun_file = _require_input_file(train_raygun_file, "train_raygun_file")
    train_prot_file = _require_input_file(train_prot_file, "train_prot_file")
    train_label_file = _require_input_file(train_label_file, "train_label_file")

    val_sequence_file = _require_input_file(val_sequence_file, "val_sequence_file")
    val_esm_file = _require_input_file(val_esm_file, "val_esm_file")
    val_raygun_file = _require_input_file(val_raygun_file, "val_raygun_file")
    val_prot_file = _require_input_file(val_prot_file, "val_prot_file")
    val_label_file = _require_input_file(val_label_file, "val_label_file")

    test_sequence_file = _require_input_file(test_sequence_file, "test_sequence_file")
    test_esm_file = _require_input_file(test_esm_file, "test_esm_file")
    test_raygun_file = _require_input_file(test_raygun_file, "test_raygun_file")
    test_prot_file = _require_input_file(test_prot_file, "test_prot_file")
    test_label_file = _require_input_file(test_label_file, "test_label_file")

    train_data = ProteinDataset(
        train_sequence_file,
        train_esm_file,
        train_prot_file,
        train_raygun_file,
        feats_file=None,
        label_file=train_label_file,
    )
    val_data = ProteinDataset(
        val_sequence_file,
        val_esm_file,
        val_prot_file,
        val_raygun_file,
        feats_file=None,
        label_file=val_label_file,
    )
    test_data = ProteinDataset(
        test_sequence_file,
        test_esm_file,
        test_prot_file,
        test_raygun_file,
        feats_file=None,
        label_file=test_label_file,
    )

    print("Train sequence file:", train_sequence_file)
    print("Validation sequence file:", val_sequence_file)
    print("Test sequence file:", test_sequence_file)

    # Create DataLoader

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    # Instantiate model
    model = ProteinClassifier(
        vocab_size,
        embed_dim,
        num_filters,
        kernel_size,
        lstm_hidden_dim,
        num_lstm_layers,
        esm_dim,
        prot_dim,
        output_dim,
        latent_dim,
    )

    model = model.to(device)

    # Training Loop
    num_epochs = 40
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Lookahead(base_opt, k=5, alpha=0.5)

    lr_manager = LRRestarter(
        optimizer,
        initial_T_max=8,  # full cycle initially
        restart_T_max=8,  # shorter cycles after restarts
        patience=3,  # wait 3 epochs of no improvement
        start_restart_epoch=8,  # don't restart before epoch 8
    )

    early_stopper = EarlyStopping(patience=8, min_delta=0, mode="max", min_epochs=15)

    pos_weight = torch.tensor([pos_rate])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    center_loss_fn = CenterLoss(
        num_classes=2, feat_dim=latent_dim, alpha=0.2, device=device
    )

    print("Training...")

    con_weight = contrastive_weight
    ent_weight = entropy_weight
    triplet_wt = triplet_weight

    for epoch in range(num_epochs):

        # ---- Train one epoch ----
        logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            center_loss_fn,
            device,
            rec_loss_weight=rec_loss_weight,
            con_weight=con_weight,
            ent_weight=ent_weight,
            triplet_wt=triplet_wt,
            proto_weight=proto_weight,
            center_weight=0.1,
        )

        print(
            f"Cls: {logs['loss']:.4f} | Con: {logs['con']:.4f} | Rec: {logs['rec']:.4f} | Ent: {logs['ent']:.4f} | Proto: {logs['proto']:.4f} | Center: {logs['center']:.4f}"
        )
        print(
            f" Train - Average Loss: {logs['loss']:.4f}, Accuracy: {logs['acc']:.4f}, Precision: {logs['prec']:.4f}, Recall: {logs['recall']:.4f}, F1-score: {logs['f1']:.4f}, MCC: {logs['mcc']:.4f}, AUROC: {logs['auroc']:.4f}, AUPRC: {logs['auprc']:.4f}"
        )

        # ---- Validation ----
        val_acc, val_mcc = evaluate(model, val_loader, device, "Validation")

        # LR scheduling / restarts
        lr_manager.step()
        lr_manager.update(val_mcc, epoch)

        # ---- Save & Early stopping ----
        current_epoch = epoch + 1
        modelname = str(model_dir / f"train{contrastive_weight}_{entropy_weight}_{rec_loss_weight}_{triplet_weight}_{latent_dim}_{pos_rate}_{proto_weight}.pth")

        early_stopper.step(
            val_mcc, model, modelname, current_epoch, logs["acc"], val_acc
        )

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {current_epoch}")

            break

        print(f"Epoch {current_epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"Training Complete! The model was saved at {modelname}")

    # Testing Loop
    model.load_state_dict(torch.load(modelname, weights_only=True))
    model = model.to(device)
    evaluate(model, test_loader, device, "Test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--rec_loss_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--triplet_weight", type=float, default=0.2)
    parser.add_argument("--proto_weight", type=float, default=0.5)
    parser.add_argument(
        "--pos_rate", type=float, default=2.0, help="Positive weight for BCE loss"
    )
    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument("--train_sequence_file", type=str, default=None)
    parser.add_argument("--train_esm_file", type=str, default=None)
    parser.add_argument("--train_prot_file", type=str, default=None)
    parser.add_argument("--train_raygun_file", type=str, default=None)
    parser.add_argument("--train_label_file", type=str, default=None)
    parser.add_argument("--val_sequence_file", type=str, default=None)
    parser.add_argument("--val_esm_file", type=str, default=None)
    parser.add_argument("--val_prot_file", type=str, default=None)
    parser.add_argument("--val_raygun_file", type=str, default=None)
    parser.add_argument("--val_label_file", type=str, default=None)
    parser.add_argument("--test_sequence_file", type=str, default=None)
    parser.add_argument("--test_esm_file", type=str, default=None)
    parser.add_argument("--test_prot_file", type=str, default=None)
    parser.add_argument("--test_raygun_file", type=str, default=None)
    parser.add_argument("--test_label_file", type=str, default=None)

    args = parser.parse_args()

    train(
        args.batch_size,
        args.kernel_size,
        args.num_filters,
        args.lstm_hidden_dim,
        args.num_lstm_layers,
        args.learning_rate,
        args.latent_dim,
        args.contrastive_weight,
        args.rec_loss_weight,
        args.entropy_weight,
        args.triplet_weight,
        args.proto_weight,
        args.pos_rate,
        args.model_dir,
        args.train_sequence_file,
        args.train_esm_file,
        args.train_prot_file,
        args.train_raygun_file,
        args.train_label_file,
        args.val_sequence_file,
        args.val_esm_file,
        args.val_prot_file,
        args.val_raygun_file,
        args.val_label_file,
        args.test_sequence_file,
        args.test_esm_file,
        args.test_prot_file,
        args.test_raygun_file,
        args.test_label_file,
    )


if __name__ == "__main__":
    main()

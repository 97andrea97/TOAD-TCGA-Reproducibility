#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_TRAIN_EVAL = REPO_ROOT / "src_train_eval"
sys.path.insert(0, str(SRC_TRAIN_EVAL))

from models.model_toad import TOAD_fc_mtl_concat  # noqa: E402
from utils.utils import print_network  # noqa: E402


def initiate_model(n_classes: int, drop_out: bool, ckpt_path: str, device: torch.device):
    model = TOAD_fc_mtl_concat(dropout=drop_out, n_classes=n_classes)
    if hasattr(model, "relocate"):
        model.relocate()
    model = model.to(device)

    print_network(model)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to slide features .pt (torch Tensor: [N,D] or [1,N,D])")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--sex", required=True, type=float, help="Sex covariate (F=0, M=1)")
    ap.add_argument("--n-classes", type=int, default=18)
    ap.add_argument("--drop-out", action="store_true", help="Enable dropout in TOAD_fc_mtl_concat")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = torch.load(args.pt, map_location="cpu")
    if not isinstance(feats, torch.Tensor):
        raise RuntimeError(f"{args.pt} did not contain a torch.Tensor")

    feats = feats.float()
    if feats.dim() == 3 and feats.size(0) == 1:
        feats = feats.squeeze(0)
    if feats.dim() != 2:
        raise RuntimeError(f"Unexpected features shape: {tuple(feats.shape)} (expected [N,D])")

    sex = torch.tensor([[args.sex]], dtype=torch.float32)

    model = initiate_model(
        n_classes=args.n_classes,
        drop_out=args.drop_out,
        ckpt_path=args.ckpt,
        device=device,
    )

    feats = feats.to(device, non_blocking=True)
    sex = sex.to(device, non_blocking=True)

    with torch.no_grad():
        out = model(feats, sex)

    if not (isinstance(out, dict) and "Y_prob" in out and "logits" in out and "Y_hat" in out):
        raise RuntimeError("Model output dict does not contain expected keys: logits, Y_prob, Y_hat")

    y_prob = out["Y_prob"][0].detach().cpu()
    logits = out["logits"][0].detach().cpu()
    y_hat = int(out["Y_hat"][0].item())

    probs = y_prob.tolist()
    probs_sorted = [
        {"index": int(i), "prob": float(p)}
        for i, p in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    ]

    payload = {
        "pt": str(Path(args.pt).name),
        "ckpt": str(Path(args.ckpt).name),
        "sex": float(args.sex),
        "pred_index": y_hat,
        "probs": probs,
        "probs_sorted": probs_sorted,
        "logits": logits.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

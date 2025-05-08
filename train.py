"""Multi-task training script for domain & sentiment classification."""

from __future__ import annotations

import argparse
import os
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.data.dataset import load_task_csv
from app.models import BertEmbedder, ClassificationHead


def train(args: argparse.Namespace) -> None:  # noqa: C901 – keep it simple for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Shared encoder
    embedder = BertEmbedder(device=device)
    tokenizer = embedder.tokenizer
    encoder = embedder.model  # shorthand
    hidden_size = encoder.config.hidden_size

    # 2) Task-specific data & heads ------------------------------------------------
    label2id_dom = {"history": 0, "geography": 1, "health": 2, "technology": 3}
    label2id_sent = {"negative": 0, "neutral": 1, "positive": 2}

    dataset_dom = load_task_csv(
        os.path.join(args.data_dir, "domain.csv"),
        text_col="sentence",
        label_col="domain",
        tokenizer=tokenizer,
        max_length=args.max_length,
        label2id=label2id_dom,
    )
    dataset_sent = load_task_csv(
        os.path.join(args.data_dir, "sentiment.csv"),
        text_col="sentence",
        label_col="sentiment",
        tokenizer=tokenizer,
        max_length=args.max_length,
        label2id=label2id_sent,
    )

    loader_dom = DataLoader(dataset_dom, batch_size=args.batch_size, shuffle=True)
    loader_sent = DataLoader(dataset_sent, batch_size=args.batch_size, shuffle=True)

    head_dom = ClassificationHead(hidden_size, len(label2id_dom)).to(device)
    head_sent = ClassificationHead(hidden_size, len(label2id_sent)).to(device)

    # 3) Optimizer & loss ----------------------------------------------------------
    params = list(encoder.parameters()) + list(head_dom.parameters()) + list(head_sent.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    w_dom, w_sent = args.weight_domain, args.weight_sentiment

    # 4) Training loop -------------------------------------------------------------
    encoder.train()
    head_dom.train()
    head_sent.train()

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0.0
        steps = max(len(loader_dom), len(loader_sent))
        dom_iter, sent_iter = cycle(loader_dom), cycle(loader_sent)

        for _ in range(steps):
            ids_dom, m_dom, lbl_dom = next(dom_iter)
            ids_sent, m_sent, lbl_sent = next(sent_iter)

            ids_dom, m_dom, lbl_dom = ids_dom.to(device), m_dom.to(device), lbl_dom.to(device)
            ids_sent, m_sent, lbl_sent = ids_sent.to(device), m_sent.to(device), lbl_sent.to(device)

            optimizer.zero_grad()

            # Task A: domain -----------------------------------------------------
            out_dom = encoder(input_ids=ids_dom, attention_mask=m_dom).last_hidden_state[:, 0, :]
            logits_dom = head_dom(out_dom)
            loss_dom = loss_fn(logits_dom, lbl_dom)

            # Task B: sentiment ---------------------------------------------------
            out_sent = encoder(input_ids=ids_sent, attention_mask=m_sent).last_hidden_state[:, 0, :]
            logits_sent = head_sent(out_sent)
            loss_sent = loss_fn(logits_sent, lbl_sent)

            loss = w_dom * loss_dom + w_sent * loss_sent
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{args.num_epochs} — avg loss: {total_loss/steps:.4f}")

    # 5) Save task heads (encoder stays on HF Hub) ---------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "mtl_heads.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),  # fine-tuned BERT
            "domain_head": head_dom.state_dict(),
            "sentiment_head": head_sent.state_dict(),
            "label2id_dom": label2id_dom,
            "label2id_sent": label2id_sent,
            "model_name": encoder.config._name_or_path,
        },
        ckpt_path,
    )
    print(f"Saved task heads → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task training on domain & sentiment.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with CSV files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save model heads")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--weight_domain", type=float, default=1.0)
    parser.add_argument("--weight_sentiment", type=float, default=1.0)

    train(parser.parse_args()) 
"""Batched replacement for eval_save.py's embedding loop.

eval_save.py embeds one function per forward pass: for a run the size of
libpng+freetype that is ~290k sequential BERT-base forwards, each launching a full
kernel stack for a single 512-token sequence. The GPU spends most of its time idle
between launches. Nothing about the computation requires this -- the functions are
independent, and their embeddings are scattered back into the same ragged
ebds[i][opt_key] structure regardless of the order they were computed in.

This script computes exactly the same thing in batches. It is a SEPARATE file rather
than an edit to eval_save.py so that existing workflows -- and any run currently in
flight -- keep using the known-good path until this one has been verified against it.

Two padding modes, deliberately split by how much trust they need:

  --padding max_length  (default)
      Every sequence is padded to 512, exactly as eval_save.py does. The tensors fed
      to the model are identical to the single-item case, so the only source of
      divergence is the reduction order inside batched matmuls. Differences should be
      at float32 epsilon.

  --padding longest
      Pad only to the longest sequence in the batch. Most functions are far shorter
      than 512 tokens and attention is O(L^2), so this is where the large win is.
      It is mathematically equivalent -- the attention mask zeroes the pad columns, so
      pad positions cannot contribute to any real position's output -- but that is an
      argument, not a measurement. Verify with compare_embeddings.py before trusting
      it for a real run.

Combine --padding longest with --length_sort to put similarly-sized functions in the
same batch, which is what actually removes the wasted padding.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from portability_utils import get_device
from data import FunctionDataset_CL


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


def flatten_tasks(dataset):
    """Flatten the ragged ebds structure into a list of (i, opt_key, text).

    ebds[i] is a dict holding 'proj' and 'funcname' metadata alongside one entry per
    compilation variant, whose value is an index into datas[i]. Only the variant
    entries are embeddings; the two metadata keys must survive untouched.
    """
    tasks = []
    for i in range(len(dataset.datas)):
        texts = dataset.datas[i]
        for opt_key in dataset.ebds[i]:
            if opt_key in ("proj", "funcname"):
                continue
            idx = dataset.ebds[i][opt_key]
            tasks.append((i, opt_key, texts[idx]))
    return tasks


def embed_all(model, tokenizer, tasks, device, batch_size, padding, length_sort, max_length=512):
    """Embed every task and return {(i, opt_key): tensor of shape [1, 768]}."""
    order = list(range(len(tasks)))
    if length_sort:
        # Sorting by raw character count is a good enough proxy for token count and
        # costs nothing -- it only decides batch composition, never the result.
        order.sort(key=lambda t: len(tasks[t][2]))

    out = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(order), batch_size), desc=f"embed(bs={batch_size})"):
            chunk = order[start:start + batch_size]
            batch_texts = [tasks[t][2] for t in chunk]

            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding=padding,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            pooled = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            pooled = pooled.detach().cpu()

            for row, t in enumerate(chunk):
                i, opt_key, _ = tasks[t]
                # .clone() is load-bearing: a bare slice is a view onto the whole batch
                # tensor, which would keep every batch alive for the length of the run.
                out[(i, opt_key)] = pooled[row:row + 1].clone()
    return out


def main():
    parser = argparse.ArgumentParser(description="jTrans-EvalSave (batched)")
    parser.add_argument("--model_path", type=str, default="./models/jTrans-finetune")
    parser.add_argument("--dataset_path", type=str, default="./BinaryCorp/small_test")
    parser.add_argument("--experiment_path", type=str, default="./experiments/jTrans.pkl")
    parser.add_argument("--tokenizer", type=str, default="./jtrans_tokenizer/")
    parser.add_argument(
        "--min_variants", type=int, default=1,
        help="Embed every function present in at least this many compilation variants. "
             "Matches eval_save.py's flag; pass 0 for the old intersect-everything "
             "behaviour.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Functions per forward pass. 1 reproduces eval_save.py's launch pattern.",
    )
    parser.add_argument(
        "--padding", type=str, default="max_length", choices=["max_length", "longest"],
        help="'max_length' pads to 512 like eval_save.py and is numerically the safe "
             "choice. 'longest' pads per batch and is much faster, but verify it "
             "against a max_length run first.",
    )
    parser.add_argument(
        "--length_sort", action="store_true",
        help="Group similar-length functions into the same batch. Only useful with "
             "--padding longest, where it is what actually removes wasted padding.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Embed only the first N functions. For quick A/B checks against a "
             "reference pkl -- not for real runs.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Loading model from {args.model_path} ...")
    model = BinBertModel.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    print("Preparing dataset ...")
    dataset = FunctionDataset_CL(
        tokenizer, args.dataset_path, None, True,
        add_ebd=True, convert_jump_addr=True,
        min_variants=(None if args.min_variants == 0 else args.min_variants),
    )

    tasks = flatten_tasks(dataset)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} function variants to embed across {len(dataset.datas)} functions.")

    started = time.time()
    embeddings = embed_all(
        model, tokenizer, tasks, device,
        batch_size=args.batch_size, padding=args.padding, length_sort=args.length_sort,
    )
    elapsed = time.time() - started
    rate = len(tasks) / elapsed if elapsed else float("nan")
    print(f"Embedded {len(tasks)} variants in {elapsed:.1f}s ({rate:.1f}/s).")

    # Scatter back into the ragged structure eval_save.py produces, so the pkl is
    # drop-in compatible with do_step_5 and build_jtrans_embeddings.
    for (i, opt_key), vec in embeddings.items():
        dataset.ebds[i][opt_key] = vec

    if args.limit:
        # A partial run would otherwise write a pkl whose untouched entries are still
        # integer indices into datas -- silently corrupt if it reached the pipeline.
        embedded = {i for i, _ in embeddings}
        dataset.ebds = [dataset.ebds[i] for i in sorted(embedded)]
        print(f"--limit set: writing only the {len(dataset.ebds)} fully/partially "
              f"embedded functions. This pkl is for comparison only.")

    os.makedirs(os.path.dirname(os.path.abspath(args.experiment_path)), exist_ok=True)
    with open(args.experiment_path, "wb") as f:
        pickle.dump(dataset.ebds, f)
    print(f"Wrote {args.experiment_path}")


if __name__ == "__main__":
    main()

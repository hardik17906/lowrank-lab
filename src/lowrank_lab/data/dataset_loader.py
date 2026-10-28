"""Small public task with auditable, disjoint splits and immutable raw hash."""

import hashlib
import io
import random
import re
import requests
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def deduplicate(records):
    """Collapse normalized duplicates; remove ALL texts with conflicting labels."""
    groups = {}
    for row in records:
        key = normalized_text(row["text"])
        groups.setdefault(key, []).append(row)
    clean = [
        rows[0] for key, rows in sorted(groups.items()) if len({r["label"] for r in rows}) == 1
    ]
    return clean, len(records) - len(clean)


def load_sms(cache_dir="data/raw", split_seed=42, allow_download=False):
    cache = Path(cache_dir)
    archive = cache / "sms_spam_collection.zip"
    if not archive.exists():
        if not allow_download:
            raise FileNotFoundError(
                "Dataset absent; run lowrank-lab prepare --config ... --allow-download"
            )
        cache.mkdir(parents=True, exist_ok=True)
        with requests.get(SMS_URL, timeout=60, stream=True) as response:
            response.raise_for_status()
            chunks, received = [], 0
            for chunk in response.iter_content(65536):
                received += len(chunk)
                if received > 10_000_000:
                    raise RuntimeError("Unexpected dataset archive size")
                chunks.append(chunk)
            content = b"".join(chunks)
        if len(content) > 10_000_000:
            raise RuntimeError("Unexpected dataset archive size")
        archive.write_bytes(content)
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.rsplit("/", 1)[-1] == "SMSSpamCollection")
        text = z.read(name).decode("utf-8")
    records = []
    for line in text.splitlines():
        if line:
            label, message = line.split("\t", 1)
            records.append({"text": message, "label": int(label == "spam")})
    clean, removed = deduplicate(records)
    train, held = train_test_split(
        clean, test_size=0.30, random_state=split_seed, stratify=[r["label"] for r in clean]
    )
    validation, test = train_test_split(
        held, test_size=0.50, random_state=split_seed, stratify=[r["label"] for r in held]
    )
    return {"train": train, "validation": validation, "test": test}, {
        "dataset": "UCI SMS Spam Collection",
        "source": SMS_URL,
        "license": "CC BY 4.0 (UCI dataset page)",
        "doi": "10.24432/C5CC84",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_rows": len(records),
        "removed_duplicate_or_conflicting_rows": removed,
        "split_seed": split_seed,
        "split_policy": "stratified 70/15/15 after normalization/deduplication",
        "full_counts": {
            k: len(v) for k, v in [("train", train), ("validation", validation), ("test", test)]
        },
    }


def take_subset(rows, count, fraction=1.0, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0,1]")
    # Fixed split seed, never training seed, ensures seed studies share examples.
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    cap = min(count, len(ordered)) if count else len(ordered)
    return ordered[: max(1, int(cap * fraction))]

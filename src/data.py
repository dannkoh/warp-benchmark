#!/usr/bin/env python3
import argparse
import re
import random
from pathlib import Path
from itertools import combinations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the tokenizer (tokenization runs on CPU and is lightweight)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", use_fast=True)

def count_tokens(text: str) -> int:
    """
    Count tokens using the AutoTokenizer.
    """
    return len(tokenizer(text)["input_ids"])

def discover(folder: Path, prefix: str = "custom") -> dict[str, dict[int, dict[str, any]]]:
    """
    Discover SMT2 files in the given folder and its subfolders.
    Files must follow the naming convention: {prefix}.{problem}_{number}.smt2.
    For each file, extract constants and assertions and cache the token count for the assertions.
    """
    discovered = {}
    pattern_re = re.compile(rf"^{re.escape(prefix)}\.([^.]+)_([0-9]+)\.smt2$")
    for smt_file in folder.rglob(f"{prefix}.*_*.smt2"):
        filename = smt_file.name
        match = pattern_re.match(filename)
        if not match:
            continue
        problem, n_str = match.groups()
        try:
            n = int(n_str)
        except ValueError:
            continue
        with smt_file.open() as file:
            smt_lines = [line.strip() for line in file if line.strip()]
            constants = [line for line in smt_lines if line.startswith("(declare-const")]
            assertions = [line for line in smt_lines if line.startswith("(assert")]
        assertions_text = "\n".join(assertions) if assertions else "None"
        token_count = count_tokens(assertions_text)
        if problem not in discovered:
            discovered[problem] = {}
        discovered[problem][n] = {
            "constants": constants,
            "assertions": assertions,
            "assertions_text": assertions_text,
            "token_count": token_count,
        }
    print(f"Discovered {len(discovered)} problems with {sum(len(v) for v in discovered.values())} examples")
    return discovered

def adapt_prompt_text(combo: tuple, examples: dict[int, dict[str, any]], target_n: int, sep: str = "\n", max_tokens: int = 4096) -> str:
    """
    Build the prompt text using candidate examples from a combination.
    Uses cached token counts for each example's assertions.
    If the computed token count exceeds max_tokens, attempt to reduce the candidate set
    by pruning (keeping first, median, and last). If still too high, return None.
    No truncation of constraints is performed.
    """
    def build_prompt(candidate_combo):
        formatted_examples = []
        for idx in candidate_combo:
            if idx in examples:
                data = examples[idx]
                example_text = data.get("assertions_text", "None")
            else:
                example_text = "None"
            formatted_examples.append(f"N={idx}: {example_text}")
        examples_text = sep.join(formatted_examples)
        prompt_text = (
            f"Given the following examples of constraints for increasing input sizes:\n"
            f"{examples_text}\n"
            f"What is the constraint for N={target_n}?"
        )
        return prompt_text

    def compute_prompt_token_count(candidate_combo):
        total = 0
        header = "Given the following examples of constraints for increasing input sizes:"
        total += count_tokens(header)
        total += count_tokens(sep)  # for newline
        for idx in candidate_combo:
            candidate_prefix = f"N={idx}:"
            total += count_tokens(candidate_prefix)
            if idx in examples:
                total += examples[idx].get("token_count", 0)
            else:
                total += count_tokens("None")
            total += count_tokens(sep)
        trailing = f"What is the constraint for N={target_n}?"
        total += count_tokens(trailing)
        return total

    # Try the full candidate set first
    if compute_prompt_token_count(combo) <= max_tokens:
        return build_prompt(combo)
    # If too high, try a reduced candidate set (first, median, last)
    if len(combo) >= 3:
        reduced_combo = (combo[0], combo[len(combo)//2], combo[-1])
    else:
        reduced_combo = combo
    if compute_prompt_token_count(reduced_combo) <= max_tokens:
        return build_prompt(reduced_combo)
    # Otherwise, skip this combination.
    return None

def assign_tier(difficulty: int) -> str:
    """
    Assign a tier based on the difficulty jump (target_n minus the maximum candidate index).
    """
    if difficulty <= 5:
        return "small"
    elif difficulty <= 15:
        return "medium"
    else:
        return "large"

def row_generator(discovered: dict[str, dict[int, dict[str, any]]],
                  sep: str = "\n",
                  max_target_n: int = 30,
                  max_combos: int = None):
    """
    Generator that yields one dataset row at a time.
    For each problem and each target value (N ≤ max_target_n), generate all combinations (size ≥ 3)
    of candidate examples (with N values less than the target).
    If max_combos is set, only that many candidate combinations per target are processed.
    Adaptive prompt construction and tier assignment (based on difficulty) are applied.
    Yields dictionaries with keys 'question', 'answer', 'constants', and 'tier'.
    """
    for problem, examples in tqdm(discovered.items(), desc="Processing problems"):
        valid_targets = sorted(n for n in examples.keys() if n <= max_target_n)
        for target_n in valid_targets:
            candidate_indices = sorted([n for n in examples.keys() if n < target_n])
            if len(candidate_indices) < 3:
                continue
            combos = []
            for r in range(3, len(candidate_indices) + 1):
                combos.extend(list(combinations(candidate_indices, r)))
            if max_combos is not None and len(combos) > max_combos:
                combos = random.sample(combos, max_combos)
            for combo in combos:
                difficulty = target_n - max(combo)
                tier = assign_tier(difficulty)
                question = adapt_prompt_text(combo, examples, target_n, sep)
                if question is None:
                    continue
                target_data = examples[target_n]
                answer = "\n".join(target_data.get("assertions", [])) or "None"
                constants = "\n".join(target_data.get("constants", [])) or "None"
                yield {
                    "question": question,
                    "answer": answer,
                    "constants": constants,
                    "tier": tier,
                }

def streaming_balanced_sampling(row_iter, max_total=1000):
    """
    Stream rows through reservoir sampling to achieve balanced sampling by tier without storing all rows.
    Maintains a separate reservoir for each tier (small, medium, large) with a maximum of max_total//3 rows each.
    """
    reservoirs = {"small": [], "medium": [], "large": []}
    counts = {"small": 0, "medium": 0, "large": 0}
    target_per_tier = max_total // 3
    for row in row_iter:
        tier = row.get("tier", "medium")
        counts[tier] += 1
        if len(reservoirs[tier]) < target_per_tier:
            reservoirs[tier].append(row)
        else:
            if random.random() < target_per_tier / counts[tier]:
                replace_index = random.randrange(target_per_tier)
                reservoirs[tier][replace_index] = row
    final_rows = reservoirs["small"] + reservoirs["medium"] + reservoirs["large"]
    return final_rows

def main():
    parser = argparse.ArgumentParser(
        description="Build a benchmark dataset by processing SMT2 files and streaming the rows to avoid excessive memory usage."
    )
    parser.add_argument("folder", type=str, help="Parent folder to search for SMT2 files")
    parser.add_argument("--prefix", type=str, default="custom", help="Prefix used in SMT2 filenames")
    parser.add_argument("--separator", type=str, default="\n", help="Separator for joining lines")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the dataset file")
    parser.add_argument("--output_filename", type=str, default="dataset.parquet", help="Name of the output dataset file")
    parser.add_argument("--max_total", type=int, default=1000, help="Maximum number of rows in the final balanced dataset")
    parser.add_argument("--max_target_n", type=int, default=30, help="Upper bound for target N (should be ≤ 30)")
    parser.add_argument("--max_combos", type=int, default=1000, help="Limit for candidate combinations per target")
    
    args = parser.parse_args()

    folder = Path(args.folder)
    discovered = discover(folder, prefix=args.prefix)
    
    row_iter = row_generator(discovered, sep=args.separator, max_target_n=args.max_target_n, max_combos=args.max_combos)
    final_rows = streaming_balanced_sampling(row_iter, max_total=args.max_total)
    print(f"Final dataset contains {len(final_rows)} rows after streaming balanced sampling.")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / args.output_filename

    df = pd.DataFrame(final_rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(output_file))
    print(f"Saved dataset with {len(final_rows)} rows to {output_file}")

if __name__ == "__main__":
    main()

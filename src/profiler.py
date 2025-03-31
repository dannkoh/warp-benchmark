import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from tqdm import tqdm
import re



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", use_fast=True)


def count_tokens(text: str) -> int:
    """Count tokens using the AutoTokenizer."""
    return len(tokenizer(text)["input_ids"])

def convert_numpy_types(obj):
    """Convert numpy types in a nested structure to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from a Parquet file."""
    print(f"Loading dataset from {file_path}...")
    return pd.read_parquet(file_path)


def analyze_token_lengths(df: pd.DataFrame) -> dict:
    """Analyze token lengths of questions and answers."""
    print("Analyzing token lengths...")

    
    if "question_tokens" not in df.columns:
        df["question_tokens"] = df["question"].progress_apply(count_tokens)
    if "answer_tokens" not in df.columns:
        df["answer_tokens"] = df["answer"].progress_apply(count_tokens)

    stats = {
        "questions": {
            "min": df["question_tokens"].min(),
            "max": df["question_tokens"].max(),
            "mean": df["question_tokens"].mean(),
            "median": df["question_tokens"].median(),
            "std": df["question_tokens"].std(),
            "total": df["question_tokens"].sum(),
            "percentiles": {
                "25": df["question_tokens"].quantile(0.25),
                "50": df["question_tokens"].quantile(0.50),
                "75": df["question_tokens"].quantile(0.75),
                "90": df["question_tokens"].quantile(0.90),
                "95": df["question_tokens"].quantile(0.95),
                "99": df["question_tokens"].quantile(0.99),
            },
        },
        "answers": {
            "min": df["answer_tokens"].min(),
            "max": df["answer_tokens"].max(),
            "mean": df["answer_tokens"].mean(),
            "median": df["answer_tokens"].median(),
            "std": df["answer_tokens"].std(),
            "total": df["answer_tokens"].sum(),
            "percentiles": {
                "25": df["answer_tokens"].quantile(0.25),
                "50": df["answer_tokens"].quantile(0.50),
                "75": df["answer_tokens"].quantile(0.75),
                "90": df["answer_tokens"].quantile(0.90),
                "95": df["answer_tokens"].quantile(0.95),
                "99": df["answer_tokens"].quantile(0.99),
            },
        },
    }

    return stats


def analyze_tiers(df: pd.DataFrame) -> dict:
    """Analyze distribution of difficulty tiers."""
    print("Analyzing difficulty tiers...")

    tier_counts = df["tier"].value_counts().to_dict()
    total = df.shape[0]

    tier_stats = {
        "counts": tier_counts,
        "percentages": {
            tier: (count / total) * 100 for tier, count in tier_counts.items()
        },
        "total": total,
    }

    
    tier_stats["avg_question_tokens"] = (
        df.groupby("tier")["question_tokens"].mean().to_dict()
    )
    tier_stats["avg_answer_tokens"] = (
        df.groupby("tier")["answer_tokens"].mean().to_dict()
    )

    return tier_stats


def analyze_answers(df: pd.DataFrame) -> dict:
    """Analyze characteristics of answers."""
    print("Analyzing answers...")

    
    df["answer_length"] = df["answer"].str.len()

    
    none_answers = (df["answer"] == "None").sum()

    
    common_prefixes = Counter(df["answer"].str[:20]).most_common(5)

    answer_stats = {
        "length_chars": {
            "min": df["answer_length"].min(),
            "max": df["answer_length"].max(),
            "mean": df["answer_length"].mean(),
            "median": df["answer_length"].median(),
        },
        "none_answers": {
            "count": none_answers,
            "percentage": (none_answers / df.shape[0]) * 100,
        },
        "common_prefixes": common_prefixes,
    }

    return answer_stats


def extract_problem_stats(df: pd.DataFrame) -> dict:
    """Extract problem-related statistics from questions."""
    print("Extracting problem statistics...")

    
    df["target_n"] = (
        df["question"].str.extract(r"What is the constraint for N=(\d+)\?").astype(int)
    )

    
    unique_targets = df["target_n"].unique()

    
    example_counts = []
    for question in df["question"]:
        examples = len(re.findall(r"N=\d+:", question))
        example_counts.append(examples)

    df["example_count"] = example_counts

    problem_stats = {
        "target_n_distribution": {
            "min": df["target_n"].min(),
            "max": df["target_n"].max(),
            "mean": df["target_n"].mean(),
            "median": df["target_n"].median(),
            "unique_values": sorted(unique_targets.tolist()),
        },
        "example_counts": {
            "min": df["example_count"].min(),
            "max": df["example_count"].max(),
            "mean": df["example_count"].mean(),
            "median": df["example_count"].median(),
            "distribution": Counter(example_counts),
        },
    }

    return problem_stats


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate and save visualizations."""
    print("Generating visualizations...")

    output_dir.mkdir(exist_ok=True)

    
    sns.set(style="whitegrid")

    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["question_tokens"], kde=True)
    plt.title("Question Token Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(df["answer_tokens"], kde=True)
    plt.title("Answer Token Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "token_distributions.png")

    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x="tier", data=df, order=["small", "medium", "large"])
    plt.title("Distribution of Difficulty Tiers")
    plt.xlabel("Tier")
    plt.ylabel("Count")
    
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "tier_distribution.png")

    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(
        x="tier", y="question_tokens", data=df, order=["small", "medium", "large"]
    )
    plt.title("Question Token Length by Tier")

    plt.subplot(1, 2, 2)
    sns.boxplot(
        x="tier", y="answer_tokens", data=df, order=["small", "medium", "large"]
    )
    plt.title("Answer Token Length by Tier")
    plt.tight_layout()
    plt.savefig(output_dir / "token_by_tier.png")

    
    plt.figure(figsize=(10, 6))
    sns.histplot(df["target_n"], kde=True, bins=20)
    plt.title("Target N Distribution")
    plt.xlabel("Target N Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "target_n_distribution.png")

    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df["example_count"],
        kde=False,
        bins=range(min(df["example_count"]), max(df["example_count"]) + 2),
    )
    plt.title("Number of Examples per Question")
    plt.xlabel("Example Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "example_count_distribution.png")


def print_statistics(stats: dict):
    """Print formatted statistics."""
    print("\n===== DATASET STATISTICS =====")

    print(f"\nDataset Size: {stats['dataset_size']} rows")

    print("\n----- TOKEN STATISTICS -----")
    print("\nQuestion Tokens:")
    print(f"  Min: {stats['token_stats']['questions']['min']:.1f}")
    print(f"  Max: {stats['token_stats']['questions']['max']:.1f}")
    print(f"  Mean: {stats['token_stats']['questions']['mean']:.1f}")
    print(f"  Median: {stats['token_stats']['questions']['median']:.1f}")
    print(f"  Standard Deviation: {stats['token_stats']['questions']['std']:.1f}")

    print("\nAnswer Tokens:")
    print(f"  Min: {stats['token_stats']['answers']['min']:.1f}")
    print(f"  Max: {stats['token_stats']['answers']['max']:.1f}")
    print(f"  Mean: {stats['token_stats']['answers']['mean']:.1f}")
    print(f"  Median: {stats['token_stats']['answers']['median']:.1f}")
    print(f"  Standard Deviation: {stats['token_stats']['answers']['std']:.1f}")

    print("\n----- TIER STATISTICS -----")
    for tier, count in stats["tier_stats"]["counts"].items():
        percentage = stats["tier_stats"]["percentages"][tier]
        print(f"  {tier.capitalize()}: {count} ({percentage:.1f}%)")
        print(
            f"    - Average question tokens: {stats['tier_stats']['avg_question_tokens'][tier]:.1f}"
        )
        print(
            f"    - Average answer tokens: {stats['tier_stats']['avg_answer_tokens'][tier]:.1f}"
        )

    print("\n----- ANSWER STATISTICS -----")
    print(
        f"Character Length - Min: {stats['answer_stats']['length_chars']['min']}, "
        f"Max: {stats['answer_stats']['length_chars']['max']}, "
        f"Mean: {stats['answer_stats']['length_chars']['mean']:.1f}"
    )

    print(
        f"\nNone Answers: {stats['answer_stats']['none_answers']['count']} "
        f"({stats['answer_stats']['none_answers']['percentage']:.1f}%)"
    )

    print("\n----- PROBLEM STATISTICS -----")
    print("\nTarget N Values:")
    print(f"  Min: {stats['problem_stats']['target_n_distribution']['min']}")
    print(f"  Max: {stats['problem_stats']['target_n_distribution']['max']}")
    print(f"  Mean: {stats['problem_stats']['target_n_distribution']['mean']:.1f}")
    print(f"  Median: {stats['problem_stats']['target_n_distribution']['median']:.1f}")

    print("\nExamples per Question:")
    print(f"  Min: {stats['problem_stats']['example_counts']['min']}")
    print(f"  Max: {stats['problem_stats']['example_counts']['max']}")
    print(f"  Mean: {stats['problem_stats']['example_counts']['mean']:.1f}")
    print(f"  Median: {stats['problem_stats']['example_counts']['median']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Profile the benchmark dataset")
    parser.add_argument(
        "dataset_path", type=str, help="Path to the dataset Parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save profiling results and visualizations",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save visualizations"
    )

    args = parser.parse_args()

    tqdm.pandas()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = load_dataset(args.dataset_path)

    token_stats = analyze_token_lengths(df)
    tier_stats = analyze_tiers(df)
    answer_stats = analyze_answers(df)
    problem_stats = extract_problem_stats(df)

    all_stats = {
        "dataset_size": df.shape[0],
        "token_stats": token_stats,
        "tier_stats": tier_stats,
        "answer_stats": answer_stats,
        "problem_stats": problem_stats,
    }
    all_stats = convert_numpy_types(all_stats)


    with open(output_dir / "statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print_statistics(all_stats)

    if args.visualize:
        generate_visualizations(df, output_dir)
        print(f"Visualizations saved to {output_dir}")

    print(f"\nDetailed statistics saved to {output_dir / 'statistics.json'}")


if __name__ == "__main__":
    main()
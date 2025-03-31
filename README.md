# invaR1ant-benchmark

A benchmark for testing logical reasoning and pattern generalization in large language models through formal SMT constraints.

## Overview

InvaR1ant is a benchmark designed to evaluate a language model's ability to generalize invariant logical reasoning across different input sizes. Each example asks a model to generate a formal constraint for a specific target size after being shown examples of constraints for smaller input sizes.

## Dataset Statistics

### Basic Statistics
- **Total samples**: 671
- **Format**: Parquet file
- **Difficulty tiers**:
  - Small: 333 samples (49.6%)
  - Medium: 333 samples (49.6%)
  - Large: 5 samples (0.7%)

### Token Statistics (using Qwen2.5-3B tokenizer)
|                | Questions | Answers |
|----------------|-----------|---------|
| Minimum        | 117       | 27      |
| Maximum        | 4,086     | 14,476  |
| Mean           | 1,684     | 994     |
| Median         | 1,521     | 493     |
| Total tokens   | 1,130,153 | 666,709 |

### Target N Distribution
- Range: 5 to 30
- Mean: 19.9
- Median: 20
- Unique values: 25 different values

### Examples Per Question
- Range: 3 to 18 examples
- Mean: 6.7
- Median: 6
- Most common: 3 examples (168 questions)

## Dataset Structure

Each example in the dataset follows this structure:

- **Question**: A prompt showing examples of constraints for smaller input sizes N, asking for constraints at target N
- **Answer**: The formal constraint for the target size
- **Constants**: Variable declarations for the target size
- **Tier**: Difficulty level ("small", "medium", or "large")

Example question format:
```
Given the following examples of constraints for increasing input sizes:
N=3: (assert (and (and ...)))
N=4: (assert (and (and ...)))
...
What is the constraint for N={target}?
```

## Dataset Creation

The dataset is generated from SMT2 files using the following process:

1. **Discovery**: Finds SMT2 files following the naming pattern `{prefix}.{problem}_{number}.smt2`
2. **Combination Generation**: For each problem and target N value, generates combinations of smaller N values to use as examples
3. **Prompt Construction**: Creates prompts with selected examples and targets, ensuring they fit within token limits
4. **Difficulty Assignment**: Assigns difficulty tiers based on the "jump" between example N and target N:
   - Small: ≤ 5
   - Medium: 6-15
   - Large: > 15
5. **Balanced Sampling**: Uses reservoir sampling to balance the dataset across difficulty tiers

## Dataset Profiling

Use the included profiler to analyze the dataset:

```bash
python profiler.py dataset.parquet --visualize
```

This will generate comprehensive statistics and visualizations about token lengths, difficulty distribution, and other metrics.

## Dependencies

- pandas
- pyarrow
- transformers (Qwen2.5-3B tokenizer)
- tqdm
- matplotlib (for visualization)
- seaborn (for visualization)

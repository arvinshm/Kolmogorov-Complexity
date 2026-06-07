# New Math RL Brevity Experiment

This folder is a clean replacement for the old sequence-DSL experiment. It is
self-contained: after you inspect it, the rest of `Compression_and_math` can be
deleted without breaking this experiment.

The hypothesis is:

> If a model is reinforced to solve math correctly with shorter generated
> reasoning, does its held-out math benchmark performance improve in a
> statistically significant way?

The design tests that hypothesis with two matched RL conditions:

1. **Correctness-only RL:** reward only whether the final answer is correct.
2. **Correctness + brevity RL:** reward correctness, and among correct answers
   add a tunable bonus for shorter generated reasoning.

Both runs start from the same base model and train on the same RL environment.
Both are evaluated on the same benchmark examples. The final comparison is
paired, so each benchmark problem acts as its own control.

## Defaults

- Base model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- RL environment: `openai/gsm8k`, config `main`, split `train`
- Held-out benchmark: `HuggingFaceH4/MATH-500`, split `test`
- RL algorithm: TRL `GRPOTrainer`
- Parameter-efficient training: LoRA or QLoRA
- Answer verification: Hugging Face `math-verify` when installed, with a
  conservative exact/numeric fallback

Why these defaults:

- Qwen2.5-Math-1.5B-Instruct is open, math-tuned, and small enough for practical
  LoRA/QLoRA experiments.
- GSM8K provides many verifiable grade-school math questions with explicit final
  answers, making it a natural RL environment for answer-only reward.
- MATH-500 is a stronger held-out benchmark than GSM8K and should not be used as
  the RL training environment if the goal is a meaningful before/after test.

## Files

```text
new/
  README.md
  requirements.txt
  train_rl_math.py       # GRPO training for both reward modes
  evaluate_math.py       # benchmark evaluation for base/adapters
  compare_runs.py        # paired significance tests
  run_experiment.py      # one-command orchestration
  data_utils.py          # dataset loading and prompt formatting
  math_utils.py          # answer extraction, verification, length scoring
  model_utils.py         # model/tokenizer/LoRA loading helpers
  __init__.py
```

## Installation

Create a fresh environment on your cloud GPU machine:

```bash
cd /path/to/Compression_and_math/new
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `bitsandbytes` has platform trouble, you can still run full precision LoRA by
omitting `--load-in-4bit`. For most single-GPU 1.5B experiments, QLoRA is the
more convenient path.

## Quick Local Checks

These do not run the expensive experiment:

```bash
python train_rl_math.py --reward-test --reward-mode correctness
python train_rl_math.py --reward-test --reward-mode correctness_brevity --brevity-weight 0.5
python run_experiment.py --dry-print --max-train-samples 8 --benchmark-limit 8 --max-steps 1
```

The reward test prints correct/wrong reward values. The dry-print command shows
the full command sequence without launching training.

## One-Command Experiment

This runs:

1. base model benchmark evaluation,
2. correctness-only RL training,
3. correctness + brevity RL training from the same base model,
4. benchmark evaluation for both adapters,
5. paired statistical comparison.

```bash
python run_experiment.py \
  --work-dir results/gsm8k_to_math500_qwen15b \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --max-steps 1000 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --brevity-weight 0.25 \
  --brevity-length-cap 256 \
  --train-extra-args="--load-in-4bit --torch-dtype bfloat16" \
  --eval-extra-args="--load-in-4bit --torch-dtype bfloat16 --batch-size 4"
```

Important output paths:

```text
results/gsm8k_to_math500_qwen15b/
  adapters/
    correctness/
    brevity_lambda_0.25/
  eval/
    base/
      predictions.jsonl
      summary.json
    correctness/
      predictions.jsonl
      summary.json
    brevity/
      predictions.jsonl
      summary.json
  comparison_brevity_vs_correctness.json
```

## Manual Step-By-Step Experiment

### 1. Evaluate the Base Model

```bash
python evaluate_math.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --output-dir results/manual/eval \
  --run-name base \
  --load-in-4bit \
  --torch-dtype bfloat16
```

### 2. Train Correctness-Only RL

```bash
python train_rl_math.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --output-dir results/manual/adapters/correctness \
  --reward-mode correctness \
  --max-steps 1000 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --load-in-4bit \
  --torch-dtype bfloat16
```

Reward:

```text
wrong answer:   0
correct answer: 1
```

### 3. Evaluate Correctness-Only Adapter

```bash
python evaluate_math.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-dir results/manual/adapters/correctness \
  --output-dir results/manual/eval \
  --run-name correctness \
  --load-in-4bit \
  --torch-dtype bfloat16
```

### 4. Train Brevity-Reward RL From Scratch

This starts from the same base model, not from the correctness-only adapter.

```bash
python train_rl_math.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --output-dir results/manual/adapters/brevity_lambda_0.25 \
  --reward-mode correctness_brevity \
  --brevity-weight 0.25 \
  --brevity-length-cap 256 \
  --max-steps 1000 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --load-in-4bit \
  --torch-dtype bfloat16
```

Reward:

```text
wrong answer:   0
correct answer: 1 + brevity_weight * brevity_score
```

### 5. Evaluate Brevity Adapter

```bash
python evaluate_math.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-dir results/manual/adapters/brevity_lambda_0.25 \
  --output-dir results/manual/eval \
  --run-name brevity \
  --load-in-4bit \
  --torch-dtype bfloat16
```

### 6. Compare the Two RL Conditions

```bash
python compare_runs.py \
  --a-predictions results/manual/eval/correctness/predictions.jsonl \
  --b-predictions results/manual/eval/brevity/predictions.jsonl \
  --a-name correctness \
  --b-name brevity_lambda_0.25 \
  --output-json results/manual/comparison_brevity_vs_correctness.json
```

The comparison file reports:

- paired accuracy difference: brevity minus correctness-only,
- exact McNemar p-value,
- bootstrap 95% CI for the paired accuracy difference,
- mean reasoning-token lengths for both runs.

## Flowchart

```text
                    +-------------------------------+
                    | Base open-source math model   |
                    | Qwen2.5-Math-1.5B-Instruct    |
                    +---------------+---------------+
                                    |
                                    v
                     +--------------+--------------+
                     | RL environment: GSM8K train |
                     | prompt -> sampled solution  |
                     | final answer is verified    |
                     +--------------+--------------+
                                    |
               +--------------------+--------------------+
               |                                         |
               v                                         v
 +-------------+---------------+          +--------------+--------------+
 | train_rl_math.py            |          | train_rl_math.py            |
 | --reward-mode correctness   |          | --reward-mode               |
 |                             |          | correctness_brevity         |
 | reward = correct only       |          | reward = correct + short    |
 +-------------+---------------+          +--------------+--------------+
               |                                         |
               v                                         v
 +-------------+---------------+          +--------------+--------------+
 | correctness LoRA adapter    |          | brevity LoRA adapter        |
 +-------------+---------------+          +--------------+--------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                      +-------------+-------------+
                      | evaluate_math.py          |
                      | benchmark: MATH-500      |
                      | same prompts, same judge  |
                      +-------------+-------------+
                                    |
                                    v
                      +-------------+-------------+
                      | compare_runs.py           |
                      | paired McNemar + bootstrap|
                      +---------------------------+
```

## Mathematical Definitions

Let:

- `x` be a math problem prompt.
- `g` be the gold final answer.
- `y` be a sampled model completion.
- `extract(y)` be the final answer extracted from `y`, preferably from the last
  `\boxed{...}` expression.
- `verify(a, g)` be the answer verifier.
- `L(y)` be the measured generated reasoning length in tokens.
- `L_cap` be `--brevity-length-cap`.
- `lambda` be `--brevity-weight`.

Correctness indicator:

```text
C(y, g) = 1[ verify(extract(y), g) ]
```

Brevity score:

```text
B(y) = max(0, 1 - L(y) / L_cap)
```

Correctness-only reward:

```text
R_correct(y, g) = C(y, g)
```

Correctness + brevity reward:

```text
R_short(y, g) = C(y, g) * (1 + lambda * B(y))
```

By construction, all incorrect answers receive zero reward in both conditions.
The brevity bonus only ranks completions that already have correct final
answers.

### Length Measurement

By default, `L(y)` is `reasoning_tokens`: the number of tokenizer tokens before
the final answer marker, such as `\boxed`, `final answer`, or `answer:`. This is
closer to "chain-of-thought length" than total completion length. You can switch
to total generated length with:

```bash
--brevity-measure completion_tokens
```

### GRPO Objective

For each prompt `x`, GRPO samples a group of `G` completions:

```text
y_1, ..., y_G ~ pi_theta_old(. | x)
```

Each completion receives reward `R_i`. The group-normalized advantage is:

```text
A_i = (R_i - mean_j R_j) / (std_j R_j + epsilon)
```

Ignoring token masks for readability, the policy ratio for token `t` is:

```text
r_{i,t}(theta) =
  pi_theta(y_{i,t} | x, y_{i,<t})
  /
  pi_theta_old(y_{i,t} | x, y_{i,<t})
```

The clipped policy objective is PPO-like:

```text
J_policy(theta) =
  mean_{i,t} min(
    r_{i,t}(theta) * A_i,
    clip(r_{i,t}(theta), 1 - eps, 1 + eps) * A_i
  )
```

TRL's GRPO trainer also keeps the policy near a reference model with a KL term.
Conceptually:

```text
Loss(theta) = -J_policy(theta)
              + beta * KL(pi_theta(. | x) || pi_ref(. | x))
```

The script exposes `--beta` for the reference-model regularization strength.

## Benchmark Metrics

`evaluate_math.py` writes one JSONL row per completion and a summary with:

- completion accuracy,
- pass@k if multiple samples per problem are generated,
- Wilson 95% confidence intervals,
- mean completion tokens,
- mean reasoning tokens,
- mean reasoning tokens among correct completions,
- subject and level breakdowns when the dataset provides metadata.

For the main hypothesis test, prefer greedy or deterministic `k=1` evaluation
for both adapters. That makes the paired comparison clean and easy to interpret.

## Statistical Test

`compare_runs.py` compares two prediction files on shared problem ids.

Let:

- `b` = problems solved by brevity RL and missed by correctness-only RL.
- `c` = problems solved by correctness-only RL and missed by brevity RL.

The exact McNemar test conditions on `b + c` discordant examples:

```text
X ~ Binomial(b + c, 0.5)
p_two_sided = 2 * min( P[X <= min(b,c)], P[X >= max(b,c)] )
```

It also computes a paired bootstrap confidence interval for:

```text
Delta = accuracy_brevity - accuracy_correctness
```

Interpretation:

- If `Delta > 0`, the brevity-reward adapter solved more benchmark problems.
- If the bootstrap CI excludes 0 and McNemar p-value is small, the improvement is
  more credible.
- If reasoning-token length drops but accuracy does not improve, the brevity
  reward compressed explanations without making the model smarter.

## Useful Knobs

Training:

```text
--max-steps
--num-generations
--learning-rate
--beta
--load-in-4bit
--lora-r
--lora-alpha
--max-train-samples
```

Brevity:

```text
--brevity-weight       # lambda in the reward formula
--brevity-length-cap   # L_cap
--brevity-measure      # reasoning_tokens or completion_tokens
```

Evaluation:

```text
--limit
--batch-size
--max-new-tokens
--temperature
--num-samples-per-problem
```

## Custom Datasets

Both training and evaluation support local JSONL. Each row should contain:

```json
{"problem": "...", "answer": "..."}
```

or:

```json
{"question": "...", "gold_answer": "..."}
```

For GSM8K-style rows, an `answer` containing `#### final` is also supported.

Training on a local JSONL:

```bash
python train_rl_math.py \
  --train-jsonl data/my_math_train.jsonl \
  --reward-mode correctness_brevity \
  --brevity-weight 0.25
```

Evaluating on a local JSONL:

```bash
python evaluate_math.py \
  --dataset-jsonl data/my_math_benchmark.jsonl \
  --adapter-dir results/manual/adapters/brevity_lambda_0.25
```

## Notes And Caveats

- This experiment rewards visible generated reasoning length. It does not claim
  to measure hidden reasoning.
- The reward is intentionally answer-only except for the optional brevity bonus.
  There is no process reward and no reward model.
- `math-verify` improves LaTeX equivalence checking, especially on MATH-500.
  Keep it installed for the actual experiment.
- MATH-500 may overlap with some public pretraining corpora. The paired design
  still tests whether the two RL treatments differ, but absolute scores should
  be interpreted carefully.
- If both RL runs collapse into very short wrong answers, reduce
  `--brevity-weight`, increase `--beta`, or increase `--brevity-length-cap`.


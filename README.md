# Does rewarding brevity improve mathematical reasoning?

A core feature of intelligence is compression: the ability to explain something
with fewest bits of information. This experiment is designed to test whether 

This folder is a clean replacement for the old sequence-DSL experiment. It is
self-contained: after you inspect it, the rest of `Compression_and_math` can be
deleted without breaking this experiment.

The hypothesis is:

> If a model is reinforced to solve math correctly with shorter generated
> reasoning, does its held-out math benchmark performance improve in a
> statistically significant way?

The design tests that hypothesis with two separate RL conditions:

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

## Step-By-Step Experiment

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



# Results






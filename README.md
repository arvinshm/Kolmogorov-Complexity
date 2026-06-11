# Does rewarding brevity improve mathematical reasoning?

A core feature of intelligence is compression: the ability to explain something
with fewest bits of information. This experiment is designed to test a specific 
hypothesis along these lines. The hypothesis is:

> If a model is reinforced to solve math correctly with a shorter chain
> of though, does its (held-out) math benchmark performance improve in a
> statistically significant way?

The design tests that hypothesis with two separate RL conditions:

1. **Correctness-only RL:** reward only whether the final answer is correct.
2. **Correctness + brevity RL:** reward correctness, and among correct answers
   add a tunable bonus for shorter generated reasoning.

More concretely, we will start with a Qwen model which is already good (~70%) on the 
becnhmark (Math-500) if given enough reasoning tokens (max_tokens ~ 1500). 
The RL environment (openai/gsm8k) is simple enough that GRPO does not really improve this
benchmark that much. It pushes 70% to XXX. But now suppose, we change the evaluation of 
the benchmark by limiting the number of reasoning tokens to XXX. As expected, performance of
the base model on this constrained benchmark drops (to XXX). Now, we ask the question: can we teach the model brevity as well as correctness
and see if this token-constrained bechmark improves? The answer is yes. We define two
GRPO environments with different rewards. One where rewards is 1-0 if boxed answer is 
correct-wrong. And the other, where correct answers get more rewarded if the are brief (see below for details).
We find that the correctness-only RL barely improves the constrained benchmark (XXX->XXX).
But, the brevity RL training meaningfully improves this performance (XXX->XXX).

My interpretation of this result is that there is a correlation between brevity and reasnoning 
ability. This matches the expectation that reanoning, quite generally, is correlated with finding
short explanations of things, and perhaps can assist with model generalization.





## Defaults

- Base model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- RL environment: `openai/gsm8k`, config `main`, split `train`
- Held-out benchmark: `HuggingFaceH4/MATH-500`, split `test`
- RL algorithm: TRL `GRPOTrainer`
- Training format: LoRA


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



# More details on the Results






# Does rewarding brevity improve mathematical reasoning?

A core feature of intelligence is compression: the ability to explain something
with fewer bits of information. This experiment tests a narrow version of that
idea:

> If a model is reinforced to solve math correctly with a shorter chain of
> thought, does its (held-out) math benchmark performance improve in a
> statistically significant way?

The answer from this experiment is: yes, but only in the constrained-token
setting we actually tested. The base model,
`Qwen/Qwen2.5-Math-1.5B-Instruct`, scores `372/500 = 74.4%` on MATH-500 when
given `max_new_tokens = 1536`. GSM8K correctness-only GRPO does not improve
that high-budget benchmark: it scores `371/500 = 74.2%`.

The interesting result appears when the benchmark is evaluated with a much
tighter generation budget, `max_new_tokens = 512`. Under that cap, the base
model drops to `258/500 = 51.6%`. Correctness-only GRPO barely changes this:
`260/500 = 52.0%`. But correctness-plus-brevity GRPO improves the constrained
benchmark to `291/500 = 58.2%`. A paired McNemar test gives `p = 0.0020` versus
the base model and `p = 0.0039` versus correctness-only GRPO.

My interpretation is that this is evidence for a constrained capability
improvement, not a general MATH-500 capability improvement. Brevity training
made the model much less likely to run into the 512-token wall, and under that
wall it answered more problems correctly. When the wall is mostly removed
(`max_new_tokens = 1536`), the brevity-trained model is worse than the base
model: `301/500 = 60.2%`. So the result is : "when reasoning budget is scarce, rewarding
correct short solutions moves the model to a better budgeted-reasoning."

## Relevant Files

The code for the experiment is in this folder:

```text
README.md             # this summary
requirements.txt      # Python dependencies
train_rl_math.py      # GRPO training for correctness and brevity rewards
evaluate_math.py      # greedy MATH-500 evaluation for base/adapters
compare_runs.py       # paired significance tests
data_utils.py         # dataset loading and prompt formatting
math_utils.py         # answer extraction, verification, and length scoring
model_utils.py        # model, tokenizer, and LoRA loading helpers
run_experiment.py     # older orchestration helper
runpod_saved/         # downloaded result archives from RunPod
```

The result archives used for the numbers below are:

```text
runpod_saved/17137_20260606_231428_01_base_eval/base_eval_outputs.tgz
runpod_saved/30105_20260608_123716_02_train_correctness/correctness_training_outputs.tgz
runpod_saved/30105_20260608_215859_03_eval_correctness/correctness_eval_outputs.tgz
runpod_saved/30105_20260608_220105_04_train_brevity_l025/brevity_training_outputs.tgz
runpod_saved/30105_20260609_085736_05_eval_brevity_l025/brevity_eval_outputs.tgz
runpod_saved/31556_20260609_201659_10_eval_base_1536/base_1536_eval_outputs.tgz
runpod_saved/39942_20260609_120459_06_eval_correctness_1536/correctness_1536_eval_outputs.tgz
runpod_saved/39942_20260609_122130_07_eval_brevity_l025_1536/brevity_1536_eval_outputs.tgz
```

## Experimental Setup

Base model:

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

RL training data:

```text
openai/gsm8k
config: main
split: train
training examples: 7473
```

Held-out benchmark:

```text
HuggingFaceH4/MATH-500
split: test
benchmark examples: 500
```

Training method:

```text
TRL GRPOTrainer
LoRA adapters, not full fine-tuning
torch_dtype: bfloat16
load_in_4bit: false
seed: 42
```

Both RL runs start from the same base model. The brevity run is not initialized
from the correctness-only adapter.

## Flowchart

```text
                    +--------------------------------------+
                    | Qwen2.5-Math-1.5B-Instruct          |
                    | base model                           |
                    +------------------+-------------------+
                                       |
                                       v
                    +------------------+-------------------+
                    | GSM8K train prompts                   |
                    | sample 4 completions per prompt group |
                    | verify extracted final answers        |
                    +------------------+-------------------+
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
 +--------------+---------------+             +---------------+--------------+
 | train_rl_math.py             |             | train_rl_math.py             |
 | reward-mode correctness      |             | reward-mode correctness_brevity |
 | reward = correct only        |             | reward = correct + shortness |
 +--------------+---------------+             +---------------+--------------+
                |                                             |
                v                                             v
 +--------------+---------------+             +---------------+--------------+
 | correctness LoRA adapter     |             | brevity LoRA adapter         |
 +--------------+---------------+             +---------------+--------------+
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                    +------------------+-------------------+
                    | evaluate_math.py                     |
                    | MATH-500, greedy decoding            |
                    | max_new_tokens = 512 or 1536         |
                    +------------------+-------------------+
                                       |
                                       v
                    +------------------+-------------------+
                    | compare_runs.py / paired test         |
                    | McNemar on matched MATH-500 examples  |
                    +--------------------------------------+
```

## Reward And Verification

Let:

```text
x = a math problem prompt
g = the gold final answer
y = a sampled model completion
extract(y) = the final answer extracted from y
verify(a, g) = the mathematical answer verifier
L(y) = measured generated reasoning length in tokens
L_cap = brevity length cap
lambda = brevity weight
```

The correctness indicator is:

```text
C(y, g) = 1[ verify(extract(y), g) ]
```

The brevity score is:

```text
B(y) = max(0, 1 - L(y) / L_cap)
```

Correctness-only reward:

```text
R_correct(y, g) = C(y, g)
```

Correctness-plus-brevity reward:

```text
R_short(y, g) = C(y, g) * (1 + lambda * B(y))
```

In the main brevity run:

```text
lambda = 0.25
L_cap = 256
L(y) = reasoning_tokens
```

Incorrect answers always receive reward `0`. The brevity bonus only ranks
answers that are already correct. A correct answer with `L(y) >= 256` receives
reward `1.0`; a correct answer with zero measured reasoning tokens would receive
the maximum possible reward `1.25`.

Final answers are extracted in this order:

```text
1. last \boxed{...} expression
2. GSM8K #### delimiter, when present
3. "final answer is", "final answer:", "answer is", or "answer:"
4. last non-empty line
```

Equivalence is checked with `math-verify` when available, then normalized string
matching, then simple numeric equivalence. For brevity length, `reasoning_tokens`
means tokenizer tokens before the final answer marker, such as `\boxed`,
`final answer`, or `answer:`.

## GRPO Objective

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

TRL's GRPO trainer also keeps the policy near a reference model with a KL term:

```text
Loss(theta) = -J_policy(theta)
              + beta * KL(pi_theta(. | x) || pi_ref(. | x))
```

For these runs:

```text
beta = 0.04
num_generations = 4
max_prompt_length = 768
max_completion_length = 512
temperature = 0.9
top_p = 0.95
learning_rate = 1e-5
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
LoRA target modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

## Exact Training Settings

The correctness-only run used:

```text
reward_mode = correctness
max_steps = 1000
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
num_generations = 4
save_steps = 50
save_total_limit = 4
```

The brevity run used:

```text
reward_mode = correctness_brevity
brevity_weight = 0.25
brevity_length_cap = 256
brevity_measure = reasoning_tokens
max_steps = 1000
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
num_generations = 4
save_steps = 50
save_total_limit = 4
```

The two settings both give an effective GRPO batch of `8` completions per
optimizer step, organized as roughly `2` prompts times `4` completions. The logs
report `1000` optimizer steps and `8000` sampled completions total. Since GSM8K
train has `7473` examples, this was about `0.27` of one epoch by TRL's accounting.

## Evaluation Settings

The constrained benchmark result used greedy decoding:

```text
dataset = HuggingFaceH4/MATH-500
split = test
temperature = 0.0
top_p = 1.0
max_new_tokens = 512
```

The high-budget benchmark rerun used the same evaluation code but:

```text
max_new_tokens = 1536
batch_size = 256
```

One minor artifact detail: the original base `max_new_tokens = 512` run used
`batch_size = 4`, while the adapter evaluations used `batch_size = 256`.
Because decoding was greedy (`temperature = 0.0`), this should not change the
answers, but it is worth recording.

## More Details On The Results

### Constrained MATH-500, `max_new_tokens = 512`

The key comparison is under the 512-token generation cap:

| Run | Accuracy | Delta vs base | Median completion tokens | Median reasoning tokens | Median correct reasoning tokens | Max-token completions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | `258/500 = 51.6%` | - | `477.0` | `465.5` | `337.0` | `216/500 = 43.2%` |
| Correctness-only GRPO | `260/500 = 52.0%` | `+0.4` points | `467.5` | `459.0` | `331.0` | `220/500 = 44.0%` |
| Correctness + brevity GRPO | `291/500 = 58.2%` | `+6.6` points | `145.0` | `132.0` | `110.0` | `47/500 = 9.4%` |

This is the main positive result. Correctness-only GRPO did almost nothing on
the constrained benchmark. Brevity GRPO changed the length distribution
dramatically: median reasoning length fell from `465.5` tokens for the base
model to `132.0` tokens, and the rate of completions hitting the 512-token cap
fell from `43.2%` to `9.4%`.

The paired significance tests are:

```text
Base vs correctness-only:
  base-only correct = 8
  correctness-only correct = 10
  discordant examples = 18
  exact McNemar p = 0.8145

Base vs brevity:
  base-only correct = 38
  brevity-only correct = 71
  discordant examples = 109
  exact McNemar p = 0.0020

Correctness-only vs brevity:
  correctness-only correct = 39
  brevity-only correct = 70
  discordant examples = 109
  exact McNemar p = 0.0039
```

So the constrained improvement is not just noise in the aggregate accuracy.
The brevity adapter solved many more examples that the base and correctness-only
models missed than it lost in the opposite direction.

### High-Budget MATH-500, `max_new_tokens = 1536`

The same models were also evaluated with a much larger generation budget:

| Run | Accuracy | Delta vs base | Median completion tokens | Median reasoning tokens | Median correct reasoning tokens | Max-token completions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | `372/500 = 74.4%` | - | `471.5` | `461.0` | `409.0` | `8/500 = 1.6%` |
| Correctness-only GRPO | `371/500 = 74.2%` | `-0.2` points | `467.5` | `459.0` | `408.0` | `9/500 = 1.8%` |
| Correctness + brevity GRPO | `301/500 = 60.2%` | `-14.2` points | `145.0` | `132.0` | `112.0` | `8/500 = 1.6%` |

This is the main caveat. Once the benchmark gives the base model enough room to
reason, the base model is already much better. Correctness-only GRPO still
does not help. Brevity GRPO keeps producing short answers, but those short
answers are no longer enough for many MATH-500 problems.

The paired tests reflect the same picture:

```text
Base vs correctness-only:
  exact McNemar p = 1.0000

Base vs brevity:
  exact McNemar p = 2.23e-11
```

So the brevity run is significantly better in the constrained 512-token regime
and significantly worse in the high-budget 1536-token regime.

### GRPO Training Outcomes

The correctness-only training run finished `1000/1000` optimizer steps in about
`9.07` hours. The final logged training values were:

```text
reward = 0.925
completion_length = 280.65
KL = 3.72e-5
train_loss = 5.23e-6
```

Across the saved trainer-state logs, the median logged completion length was
`286.55` tokens and the median logged KL was `1.16e-4`. The near-zero KL and
near-saturated reward are consistent with the later benchmark result:
correctness-only GSM8K RL did not move the model much.

The brevity training run finished `1000/1000` optimizer steps in about `3.33`
hours. The final logged training values were:

```text
reward = 0.8918
completion_length = 93.825
KL = 0.3377
train_loss = 0.00840
```

Across the saved trainer-state logs, the median logged completion length was
`118.9` tokens and the median logged KL was `0.2571`. This run moved the policy
much more than correctness-only RL. It successfully taught the model to produce
shorter solutions, but the benchmark results show that this is useful only when
the evaluation itself is token-budget constrained.

### Main Lesson

The cleanest summary is:

```text
High budget, 1536 tokens:
  base = 74.4%
  correctness-only RL = 74.2%
  brevity RL = 60.2%

Constrained budget, 512 tokens:
  base = 51.6%
  correctness-only RL = 52.0%
  brevity RL = 58.2%
```

This supports a narrower version of the compression hypothesis: brevity can be
an instrumentally useful RL signal when the model is forced to answer inside a
tight reasoning budget. It does not show that shorter reasoning is generally
better, and it does not show that GSM8K GRPO improves unconstrained MATH-500
performance for this already strong Qwen math model.

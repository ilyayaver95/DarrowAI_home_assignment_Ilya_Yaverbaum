# Report: defendant extraction

Final pipeline: `gpt-4.1`, prompt v3, deterministic pre- and post-processing.

| Measured on | P | R | F1 |
|---|---|---|---|
| dev, labels as given | 0.867 | 0.904 | **0.885** (95% interval 0.64-0.99) |
| dev, documented label corrections | 0.990 | 0.980 | 0.985 |
| my audited subset of eval (28 docs) | 1.000 | 1.000 | 1.000 |

The first row is the comparable number; a hand-typed perfect reading of the captions scores 0.899, the ceiling.

## 1. What I measured, and why

I read all 15 dev documents before writing a prompt. The labels record organizations only, the primary legal name, and inconsistent designators.

- **Matching**: one normalizer for labels and predictions, one-to-one per document. Headline is the **core** tier (name equal, designator ignored); strict and fuzzy are also printed.
- **Aggregation**: micro, with macro beside it.
- **Beyond 15 labels**: a hand-typed oracle, documented label corrections, my own labels for 28 eval documents (biased: written after seeing output), label-free KPIs.

## 2. Iterations (core micro F1)

| Change | Model | dev raw | dev corrected | eval audit |
|---|---|---|---|---|
| One-sentence prompt | 4o-mini | 0.842 | 0.928 | - |
| Individuals and Does moved out of `defendants` | 4o-mini | 0.885 | 0.975 | 0.768 |
| Prompt v1, ten complaint conventions | 4o-mini | 0.862 | 0.950 | 0.802 |
| Same prompt, larger model | 4.1 | 0.867 | 0.975 | 1.000 |
| Font-shift decoding and check-list guard in code | 4.1 | 0.867 | 0.975 | 0.993 |
| Prompts v2, v3: affiliates, fused parties (final) | 4.1 | 0.881 | 0.990 | 1.000 |
| Prompt v4, shorter evidence. **Rejected** | 4.1 | 0.896 | 0.985 | 0.974 |

- **Dev saturated at step 2.** The failures that mattered were in eval: a check-list form returned 18 options as defendants; two documents had shifted character codes.
- **The model mattered more than the prompt**: 0.768 to 0.973 on the audit subset with one sentence.
- **Raw dev rewards the wrong behaviour**: the rejected step scores best because it dropped a real, unlabeled defendant.

## 3. Error analysis, final run on dev (13 FP, 9 FN)

| Cause | FP | FN | Example |
|---|---|---|---|
| Label: scrambled | 5 | 5 | "motor toyota corporation, japanese a" |
| Label: body-only defendants missing | 3 | 0 | three Progressive companies |
| Label: typo, omission, truncation | 4 | 3 | "bytendance", TikTok Pte. Ltd. |
| Data and model: interleaved caption | 1 | 1 | D. Longo, LLC becomes "D. LLC" |

Twenty of 22 errors are label problems; one name is a model error.

## 4. Scope decisions

Decided from the dev labels where the brief is silent:

- **Individuals and Does** are left out of `defendants`, as in every dev label, and kept in `excluded_parties` (`--policy raw` restores them).
- **Defendants named only in the body** are included, tagged `source: body`: the complaint names them as defendants, though the dev labels do not.
- **A blank check-list** returns `[]`: an unmarked list does not show who is sued.
- **A fictitious entity with a real trade name** is kept, flagged as a placeholder.

## 5. What does not work

Found by attacking the finished pipeline. None of it is fixed.

**Wrong or doubtful in the submitted `predictions.jsonl`**

- `cd8b20`: "IV INC.", "PED INC.", "PROTECT, INC." are wrong (interleaved caption); the defendants are JLJ IV Enterprises, Inc. and PED Protect, Inc.
- `c83db3`: "Glenmark Pharmaceuticals, Inc. USA" is not split into name and designator.
- `ca7a70`: two different companies named "Yin Wall City, Inc." are emitted once.
- `ce8874`: pro se filing; the three names are uncertain.
- `ca03b4`, `cc4eaf`, `c5a737` follow the scope decisions in section 4.
- `us_state_of_registration`, `name_quality`, `is_organization`: no labels, spot-checked only.

**Breaks on input outside dev and eval**

- **Prompt injection.** 3 of 4 attempts changed the output; one replaced every real defendant with an invented company. Grounding cannot catch a planted name.
- **Unseen party types.** Third-party, intervenor, nominal and relief defendants are dropped; seized currency becomes an organization.
- **Heuristics misfire.** The check-list guard can drop a real defendant; the placeholder rule excludes a real "ABC Corporation".
- **Size.** About 86 parties would exceed the output limit (estimated). No spend ceiling.
- **Command line.** `--concurrency 0` hangs; `--output` equal to `--input` overwrites the input.
- **Small model.** On `gpt-4o-mini` this prompt marks 20 officers as organizations.

**Weaker than the numbers suggest**

- Fifteen labeled documents; the corrections, oracle and audit labels are all mine.
- **The prompt is overfit.** Against one sentence it changes 7 of 55 documents, all tuned on, and passes 20 of 23 fresh probes where one sentence passes 21.
- **Not deterministic.** Eval was identical across three fresh runs; 2 of 15 dev documents changed.
- No tests for the entry points or most of `monitor.py`. Python 3.14 only.

## 6. With more time

1. Injection hardening: delimiters a document cannot close, a structural check in code.
2. A labeling spec and 100+ documents labeled by someone else.
3. Caption repair from PDF coordinates.
4. Voting across three samples on flagged documents.
5. A smaller prompt rebuilt from the probes; a cheaper tier on `gpt-4o-mini`.
6. Chunked extraction for long captions, spend guards, input validation.

Not built on purpose: few-shot examples (they would contaminate the only labeled set), retrieval, an LLM judge.

## 7. Cost, time, tools

- **Cost**: 277k input, 32k-41k output tokens per 55 documents; $0.81-0.88 on `gpt-4.1`, $0.06 on `gpt-4o-mini`. Everything: about $11 of $20. Model time 311-398 s, 50-95 s wall.
- **Model**: none came with the key; `gpt-4.1-mini` returned 403, so I chose among those allowed.
- **Time**: about 3 hours for the pipeline and sections 1 to 4. I then went over: bug fixes, the prompt check and section 5.
- **AI assistant**: Claude Code, used throughout.

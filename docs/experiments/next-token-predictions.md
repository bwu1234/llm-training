# Next-token predictions

Prompt: <code>"Every effort moves you"</code>

Checkpoint: <code>"/Users/bwu19/llm-training/runs/chapter-reproduction-20261025/port-runs/20261025T020735Z-8779a178/model.pt"</code>

Training update: 90; device: cpu.

Probabilities use the full-vocabulary softmax at temperature 1. Top-k only limits
the displayed candidates; it does not renormalize probabilities or change generation.
Each step appends the greedy winner. Quoted token strings expose leading spaces
and escapes such as `\n`. A token can be part of a word or a UTF-8 character;
the JSON report preserves exact token bytes. Percentages are rounded for display.

## Step 1

Context: <code>"Every effort moves you"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 1701 | <code>"?\""</code> | 22.8895% | yes |
| 2 | 760 | <code>" know"</code> | 18.4972% |  |
| 3 | 287 | <code>" in"</code> | 2.8004% |  |
| 4 | 910 | <code>" say"</code> | 2.6309% |  |
| 5 | 1807 | <code>" thought"</code> | 2.3265% |  |

Other tokens: 50.8554% probability mass.

## Step 2

Context: <code>"Every effort moves you?\""</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 198 | <code>"\n"</code> | 97.0664% | yes |
| 2 | 632 | <code>" It"</code> | 0.6132% |  |
| 3 | 314 | <code>" I"</code> | 0.0968% |  |
| 4 | 1114 | <code>" For"</code> | 0.0764% |  |
| 5 | 1375 | <code>" She"</code> | 0.0579% |  |

Other tokens: 2.0894% probability mass.

## Step 3

Context: <code>"Every effort moves you?\"\n"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 198 | <code>"\n"</code> | 69.2938% | yes |
| 2 | 1 | <code>"\""</code> | 11.5995% |  |
| 3 | 1544 | <code>"He"</code> | 4.7666% |  |
| 4 | 6653 | <code>"His"</code> | 2.6075% |  |
| 5 | 5297 | <code>"Yes"</code> | 0.8746% |  |

Other tokens: 10.8579% probability mass.

## Step 4

Context: <code>"Every effort moves you?\"\n\n"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 1 | <code>"\""</code> | 94.6745% | yes |
| 2 | 1544 | <code>"He"</code> | 0.7277% |  |
| 3 | 6653 | <code>"His"</code> | 0.2936% |  |
| 4 | 1870 | <code>"And"</code> | 0.1610% |  |
| 5 | 40 | <code>"I"</code> | 0.1493% |  |

Other tokens: 3.9940% probability mass.

## Step 5

Context: <code>"Every effort moves you?\"\n\n\""</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 5297 | <code>"Yes"</code> | 82.6205% | yes |
| 2 | 40 | <code>"I"</code> | 1.6070% |  |
| 3 | 5812 | <code>"Oh"</code> | 0.9864% |  |
| 4 | 257 | <code>" a"</code> | 0.3581% |  |
| 5 | 198 | <code>"\n"</code> | 0.3257% |  |

Other tokens: 14.1023% probability mass.

## Step 6

Context: <code>"Every effort moves you?\"\n\n\"Yes"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 438 | <code>"--"</code> | 99.3649% | yes |
| 2 | 7121 | <code>" pushed"</code> | 0.0254% |  |
| 3 | 11 | <code>","</code> | 0.0123% |  |
| 4 | 736 | <code>" back"</code> | 0.0108% |  |
| 5 | 276 | <code>"ed"</code> | 0.0105% |  |

Other tokens: 0.5761% probability mass.

## Step 7

Context: <code>"Every effort moves you?\"\n\n\"Yes--"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 37121 | <code>"quite"</code> | 70.0145% | yes |
| 2 | 292 | <code>"as"</code> | 1.7803% |  |
| 3 | 40 | <code>"I"</code> | 1.1558% |  |
| 4 | 1165 | <code>" too"</code> | 0.9022% |  |
| 5 | 392 | <code>"and"</code> | 0.8393% |  |

Other tokens: 25.3079% probability mass.

## Step 8

Context: <code>"Every effort moves you?\"\n\n\"Yes--quite"</code>

| Rank | Token ID | Token text | Probability | Selected |
| --- | --- | --- | --- | --- |
| 1 | 1035 | <code>" ins"</code> | 89.9085% | yes |
| 2 | 1498 | <code>" able"</code> | 0.2237% |  |
| 3 | 27846 | <code>" glanced"</code> | 0.2024% |  |
| 4 | 416 | <code>" by"</code> | 0.1556% |  |
| 5 | 329 | <code>" for"</code> | 0.1206% |  |

Other tokens: 9.3893% probability mass.

## Generated text

<code>"Every effort moves you?\"\n\n\"Yes--quite ins"</code>


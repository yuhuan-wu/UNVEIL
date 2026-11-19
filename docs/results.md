# Results

Scores of the released checkpoint, from `scripts/evaluate.py` with the default `--shards 8`.

## PCM-Bench

| Set | Metric | Value | | Set | Metric | Value |
| --- | --- | --- | --- | --- | --- | --- |
| DUTS-TE | S-measure | 0.9473 | | SBU | BER | 2.8322 |
| ECSSD | S-measure | 0.9559 | | ISTD | BER | 0.8398 |
| HKU-IS | S-measure | 0.9438 | | Trans10K | IoU | 0.9560 |
| DUT-OMRON | S-measure | 0.9011 | | MSD | IoU | 0.9636 |
| PASCAL-S | S-measure | 0.8968 | | BSDS500 | ODS | 0.8561 |
| CAMO | S-measure | 0.9177 | | SK-LARGE | ODS | 0.7924 |
| COD10K | S-measure | 0.9325 | | SYM-PASCAL | ODS | 0.6469 |
| NC4K | S-measure | 0.9265 | | | | |

The task scores are 92.9, 92.6, 98.2, 95.6, 96.4, 85.6 and 72.0, with a mean of 90.5, as in Tab. 4 of the paper.
Two scorings of the same edge or skeleton maps differ by about 1e-4 in ODS, since pyEdgeEval samples outlier pixels
with a clock-seeded generator.

## Extra test sets and LVIS retention

The six extra test sets give a BER of 5.98 on UCF, IoU of 0.887 on GSD and 0.752 on PMD, an S-measure of 0.945 on
CHAMELEON and ODS of 0.703 on BIPEDv2 and 0.879 on WH-SYMMAX, the values of Tab. 15 of the paper. LVIS retention
(`--stage retention`) gives a mean IoU of 0.8408 for UNVEIL and 0.8307 for SAM 3's own semantic path over the 2,294
held-out pairs.

## Presence decisions

The presence decisions (`--stage decisions`) give the values of Tab. 6 of the paper. Per source, the run behind the
paper gives the same values, except 58.4 for single queries on USC12K.

| Query | Answers when present | Says "none" when absent | COD10K background | USC12K | LVIS |
| --- | --- | --- | --- | --- | --- |
| single | 98.8 | 84.7 | 97.6 | 58.5 | 98.0 |
| joint | 94.3 | 97.5 | 99.0 | 95.6 | 98.0 |

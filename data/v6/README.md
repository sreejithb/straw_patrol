# data/v6: train / calib / test

| split | generated | CoCoLoFa | MAFALDA |
|---|---|---|---|
| train | all earlier generated training files | official train split, top-level comments, capped per type | not used |
| calib | generated calibration set | official dev split slice | not used |
| test  | generated eval sets | official test split slice | gold set (out-of-domain) |

Not in any split (reserved): the rest of CoCoLoFa train, MAFALDA's 9,545 unlabeled texts.

Rules
- Public datasets keep their own splits; a CoCoLoFa article never appears in two splits.
- A text appears in one split only; train records that duplicate or share an 8-word run with a calib/test text are dropped.
- CoCoLoFa types are mapped to one canonical label of ours for training (appeal to authority -> Appeal to false authority, appeal to majority -> Appeal to Common Belief,
  appeal to worse problems -> Relative privation, appeal to tradition -> Appeal to Tradition); at test time several names count as correct (`gold`).
- Public data have no explanation (`answer` is null): explanations are generated later, when the training file is built.
- Public-data files are local only (CoCoLoFa states no licence, MAFALDA is CC-BY-SA); only MANIFEST.json and this README are tracked in git.

Build: `python make_splits.py` (deterministic, seed 0).

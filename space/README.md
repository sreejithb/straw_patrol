---
title: Fallacy checker
emoji: 🧐
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12.12
app_file: app.py
pinned: false
short_description: Name the logical fallacy in a comment, with a confidence.
---

# Fallacy checker

A demo of a LoRA fine-tune of Qwen2.5-1.5B-Instruct that names the logical fallacy in a short comment.

- Output: the most likely fallacy, other possible ones when the model is unsure, or "no fallacy", plus a one-sentence explanation.
- Confidence is calibrated on held-out data (temperature 2.03). Names are shown until they cover 80% of the probability, at most three.
- Limits: on real online comments it is right roughly 30% to 60% of the time in our tests. It is an experiment.

Settings this Space needs: secret `HF_TOKEN` (read access) and variable `ADAPTER_REPO` (the adapter repo).
Hardware: ZeroGPU (a GPU is attached only while a request runs). The app scores all 187 label names for every text, which is slow on CPU.

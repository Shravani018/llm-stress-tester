# 🧪 llm-stress-tester



A framework to evaluate how open-source language models handle adversarial prompts. Tests across hallucination traps and jailbreak scenarios.

## How it works

Each prompt targets a specific failure mode: fabricated citations, fake statistics, roleplay bypasses, prompt injections, and more. The model generates a response, which is then scored against known safe and unsafe signal keywords. A robustness score is computed as the percentage of prompts handled correctly, and a full markdown report is saved per run.

## Scoring

| Outcome | Meaning |
|---------|---------|
| Safe | Model refused, expressed uncertainty, or flagged the issue |
| Unsafe | Model produced the harmful or hallucinated content |
| Unclear | Mixed or no signals detected |

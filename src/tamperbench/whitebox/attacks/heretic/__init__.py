r"""Heretic: fully automatic, optimisation-driven refusal ablation (abliteration).

Heretic edits attention out-projection and MLP down-projection weights to remove
refusal-related directions (Arditi et al., 2024), and runs an Optuna search over the
direction choice and per-layer ablation weights that minimises keyword-detected
refusals and first-token KL divergence on harmless prompts. Each trial on the
resulting Pareto front is a candidate attacked model.

Code: https://github.com/p-e-w/heretic (AGPL-3.0), driven here as a separate process
at commit 3521f8648a0dccf6e12a92666862632235fac7e6.

```
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```
"""

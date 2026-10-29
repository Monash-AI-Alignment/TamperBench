r"""Safety and over-refusal evaluations scored by the Jev judge (TypeSafe System One).

- `jev_strong_reject`: StrongREJECT's forbidden prompts, generated exactly as the
  `strong_reject` eval does, then judged by Jev for refusal and for a substantive
  harmful instance. Attack success is a harmful response, so a refusal that still
  leaks counts as success. Objective: `jev_harmful_rate`.
- `jev_xstest`: XSTest prompts (Röttger et al., 2024), judged by Jev for refusal.
  Objective: over-refusal on the safe prompts, `jev_xstest_safe_refusal_rate`.

The judge prompts are ported from Monash-AI-Alignment/open-steering
(`open_steering/judge.py`, `JevSafetyJudge`); see `tamperbench.whitebox.utils.jev`.

XSTest citation:

```
@inproceedings{rottger2024xstest,
  title={{XST}est: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models},
  author={R{\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle={Proceedings of NAACL},
  year={2024}
}
```
"""

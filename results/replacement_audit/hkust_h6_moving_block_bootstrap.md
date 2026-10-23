# HKUST h=6 moving-block bootstrap

Daily blocks were resampled within each of 24 systems (2,000 replicates, seed 42).

- Persistence MAE: **0.185542**
- LightGBM MAE: **0.059723**
- Skill: **67.81%** (95% CI **67.37%–68.24%**)

The interval retains daily temporal dependence but is descriptive rather than a formal Diebold–Mariano test.

# Cross-testbed context at h=6

This table is a transparent context check, not an apples-to-apples replacement for the locked DKASC/PVOD analysis. HKUST has a full calendar-year test and uses persistence; the original testbeds use smart persistence and their historical partitions.

| Testbed | Systems | Test period | Forecast | Reference | MAE | Reference MAE | Skill | N |
|---|---:|---|---|---|---:|---:|---:|---:|
| DKASC | 1 | locked original test partition | LightGBM | smart persistence | 0.075199 | 0.140678 | 46.55% | 7219 |
| PVOD pooled | 8 | 22 Apr–13 Jun 2019 | LightGBM | smart persistence | 0.091031 | 0.184438 | 49.70% | 4076 |
| HKUST campus fleet | 24 | calendar year 2023 | LightGBM | persistence | 0.059732 | 0.183454 | 67.44% | 199832 |

The raw DKASC/PVOD observations remain provider-restricted. Only locked aggregate metrics are read here; HKUST observations and derived predictions remain private.

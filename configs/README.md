# configs/

Two kinds of configuration, kept strictly apart.

| Kind | Location | May depend on hardware? | Hashed into run ID? |
|------|----------|-------------------------|---------------------|
| Scientific (per experiment condition) | `configs/<EXP_ID>_*.yaml` (not yet created) | **No** | Yes |
| Engineering / runtime | `configs/runtime/` | Yes | No (recorded in provenance) |
| Templates | `configs/templates/` | n/a | n/a |

Rules (Claude_Code_Plan.txt, decisions D-31 to D-34):

- No value that affects a result is passed on the command line.
- A run whose scientific config hash matches no file here is unreportable.
- If different hardware forces a scientifically meaningful change, it is
  logged as a deviation and approved **before** running.
- Experiment IDs are P0, E1..E7, X1..X6 only.

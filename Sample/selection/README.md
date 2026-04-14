# Selection Module Usage

This folder contains refactored analysis logic previously implemented directly in
`reconstructed_level_selection.ipynb`.

## Structure

- `core/cuts.py`: SR/CR event selection logic
- `core/parallel.py`: parallel chunk processing and cutflow merge
- `yields/mg5.py`: MG5 expected-yield computation
- `yields/sherpa.py`: Sherpa expected-yield computation

## Notebook Imports

Use these imports in notebook cells:

```python
from selection.core.cuts import pass_SR_cuts, pass_low_mjj_cr_cuts, pass_WZ_CR_cuts
from selection.core.parallel import selection_cut_parallel, selection_cut_with_region
from selection.yields import (
    compute_expected_events_mg_by_region,
    compute_expected_events_sherpa_by_region,
)
```

If a running kernel keeps old exports cached after edits, run:

```python
import importlib
import selection.yields as selection_yields
importlib.reload(selection_yields)
```

## Validation Baseline (Yields block)

Expected table (L = 139 fb^-1):

- `WWjj_EW-NWA`: `177.98 / 23.60 / 0.00`
- `WWjj_QCD-NWA`: `14.38 / 7.87 / 0.00`
- `WZjj_EW-NWA`: `9.12 / 1.44 / 13.80`
- `WZjj_QCD-NWA`: `18.47 / 11.46 / 36.71`
- `SUM`: `219.96 / 44.37 / 50.51`

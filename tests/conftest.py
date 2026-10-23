import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture
def synthetic_config(tmp_path):
    """Exercise real loaders and runners without requiring private data files."""
    from dataclasses import replace
    import numpy as np
    import pandas as pd
    from formal.config import DatasetSpec, load_experiment_config

    path = tmp_path / 'synthetic.csv'
    rows = 1000
    pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=rows,freq='15min'),
                      TargetObserved=np.ones(rows),
                      Target=np.linspace(0.,10.,rows))).to_csv(path,index=False)
    config = load_experiment_config(ROOT/'configs/formal_experiment.json')
    return replace(config,datasets=tuple(DatasetSpec(s.name,path) for s in config.datasets))

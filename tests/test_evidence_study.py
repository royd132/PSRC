"""Controls must remove only prior correspondence, not the PSRC input."""
import numpy as np
from formal.evidence_study import controlled_prior, ARMS, SETTINGS


def test_prior_control_contract():
    raw = np.arange(80, dtype=np.float32).reshape(5, 16)
    psrc = raw + 100
    np.testing.assert_array_equal(controlled_prior(raw, psrc, "none", 7), psrc)
    np.testing.assert_allclose(controlled_prior(raw, psrc, "real", 7), (raw+psrc)/2)
    shuffled = controlled_prior(raw, psrc, "shuffle", 7)
    recovered = 2*shuffled-psrc
    np.testing.assert_array_equal(np.sort(recovered, axis=0), np.sort(raw, axis=0))
    assert not np.array_equal(recovered, raw)
    assert len(ARMS) == 7 and len(SETTINGS) == 4

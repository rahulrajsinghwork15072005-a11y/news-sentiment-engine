import numpy as np
import pytest

from app.core.returns import log_returns, simple_returns


def test_simple_returns_basic():
    np.testing.assert_allclose(simple_returns([100, 110, 121]), [0.1, 0.1])


def test_log_returns_basic():
    np.testing.assert_allclose(log_returns([100, 100 * np.e]), [1.0])


def test_returns_reject_short_or_nonpositive():
    with pytest.raises(ValueError):
        simple_returns([100])
    with pytest.raises(ValueError):
        log_returns([100, -1])

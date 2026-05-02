"""Pure-function smoke tests — no live Bloomberg connection required.

These run in CI to catch regressions in the parts of bbg_fetch that don't
depend on a Bloomberg Terminal: ticker parsing, field-name normalization,
input coercion, and the public import surface.
"""

import bbg_fetch
from bbg_fetch import contract_to_instrument, instrument_to_active_ticker
from bbg_fetch._blp_api import _as_list, _normalize_name


def test_version():
    assert bbg_fetch.__version__ == "2.0.1"


def test_contract_to_instrument():
    assert contract_to_instrument("ES1 Index") == "ES"
    assert contract_to_instrument("TY1 Comdty") == "TY"
    assert contract_to_instrument("CL10 Comdty") == "CL"


def test_instrument_to_active_ticker():
    assert instrument_to_active_ticker("ES1 Index", num=1) == "ES1 Index"
    assert instrument_to_active_ticker("ES1 Index", num=3) == "ES3 Index"
    assert instrument_to_active_ticker("TY1 Comdty", num=2) == "TY2 Comdty"


def test_normalize_name():
    assert _normalize_name("PX_LAST") == "px_last"
    assert _normalize_name("Last Price") == "last_price"
    assert _normalize_name("BS-LT-BORROW") == "bs_lt_borrow"


def test_as_list():
    # strings stay atomic — not split into characters
    assert _as_list("AAPL US Equity") == ["AAPL US Equity"]
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list(("a", "b")) == ["a", "b"]
    # non-iterable scalar wrapped in a single-element list
    assert _as_list(42) == [42]


def test_top_level_exports():
    """Things the README promises are importable from bbg_fetch."""
    expected = [
        "fetch_field_timeseries_per_tickers",
        "fetch_vol_timeseries",
        "FX_DICT",
        "IMPVOL_FIELDS_DELTA",
        "DEFAULT_START_DATE",
        "bdp", "bdh", "bds", "disconnect",
    ]
    for name in expected:
        assert hasattr(bbg_fetch, name), f"missing public export: {name}"

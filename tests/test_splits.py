import pandas as pd

def load(n):
    return pd.read_csv(f"reports/{n}.csv")

def test_splits_do_not_overlap():
    t, v, s = (set(load(n).path) for n in ("train", "val", "test"))
    assert not (t & v) and not (t & s) and not (v & s)

def test_labels_are_binary():
    for n in ("train", "val", "test"):
        assert set(load(n).label.unique()) <= {0, 1}

def test_split_sizes():
    assert len(load("train")) > len(load("val")) > 0
import pandas as pd
import pytest

from src.evaluate import make_folds, split
from src.io import load_config

CFG = load_config()


def test_primary_and_supplementary_folds():
    folds = make_folds(CFG)
    assert [(f["name"], f["kind"]) for f in folds] == [
        ("test_c3", "primary"), ("test_c4", "primary"), ("test_c2", "supplementary")]


@pytest.mark.parametrize("fold", make_folds(CFG), ids=lambda f: f["name"])
def test_fold_is_disjoint_and_time_ordered(fold):
    tr, va, te = set(fold["train"]), set(fold["val"]), set(fold["test"])
    assert tr and va and te
    assert not (tr & va or tr & te or va & te)
    assert max(tr) < min(va) <= max(va) < min(te)
    assert not (tr | va | te) & set(CFG["excluded_cohorts"])


def test_split_returns_only_fold_cohorts():
    users = pd.DataFrame({"uid": range(7), "cohort": range(7)})
    for f in make_folds(CFG):
        for part, df in zip(["train", "val", "test"], split(users, f)):
            assert set(df.cohort) == set(f[part])


def test_excluded_cohort_in_fold_is_rejected():
    cfg = {**CFG, "excluded_cohorts": [3], "folds": {"primary_test_cohorts": [4],
                                                     "supplementary_test_cohorts": []}}
    with pytest.raises(ValueError):
        make_folds(cfg)

from src.evaluate import h3_verdict


def v(f1, roc, pr):
    return h3_verdict({"f1": f1, "roc_auc": roc, "pr_auc": pr})[0]


def test_rule_cases():
    up, flat, down_pt, down_sig = (0.02, 0.01, 0.03), (0.005, -0.01, 0.02), (-0.005, -0.02, 0.01), (-0.03, -0.05, -0.01)
    assert v(up, flat, up) == "SUPPORTED"
    assert v(flat, up, flat) == "PARTIAL"
    assert v(flat, flat, up) == "PARTIAL"
    assert v(down_pt, down_pt, down_pt) == "NOT SUPPORTED"          # O2: beats INCONCLUSIVE
    assert v(flat, flat, flat) == "INCONCLUSIVE"
    assert v(up, flat, flat) == "INCONCLUSIVE"                       # O3: F1-only
    assert h3_verdict({"f1": down_sig, "roc_auc": up, "pr_auc": flat}) == (
        "PARTIAL", ["significantly worse on f1"])                     # O4

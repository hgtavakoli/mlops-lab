from lablib.gate import check

BASE = {"mAP50_95": 0.40, "per_class": {"car": 0.60, "person": 0.50, "bike": 0.30}}


def test_identical_model_passes():
    assert check(BASE, BASE, max_overall_drop=0.02, max_class_drop=0.10).passed


def test_small_drop_passes():
    cand = {"mAP50_95": 0.39, "per_class": {"car": 0.58, "person": 0.47, "bike": 0.29}}
    assert check(BASE, cand, 0.02, 0.10).passed


def test_one_class_collapsing_fails_even_if_average_is_fine():
    # Overall drop is only 0.01, but 'person' lost 0.20: the average hides it, the gate must not.
    cand = {"mAP50_95": 0.39, "per_class": {"car": 0.62, "person": 0.30, "bike": 0.31}}
    res = check(BASE, cand, 0.02, 0.10)
    assert not res.passed
    assert [f[0] for f in res.failures] == ["person"]


def test_overall_drop_fails():
    cand = {"mAP50_95": 0.30, "per_class": dict(BASE["per_class"])}
    assert not check(BASE, cand, 0.02, 0.10).passed


def test_missing_class_counts_as_zero():
    cand = {"mAP50_95": 0.40, "per_class": {"car": 0.60, "person": 0.50}}  # 'bike' vanished
    res = check(BASE, cand, 0.02, 0.10)
    assert not res.passed
    assert res.failures[0][0] == "bike"

def test_checking_job():
    assert 5 == 4
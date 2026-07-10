import numpy as np

from semi_dynamic_widid import IncrementalWiDiD, SemiDynamicWiDiD


def make_points(center, n, seed):
    rng = np.random.default_rng(seed)
    points = np.asarray(center, dtype=float) + rng.normal(0, 0.03, size=(n, len(center)))
    return points


def test_refresh_and_scores():
    widid = SemiDynamicWiDiD(
        similarity_threshold=0.92,
        historical_update_threshold=12,
        min_cluster_fraction=0.0,
    )

    widid.partial_fit(make_points([1.0, 0.0], 8, 1), period=1)
    assert widid.updates_since_refresh == 8

    widid.partial_fit(make_points([0.0, 1.0], 8, 2), period=2)
    assert widid.updates_since_refresh == 0
    assert len(widid.snapshot()) >= 2

    scores = widid.score_shift(past_periods=[1], current_periods=[2])
    assert 0.0 <= scores.jsd <= 1.0
    assert 0.0 <= scores.pdis <= 2.0
    assert 0.0 <= scores.pdiv <= 2.0


def test_incremental_baseline_does_not_refresh():
    widid = IncrementalWiDiD(similarity_threshold=0.92)
    widid.partial_fit(make_points([1.0, 0.0], 8, 3), period=1)
    widid.partial_fit(make_points([0.0, 1.0], 8, 4), period=2)
    assert widid.updates_since_refresh == 16


if __name__ == "__main__":
    test_refresh_and_scores()
    test_incremental_baseline_does_not_refresh()
    print("ok")

from __future__ import annotations

import numpy as np

from semi_dynamic_widid import IncrementalWiDiD as OriginalWiDiD


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    period_1 = np.vstack(
        [
            rng.normal([1.0, 0.0], 0.03, size=(10, 2)),
            rng.normal([0.0, 1.0], 0.03, size=(10, 2)),
        ]
    )
    period_2 = np.vstack(
        [
            rng.normal([1.0, 0.0], 0.03, size=(5, 2)),
            rng.normal([0.0, 1.0], 0.03, size=(15, 2)),
        ]
    )

    widid = OriginalWiDiD(similarity_threshold=0.92)
    widid.partial_fit(period_1, period=1)
    widid.partial_fit(period_2, period=2)
    print(widid.snapshot())
    print(widid.score_shift(past_periods=[1], current_periods=[2]))

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/generated")

FEATURES = [
    "participation_intensity",
    "win_rate",
    "bid_position",
    "market_breadth",
    "network_exposure",
]


# ============================================================
# VERITAS VENDOR REGIME ENGINE
#
# Detects changes in a vendor's own historical behavior.
#
# Important design principles:
#
#   1. Zero participation is inactivity, not a behavioral
#      observation.
#
#   2. A zero MAD must never create infinite/astronomical
#      standardized deviations.
#
#   3. Network exposure must be temporal if we want to detect
#      network regime changes.
#
#   4. High win rate alone is not anomalous.
#      A CHANGE in win rate may be.
# ============================================================


def load_data(
    data_dir: str | Path = DATA_DIR,
):
    data_dir = Path(data_dir)

    tenders = pd.read_csv(
        data_dir / "tenders.csv"
    )

    bids = pd.read_csv(
        data_dir / "bids.csv"
    )

    tenders["publication_date"] = pd.to_datetime(
        tenders["publication_date"],
        errors="coerce",
    )

    bids["submitted_at"] = pd.to_datetime(
        bids["submitted_at"],
        errors="coerce",
    )

    return tenders, bids


def _market_key(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["category"].astype(str)
        + "|"
        + df["region"].astype(str)
        + "|"
        + df["procurement_method"].astype(str)
    )


def _safe_scale(
    values: pd.Series,
    global_scale: float = 0.05,
) -> float:
    """
    Robust scale estimator.

    MAD is preferred because procurement behavior is often
    heavy-tailed. When MAD collapses to zero, use a conservative
    empirical floor rather than allowing division by ~0.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return max(
            float(global_scale),
            1e-3,
        )

    median = float(
        values.median()
    )

    mad = float(
        (values - median)
        .abs()
        .median()
    )

    robust_scale = (
        1.4826 * mad
    )

    # IQR provides a second robust estimate.
    q25 = float(
        values.quantile(0.25)
    )

    q75 = float(
        values.quantile(0.75)
    )

    iqr_scale = (
        (q75 - q25) / 1.349
    )

    candidates = [
        robust_scale,
        iqr_scale,
        float(global_scale),
    ]

    return max(
        max(candidates),
        1e-3,
    )


def _robust_z(
    value: float,
    baseline: pd.Series,
    global_scale: float,
) -> float:
    """
    Robust standardized deviation with a scale floor.

    The final cap is deliberate. A regime engine should not let
    a numerical degeneracy turn one binary event into a million-
    sigma observation.
    """

    baseline = pd.to_numeric(
        baseline,
        errors="coerce",
    ).dropna()

    if baseline.empty:
        return 0.0

    center = float(
        baseline.median()
    )

    scale = _safe_scale(
        baseline,
        global_scale=global_scale,
    )

    z = (
        float(value) - center
    ) / scale

    return float(
        np.clip(
            z,
            -8.0,
            8.0,
        )
    )


def _shrunken_rate(
    successes: float,
    trials: float,
    prior_rate: float,
    prior_strength: float = 6.0,
) -> float:
    """
    Empirical-Bayes style shrinkage.

    A vendor with 1 win from 1 tender should not instantly become
    a 100% behavioral regime.
    """

    successes = float(
        max(successes, 0.0)
    )

    trials = float(
        max(trials, 0.0)
    )

    if trials <= 0:
        return float(
            prior_rate
        )

    return float(
        (
            successes
            + prior_strength * prior_rate
        )
        / (
            trials
            + prior_strength
        )
    )


# ============================================================
# 1. CHRONOLOGICAL EVENTS
# ============================================================

def build_vendor_events(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
) -> pd.DataFrame:

    df = bids.merge(
        tenders[
            [
                "tender_id",
                "category",
                "region",
                "procurement_method",
                "publication_date",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    df["market"] = _market_key(
        df
    )

    df = (
        df.sort_values(
            [
                "vendor_id",
                "publication_date",
                "tender_id",
            ]
        )
        .drop_duplicates(
            [
                "vendor_id",
                "tender_id",
            ]
        )
    )

    df["rank"] = pd.to_numeric(
        df["rank"],
        errors="coerce",
    )

    df["is_winner"] = (
        df["rank"] == 1
    )

    bidder_counts = (
        df.groupby(
            "tender_id"
        )["vendor_id"]
        .nunique()
        .rename(
            "tender_bidder_count"
        )
    )

    df = df.merge(
        bidder_counts,
        on="tender_id",
        how="left",
    )

    denominator = (
        df["tender_bidder_count"]
        .clip(lower=2)
        - 1.0
    )

    df["normalized_bid_position"] = (
        (df["rank"] - 1.0)
        / denominator
    ).clip(
        0.0,
        1.0,
    )

    return df[
        [
            "vendor_id",
            "tender_id",
            "publication_date",
            "market",
            "category",
            "region",
            "procurement_method",
            "is_winner",
            "normalized_bid_position",
        ]
    ].copy()


# ============================================================
# 2. TEMPORAL NETWORK EXPOSURE
# ============================================================

def build_monthly_network_exposure(
    events: pd.DataFrame,
) -> pd.DataFrame:

    df = events.copy()

    df["month"] = (
        df["publication_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    rows = []

    for (
        month,
        tender_group,
    ) in df.groupby("month"):

        tender_vendors = (
            tender_group
            .groupby("tender_id")[
                "vendor_id"
            ]
            .agg(
                lambda x: sorted(
                    set(x)
                )
            )
        )

        partner_sets = {}

        for vendors in tender_vendors:

            for vendor in vendors:

                if vendor not in partner_sets:
                    partner_sets[
                        vendor
                    ] = set()

                partner_sets[
                    vendor
                ].update(
                    v
                    for v in vendors
                    if v != vendor
                )

        vendor_tenders = (
            tender_group.groupby(
                "vendor_id"
            )["tender_id"]
            .nunique()
        )

        for vendor, count in vendor_tenders.items():

            partners = len(
                partner_sets.get(
                    vendor,
                    set(),
                )
            )

            rows.append(
                {
                    "vendor_id": vendor,
                    "month": month,
                    "network_partner_count": partners,
                    "network_exposure": (
                        partners
                        / max(
                            int(count),
                            1,
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 3. MONTHLY VENDOR PANEL
# ============================================================

def build_monthly_panel(
    events: pd.DataFrame,
) -> pd.DataFrame:

    df = events.copy()

    df["month"] = (
        df["publication_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_market = (
        df.groupby("month")[
            "tender_id"
        ]
        .nunique()
        .rename(
            "monthly_tenders"
        )
    )

    grouped = (
        df.groupby(
            [
                "vendor_id",
                "month",
            ]
        )
        .agg(
            tenders=(
                "tender_id",
                "nunique",
            ),
            wins=(
                "is_winner",
                "sum",
            ),
            mean_bid_position=(
                "normalized_bid_position",
                "mean",
            ),
            market_count=(
                "market",
                "nunique",
            ),
        )
        .reset_index()
    )

    grouped = grouped.merge(
        monthly_market,
        on="month",
        how="left",
    )

    grouped["participation_intensity"] = (
        grouped["tenders"]
        / grouped["monthly_tenders"].clip(
            lower=1
        )
    )

    # Do not use raw wins/tenders.
    # Shrink the rate toward the vendor's own long-run/global
    # reference rate later.
    grouped["raw_win_rate"] = (
        grouped["wins"]
        / grouped["tenders"].clip(
            lower=1
        )
    )

    grouped["market_breadth"] = (
        grouped["market_count"]
        / grouped["tenders"].clip(
            lower=1
        )
    )

    # Normalize the panel schema to the analytical feature names
    # consumed by the regime engine.
    grouped["bid_position"] = (
        grouped["mean_bid_position"]
    )

    network = build_monthly_network_exposure(
        df
    )

    grouped = grouped.merge(
        network,
        on=[
            "vendor_id",
            "month",
        ],
        how="left",
    )

    grouped[
        [
            "network_partner_count",
            "network_exposure",
        ]
    ] = grouped[
        [
            "network_partner_count",
            "network_exposure",
        ]
    ].fillna(0.0)

    return grouped.sort_values(
        [
            "vendor_id",
            "month",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# 4. VENDOR SELF-BASELINE / REGIME CHANGE
# ============================================================

def vendor_regime_scores(
    panel: pd.DataFrame,
    minimum_active_months: int = 8,
) -> pd.DataFrame:

    rows = []

    for vendor_id, group in panel.groupby(
        "vendor_id"
    ):

        group = group.sort_values(
            "month"
        ).reset_index(
            drop=True
        )

        active = group[
            group["tenders"] > 0
        ].copy()

        if len(active) < minimum_active_months:
            continue

        # ----------------------------------------------------
        # Split by chronological ACTIVE observations rather
        # than calendar position.
        # ----------------------------------------------------

        split = len(active) // 2

        if split < 4:
            continue

        history = active.iloc[
            :split
        ].copy()

        recent = active.iloc[
            split:
        ].copy()

        if len(recent) < 4:
            continue

        # ----------------------------------------------------
        # Vendor's long-run empirical win rate.
        # ----------------------------------------------------

        total_wins = float(
            active["wins"].sum()
        )

        total_tenders = float(
            active["tenders"].sum()
        )

        prior_rate = (
            total_wins
            / max(
                total_tenders,
                1.0,
            )
        )

        # ----------------------------------------------------
        # Aggregate feature values.
        #
        # Activity-weighted metrics are used wherever a month
        # with 1 tender should not count as much as a month with
        # 15 tenders.
        # ----------------------------------------------------

        def weighted_mean(
            frame,
            value_column,
        ):
            values = pd.to_numeric(
                frame[value_column],
                errors="coerce",
            ).fillna(0.0)

            weights = frame[
                "tenders"
            ].clip(lower=1)

            return float(
                np.average(
                    values,
                    weights=weights,
                )
            )

        history_features = {
            "participation_intensity": float(
                history[
                    "participation_intensity"
                ].mean()
            ),
            "win_rate": _shrunken_rate(
                history["wins"].sum(),
                history["tenders"].sum(),
                prior_rate,
            ),
            "bid_position": weighted_mean(
                history,
                "mean_bid_position",
            ),
            "market_breadth": weighted_mean(
                history,
                "market_breadth",
            ),
            "network_exposure": weighted_mean(
                history,
                "network_exposure",
            ),
        }

        recent_features = {
            "participation_intensity": float(
                recent[
                    "participation_intensity"
                ].mean()
            ),
            "win_rate": _shrunken_rate(
                recent["wins"].sum(),
                recent["tenders"].sum(),
                prior_rate,
            ),
            "bid_position": weighted_mean(
                recent,
                "mean_bid_position",
            ),
            "market_breadth": weighted_mean(
                recent,
                "market_breadth",
            ),
            "network_exposure": weighted_mean(
                recent,
                "network_exposure",
            ),
        }

        # ----------------------------------------------------
        # Robust standardized changes.
        #
        # Crucially, the baseline is the vendor's historical
        # distribution, but the scale has a floor appropriate
        # to the feature.
        # ----------------------------------------------------

        global_scales = {
            "participation_intensity": 0.01,
            "win_rate": 0.10,
            "bid_position": 0.10,
            "market_breadth": 0.10,
            "network_exposure": 0.10,
        }

        z_scores = {}

        for feature in FEATURES:

            historical_values = (
                history[feature]
                if feature != "win_rate"
                else history[
                    "raw_win_rate"
                ]
            )

            if feature == "win_rate":
                # Build historical monthly shrunken rates.
                historical_values = history.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

            z_scores[
                feature
            ] = _robust_z(
                recent_features[
                    feature
                ],
                historical_values,
                global_scales[
                    feature
                ],
            )

        # ----------------------------------------------------
        # Directional persistence.
        #
        # Instead of allowing a single extreme observation to
        # dominate, measure how consistently recent observations
        # move in the same direction.
        # ----------------------------------------------------

        persistence_values = []

        for feature in FEATURES:

            if feature == "win_rate":
                historical_values = history.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

                recent_values = recent.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

            else:
                historical_values = history[
                    feature
                ]

                recent_values = recent[
                    feature
                ]

            center = float(
                historical_values.median()
            )

            scale = _safe_scale(
                historical_values,
                global_scales[
                    feature
                ],
            )

            recent_z = (
                recent_values
                - center
            ) / scale

            recent_z = np.clip(
                recent_z,
                -8.0,
                8.0,
            )

            direction = np.sign(
                np.median(
                    recent_z
                )
            )

            if direction == 0:
                persistence = 0.0
            else:
                persistence = float(
                    np.mean(
                        (
                            recent_z
                            * direction
                        ) > 1.0
                    )
                )

            persistence_values.append(
                persistence
            )

        persistence_score = float(
            np.mean(
                persistence_values
            )
        )

        # ----------------------------------------------------
        # CUSUM-like persistent shift.
        # ----------------------------------------------------

        cusum_values = []

        for feature in FEATURES:

            if feature == "win_rate":
                historical_values = history.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

                recent_values = recent.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

            else:
                historical_values = history[
                    feature
                ]

                recent_values = recent[
                    feature
                ]

            center = float(
                historical_values.median()
            )

            scale = _safe_scale(
                historical_values,
                global_scales[
                    feature
                ],
            )

            z = np.clip(
                (
                    recent_values
                    - center
                ) / scale,
                -8.0,
                8.0,
            )

            direction = np.sign(
                np.median(z)
            )

            if direction == 0:
                cusum = 0.0
            else:
                aligned = (
                    z * direction
                )

                signal = np.maximum(
                    aligned - 0.75,
                    0.0,
                )

                cusum = float(
                    signal.sum()
                    / np.sqrt(
                        max(
                            len(signal),
                            1,
                        )
                    )
                )

            cusum_values.append(
                cusum
            )

        cusum_score = float(
            np.mean(
                cusum_values
            )
        )

        # ----------------------------------------------------
        # Multivariate distance.
        #
        # Use standardized feature changes. Shrink covariance
        # toward diagonal because vendor histories are short.
        # ----------------------------------------------------

        z_vector = np.array(
            [
                z_scores[f]
                for f in FEATURES
            ],
            dtype=float,
        )

        historical_matrix = []

        for feature in FEATURES:

            if feature == "win_rate":
                values = history.apply(
                    lambda row: _shrunken_rate(
                        row["wins"],
                        row["tenders"],
                        prior_rate,
                    ),
                    axis=1,
                )

            else:
                values = history[
                    feature
                ]

            center = float(
                values.median()
            )

            scale = _safe_scale(
                values,
                global_scales[
                    feature
                ],
            )

            standardized = np.clip(
                (
                    values - center
                ) / scale,
                -8.0,
                8.0,
            )

            historical_matrix.append(
                standardized.to_numpy(
                    dtype=float
                )
            )

        historical_matrix = np.column_stack(
            historical_matrix
        )

        if (
            historical_matrix.shape[0]
            >= 4
        ):
            covariance = np.cov(
                historical_matrix,
                rowvar=False,
            )

            covariance = np.atleast_2d(
                covariance
            )

            if covariance.shape != (
                len(FEATURES),
                len(FEATURES),
            ):
                covariance = np.eye(
                    len(FEATURES)
                )

        else:
            covariance = np.eye(
                len(FEATURES)
            )

        diagonal = np.diag(
            np.diag(covariance)
        )

        covariance = (
            0.80 * diagonal
            + 0.20 * covariance
        )

        covariance += (
            np.eye(
                len(FEATURES)
            )
            * 0.05
        )

        try:
            inverse_covariance = np.linalg.pinv(
                covariance
            )

            mahalanobis_sq = float(
                z_vector.T
                @ inverse_covariance
                @ z_vector
            )

            mahalanobis_distance = float(
                np.sqrt(
                    max(
                        mahalanobis_sq,
                        0.0,
                    )
                )
            )

        except np.linalg.LinAlgError:
            mahalanobis_distance = float(
                np.linalg.norm(
                    z_vector
                )
            )

        # ----------------------------------------------------
        # Convert distances to bounded evidence.
        # ----------------------------------------------------

        multivariate_signal = float(
            1.0
            - np.exp(
                -0.32
                * mahalanobis_distance
            )
        )

        cusum_signal = float(
            1.0
            - np.exp(
                -0.55
                * cusum_score
            )
        )

        regime_signal = (
            0.45
            * multivariate_signal
            + 0.30
            * cusum_signal
            + 0.25
            * persistence_score
        )

        # Require meaningful history.
        history_confidence = min(
            1.0,
            len(history) / 10.0,
        )

        regime_signal *= (
            0.70
            + 0.30
            * history_confidence
        )

        rows.append(
            {
                "vendor_id": vendor_id,
                "history_months": len(
                    history
                ),
                "recent_months": len(
                    recent
                ),
                "historical_tenders": int(
                    history["tenders"].sum()
                ),
                "recent_tenders": int(
                    recent["tenders"].sum()
                ),
                "mahalanobis_distance": round(
                    mahalanobis_distance,
                    4,
                ),
                "cusum_score": round(
                    cusum_score,
                    4,
                ),
                "persistence_score": round(
                    persistence_score,
                    4,
                ),
                "regime_signal": round(
                    regime_signal,
                    4,
                ),
                **{
                    f"z_{feature}": round(
                        z_scores[feature],
                        4,
                    )
                    for feature in FEATURES
                },
                **{
                    f"historical_{feature}":
                        round(
                            history_features[
                                feature
                            ],
                            4,
                        )
                    for feature in FEATURES
                },
                **{
                    f"recent_{feature}":
                        round(
                            recent_features[
                                feature
                            ],
                            4,
                        )
                    for feature in FEATURES
                },
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. EXPLANATIONS
# ============================================================

def attach_explanations(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    result = scores.copy()

    labels = {
        "participation_intensity":
            "participation intensity",
        "win_rate":
            "win rate",
        "bid_position":
            "bid positioning",
        "market_breadth":
            "market breadth",
        "network_exposure":
            "network exposure",
    }

    explanations = []

    for _, row in result.iterrows():

        signals = []

        ranked = sorted(
            [
                (
                    feature,
                    abs(
                        float(
                            row[
                                f"z_{feature}"
                            ]
                        )
                    ),
                )
                for feature in FEATURES
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for feature, magnitude in ranked:

            if magnitude < 1.5:
                continue

            z = float(
                row[
                    f"z_{feature}"
                ]
            )

            direction = (
                "increased"
                if z > 0
                else "decreased"
            )

            signals.append(
                f"{labels[feature]} "
                f"{direction} "
                f"({magnitude:.1f} robust deviations)"
            )

        if not signals:
            signals.append(
                "no single dominant feature; "
                "multivariate behavioral shift"
            )

        explanations.append(
            "; ".join(
                signals[:4]
            )
        )

    result[
        "explanation"
    ] = explanations

    return result


# ============================================================
# 6. SCENARIOS
# ============================================================

SCENARIOS = {
    "REPEATED_NETWORK": [
        "V0001",
        "V0024",
        "V0091",
    ],
    "LEGITIMATE_SPECIALIZATION": [
        "V0010",
        "V0020",
    ],
    "COVER_BIDDING": [
        "V0042",
        "V0043",
    ],
    "BID_ROTATION": [
        "V0051",
        "V0052",
        "V0053",
    ],
    "GEOGRAPHIC_ALLOCATION": [
        "V0061",
        "V0062",
        "V0063",
    ],
    "TIGHT_PRICE_CLUSTER": [
        "V0071",
        "V0072",
        "V0073",
        "V0074",
    ],
}


def print_scenario_audit(
    scores: pd.DataFrame,
):

    print()
    print("=" * 72)
    print("VENDOR REGIME SCENARIO AUDIT")
    print("=" * 72)

    for name, vendors in SCENARIOS.items():

        rows = scores[
            scores[
                "vendor_id"
            ].isin(vendors)
        ]

        if rows.empty:
            print(
                f"{name:<26} NO SCORE"
            )
            continue

        top = rows.sort_values(
            "regime_signal",
            ascending=False,
        ).iloc[0]

        print(
            f"{name:<26}"
            f" max={top['regime_signal']:.3f}"
            f" vendor={top['vendor_id']}"
            f" mahal={top['mahalanobis_distance']:.2f}"
            f" cusum={top['cusum_score']:.2f}"
            f" persistence={top['persistence_score']:.2f}"
        )


# ============================================================
# 7. SELF TEST
# ============================================================

def self_test(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
):

    assert not panel.empty
    assert not scores.empty

    required_panel = {
        "vendor_id",
        "month",
        "participation_intensity",
        "raw_win_rate",
        "mean_bid_position",
        "bid_position",
        "market_breadth",
        "network_exposure",
    }

    required_scores = {
        "vendor_id",
        "mahalanobis_distance",
        "cusum_score",
        "persistence_score",
        "regime_signal",
    }

    assert required_panel <= set(
        panel.columns
    )

    assert required_scores <= set(
        scores.columns
    )

    numeric_columns = [
        "mahalanobis_distance",
        "cusum_score",
        "persistence_score",
        "regime_signal",
    ]

    for column in numeric_columns:
        assert np.isfinite(
            scores[column].to_numpy(
                dtype=float
            )
        ).all()

    assert scores[
        "regime_signal"
    ].between(
        0.0,
        1.0,
    ).all()

    assert scores[
        "persistence_score"
    ].between(
        0.0,
        1.0,
    ).all()

    assert scores[
        "vendor_id"
    ].is_unique

    # Numerical sanity:
    # standardized features must never explode.
    z_columns = [
        c
        for c in scores.columns
        if c.startswith("z_")
    ]

    for column in z_columns:
        assert (
            scores[column].abs().max()
            <= 8.0001
        )

    # Network exposure must actually vary over time.
    network_variation = (
        panel.groupby(
            "vendor_id"
        )[
            "network_exposure"
        ]
        .nunique()
    )

    assert (
        network_variation.gt(1).any()
    )

    print(
        "[PASS] regime engine structural self-test"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("VERITAS VENDOR REGIME ENGINE")
    print("=" * 72)

    tenders, bids = load_data()

    print(
        f"Tenders: "
        f"{tenders['tender_id'].nunique():,}"
        f" | Bids: {len(bids):,}"
    )

    print()
    print(
        "[1/5] Building chronological vendor events"
    )

    events = build_vendor_events(
        tenders,
        bids,
    )

    print(
        f"       {len(events):,} vendor-tender events"
    )

    print(
        "[2/5] Building temporal behavioral panel"
    )

    panel = build_monthly_panel(
        events
    )

    print(
        f"       {panel['vendor_id'].nunique():,} vendors"
        f" | {panel['month'].nunique():,} months"
    )

    print(
        "[3/5] Calculating robust self-baselines"
    )

    scores = vendor_regime_scores(
        panel
    )

    print(
        f"       {len(scores):,} vendors scored"
    )

    print(
        "[4/5] Attaching behavioral explanations"
    )

    scores = attach_explanations(
        scores
    )

    print(
        "[5/5] Writing regime evidence"
    )

    output = (
        DATA_DIR
        / "vendor_regimes.csv"
    )

    scores.to_csv(
        output,
        index=False,
    )

    self_test(
        panel,
        scores,
    )

    print(
        f"[OK] {output}"
    )

    print_scenario_audit(
        scores
    )

    print()
    print("=" * 72)
    print("TOP BEHAVIORAL REGIME SHIFTS")
    print("=" * 72)

    columns = [
        "vendor_id",
        "regime_signal",
        "mahalanobis_distance",
        "cusum_score",
        "persistence_score",
        "z_participation_intensity",
        "z_win_rate",
        "z_bid_position",
        "z_market_breadth",
        "z_network_exposure",
        "explanation",
    ]

    print(
        scores[
            columns
        ]
        .sort_values(
            [
                "regime_signal",
                "persistence_score",
            ],
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/generated")


# ============================================================
# VERITAS BEHAVIORAL RELATIONSHIP ENGINE
#
# Principle:
#
#   Do not ask:
#       "Do A and B bid together unusually often?"
#
#   Ask:
#       "Do A and B bid together unusually often AFTER
#        accounting for the markets in which they operate?"
#
# This is deliberately additive to the existing detector stack.
# ============================================================


def load_data(data_dir: str | Path = DATA_DIR):
    data_dir = Path(data_dir)

    tenders = pd.read_csv(data_dir / "tenders.csv")
    bids = pd.read_csv(data_dir / "bids.csv")

    tenders["publication_date"] = pd.to_datetime(
        tenders["publication_date"],
        errors="coerce",
    )

    tenders["closing_date"] = pd.to_datetime(
        tenders["closing_date"],
        errors="coerce",
    )

    bids["submitted_at"] = pd.to_datetime(
        bids["submitted_at"],
        errors="coerce",
    )

    return tenders, bids


def _safe_ratio(a, b):
    return float(a) / max(float(b), 1e-12)


def _market_key(df):
    return (
        df["category"].astype(str)
        + "|"
        + df["region"].astype(str)
        + "|"
        + df["procurement_method"].astype(str)
    )


# ============================================================
# 1. MARKET-CONDITIONED RELATIONSHIPS
# ============================================================

def conditional_relationships(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
    min_shared_tenders: int = 3,
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

    df["market"] = _market_key(df)

    participation = (
        df[
            [
                "tender_id",
                "vendor_id",
                "market",
            ]
        ]
        .drop_duplicates()
    )

    market_tenders = (
        participation
        .groupby("market")["tender_id"]
        .nunique()
        .to_dict()
    )

    vendor_market_counts = (
        participation
        .groupby(["market", "vendor_id"])["tender_id"]
        .nunique()
    )

    # observed pair participation by market
    observed = defaultdict(int)

    for tender_id, group in participation.groupby("tender_id"):
        vendors = sorted(group["vendor_id"].unique())
        market = group["market"].iloc[0]

        for a, b in combinations(vendors, 2):
            observed[(a, b, market)] += 1

    # Aggregate pair-level market evidence.
    pair_market_rows = defaultdict(list)

    for (a, b, market), shared in observed.items():
        if shared < min_shared_tenders:
            continue

        market_n = market_tenders[market]

        n_a = int(vendor_market_counts.get((market, a), 0))
        n_b = int(vendor_market_counts.get((market, b), 0))

        if n_a == 0 or n_b == 0:
            continue

        expected = (
            n_a
            * n_b
            / max(market_n, 1)
        )

        enrichment = _safe_ratio(
            shared,
            expected,
        )

        # Jaccard-style overlap is useful because it answers:
        # "how much of their activity overlaps?"
        overlap_denominator = (
            n_a
            + n_b
            - shared
        )

        overlap = _safe_ratio(
            shared,
            overlap_denominator,
        )

        pair_market_rows[(a, b)].append(
            {
                "market": market,
                "shared": shared,
                "expected": expected,
                "enrichment": enrichment,
                "overlap": overlap,
                "market_n": market_n,
                "vendor_a_market_tenders": n_a,
                "vendor_b_market_tenders": n_b,
            }
        )

    rows = []

    for (a, b), markets in pair_market_rows.items():

        total_shared = sum(
            x["shared"] for x in markets
        )

        total_expected = sum(
            x["expected"] for x in markets
        )

        conditional_enrichment = _safe_ratio(
            total_shared,
            total_expected,
        )

        # Weight each market by the amount of actual shared activity.
        total_weight = sum(
            x["shared"] for x in markets
        )

        weighted_overlap = _safe_ratio(
            sum(
                x["overlap"] * x["shared"]
                for x in markets
            ),
            total_weight,
        )

        # Market specialization:
        # What fraction of their shared activity is concentrated
        # in their dominant market?
        market_shares = [
            x["shared"] / max(total_shared, 1)
            for x in markets
        ]

        concentration = max(
            market_shares,
            default=0.0,
        )

        rows.append(
            {
                "vendor_a": a,
                "vendor_b": b,
                "shared_tenders": total_shared,
                "expected_conditional_shared": round(
                    total_expected,
                    4,
                ),
                "conditional_enrichment": round(
                    conditional_enrichment,
                    4,
                ),
                "weighted_market_overlap": round(
                    weighted_overlap,
                    4,
                ),
                "market_count": len(markets),
                "market_concentration": round(
                    concentration,
                    4,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "conditional_enrichment",
            "shared_tenders",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 2. CONDITIONAL WIN EFFECT
# ============================================================

def conditional_win_effect(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
    min_observations: int = 8,
) -> pd.DataFrame:

    df = bids.merge(
        tenders[
            [
                "tender_id",
                "category",
                "region",
                "procurement_method",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    df["market"] = _market_key(df)

    participation = (
        df[
            [
                "tender_id",
                "vendor_id",
                "market",
                "rank",
            ]
        ]
        .drop_duplicates(
            subset=["tender_id", "vendor_id"]
        )
    )

    # Determine actual winners.
    winners = (
        df[df["rank"] == 1]
        [["tender_id", "vendor_id"]]
        .drop_duplicates()
        .rename(columns={"vendor_id": "winner_id"})
    )

    participation = participation.merge(
        winners,
        on="tender_id",
        how="left",
    )

    participation["is_winner"] = (
        participation["vendor_id"]
        == participation["winner_id"]
    )

    # For every tender, evaluate every participating pair.
    rows = []

    for tender_id, group in participation.groupby("tender_id"):
        vendors = sorted(group["vendor_id"].unique())

        winner = (
            group.loc[
                group["is_winner"],
                "vendor_id",
            ]
            .iloc[0]
            if group["is_winner"].any()
            else None
        )

        market = group["market"].iloc[0]

        for a, b in combinations(vendors, 2):

            a_wins = int(winner == a)

            rows.append(
                {
                    "tender_id": tender_id,
                    "market": market,
                    "vendor_a": a,
                    "vendor_b": b,
                    "a_wins": a_wins,
                }
            )

    pair_events = pd.DataFrame(rows)

    if pair_events.empty:
        return pair_events

    output = []

    all_tenders = (
        participation[
            ["tender_id", "vendor_id", "market", "is_winner"]
        ]
        .drop_duplicates()
    )

    for (a, b), pair_group in pair_events.groupby(
        ["vendor_a", "vendor_b"]
    ):

        shared_ids = set(
            pair_group["tender_id"]
        )

        if len(shared_ids) < min_observations:
            continue

        a_history = all_tenders[
            all_tenders["vendor_id"] == a
        ]

        with_b = a_history[
            a_history["tender_id"].isin(shared_ids)
        ]

        without_b = a_history[
            ~a_history["tender_id"].isin(shared_ids)
        ]

        if len(with_b) < min_observations:
            continue

        if len(without_b) < min_observations:
            continue

        with_rate = float(
            with_b["is_winner"].mean()
        )

        without_rate = float(
            without_b["is_winner"].mean()
        )

        uplift = with_rate - without_rate

        # Shrink tiny samples toward zero.
        sample_weight = min(
            1.0,
            np.sqrt(
                min(
                    len(with_b),
                    len(without_b),
                )
                / 30.0
            ),
        )

        shrunk_uplift = uplift * sample_weight

        output.append(
            {
                "vendor_a": a,
                "vendor_b": b,
                "shared_tenders": len(shared_ids),
                "a_win_rate_with_b": round(
                    with_rate,
                    4,
                ),
                "a_win_rate_without_b": round(
                    without_rate,
                    4,
                ),
                "win_uplift": round(
                    uplift,
                    4,
                ),
                "sample_weight": round(
                    sample_weight,
                    4,
                ),
                "shrunk_win_uplift": round(
                    shrunk_uplift,
                    4,
                ),
            }
        )

    result = pd.DataFrame(output)

    if result.empty:
        return result

    return result.sort_values(
        [
            "shrunk_win_uplift",
            "shared_tenders",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 3. RELATIONSHIP PERSISTENCE
# ============================================================

def relationship_persistence(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
) -> pd.DataFrame:

    df = bids.merge(
        tenders[
            [
                "tender_id",
                "publication_date",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    participation = (
        df[
            [
                "tender_id",
                "vendor_id",
            ]
        ]
        .drop_duplicates()
    )

    dated = participation.merge(
        tenders[
            [
                "tender_id",
                "publication_date",
            ]
        ],
        on="tender_id",
        how="left",
    )

    pair_dates = defaultdict(list)

    for tender_id, group in dated.groupby("tender_id"):
        vendors = sorted(
            group["vendor_id"].unique()
        )

        date = group["publication_date"].iloc[0]

        for a, b in combinations(vendors, 2):
            if pd.notna(date):
                pair_dates[(a, b)].append(date)

    rows = []

    for (a, b), dates in pair_dates.items():

        if len(dates) < 3:
            continue

        dates = sorted(dates)

        span_days = (
            dates[-1] - dates[0]
        ).days

        # Number of distinct calendar months with interaction.
        months = {
            (d.year, d.month)
            for d in dates
        }

        month_persistence = min(
            1.0,
            len(months) / 12.0,
        )

        span_persistence = min(
            1.0,
            span_days / 365.0,
        )

        persistence = (
            0.6 * span_persistence
            + 0.4 * month_persistence
        )

        rows.append(
            {
                "vendor_a": a,
                "vendor_b": b,
                "relationship_events": len(dates),
                "span_days": span_days,
                "active_months": len(months),
                "persistence": round(
                    persistence,
                    4,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "persistence",
            "relationship_events",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 4. RELATIONSHIP EVIDENCE FUSION
# ============================================================

def build_relationship_evidence(
    relationships: pd.DataFrame,
    win_effects: pd.DataFrame,
    persistence: pd.DataFrame,
) -> pd.DataFrame:

    if relationships.empty:
        return pd.DataFrame()

    result = relationships.copy()

    result = result.merge(
        win_effects[
            [
                "vendor_a",
                "vendor_b",
                "shrunk_win_uplift",
            ]
        ],
        on=["vendor_a", "vendor_b"],
        how="left",
    )

    result = result.merge(
        persistence[
            [
                "vendor_a",
                "vendor_b",
                "persistence",
            ]
        ],
        on=["vendor_a", "vendor_b"],
        how="left",
    )

    result["shrunk_win_uplift"] = (
        result["shrunk_win_uplift"]
        .fillna(0.0)
    )

    result["persistence"] = (
        result["persistence"]
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Transform native signals into bounded evidence units.
    #
    # Conditional enrichment:
    #   1x = expected
    #   2x = notable
    #   5x = strong
    #   10x+ = extreme
    # --------------------------------------------------------

    result["enrichment_signal"] = np.clip(
        (
            np.log1p(
                result["conditional_enrichment"]
            )
            / np.log1p(10.0)
        ),
        0.0,
        1.0,
    )

    # A positive conditional win uplift is much more meaningful
    # than raw win rate. Cap at +60 percentage points.
    result["win_effect_signal"] = np.clip(
        result["shrunk_win_uplift"] / 0.60,
        0.0,
        1.0,
    )

    # Penalize relationships that are almost entirely confined
    # to one specialized market.
    #
    # This is NOT a blanket "specialized = innocent" rule.
    # It simply reduces the relationship-only evidence when
    # the relationship has no cross-market footprint.
    result["specialization_discount"] = (
        1.0
        - 0.55
        * result["market_concentration"]
    )

    # Strong relationships that span several markets receive
    # more persistence credit.
    result["breadth_signal"] = np.clip(
        result["market_count"] / 4.0,
        0.0,
        1.0,
    )

    # Combine orthogonal-ish relationship dimensions.
    result["relationship_strength"] = (
        0.35
        * result["enrichment_signal"]
        + 0.30
        * result["win_effect_signal"]
        + 0.20
        * result["persistence"]
        + 0.15
        * result["breadth_signal"]
    )

    # Contextual adjustment.
    result["context_adjusted_strength"] = (
        result["relationship_strength"]
        * result["specialization_discount"]
    )

    return result.sort_values(
        [
            "context_adjusted_strength",
            "relationship_strength",
            "shared_tenders",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 5. PLANTED-SCENARIO AUDIT
# ============================================================

SCENARIOS = {
    "REPEATED_NETWORK": ["V0001", "V0024", "V0091"],
    "LEGITIMATE_SPECIALIZATION": ["V0010", "V0020"],
    "COVER_BIDDING": ["V0042", "V0043"],
    "BID_ROTATION": ["V0051", "V0052", "V0053"],
    "GEOGRAPHIC_ALLOCATION": ["V0061", "V0062", "V0063"],
    "TIGHT_PRICE_CLUSTER": ["V0071", "V0072", "V0073", "V0074"],
}


def scenario_rows(df, vendors):
    vendors = set(vendors)

    mask = (
        df["vendor_a"].isin(vendors)
        & df["vendor_b"].isin(vendors)
    )

    return df[mask].copy()


def self_test(
    evidence: pd.DataFrame,
    relationships: pd.DataFrame,
):
    required = {
        "vendor_a",
        "vendor_b",
        "conditional_enrichment",
        "market_concentration",
        "relationship_strength",
        "context_adjusted_strength",
    }

    missing = required - set(evidence.columns)

    assert not missing, (
        f"Missing evidence columns: {sorted(missing)}"
    )

    assert len(relationships) > 0
    assert len(evidence) > 0

    # Values must remain bounded.
    for column in [
        "relationship_strength",
        "context_adjusted_strength",
        "specialization_discount",
    ]:
        assert evidence[column].between(
            0.0,
            1.0,
        ).all(), column

    print("[PASS] behavioral engine structural self-test")


def print_scenarios(evidence: pd.DataFrame):

    print()
    print("=" * 72)
    print("CONDITIONAL RELATIONSHIP SCENARIO AUDIT")
    print("=" * 72)

    scenario_max = {}

    for name, vendors in SCENARIOS.items():

        rows = scenario_rows(
            evidence,
            vendors,
        )

        if rows.empty:
            print(
                f"{name:<26} NO DIRECT PAIR FOUND"
            )
            continue

        top = rows.iloc[0]

        scenario_max[name] = float(
            top["context_adjusted_strength"]
        )

        print(
            f"{name:<26}"
            f" strength={top['context_adjusted_strength']:.3f}"
            f" raw={top['relationship_strength']:.3f}"
            f" enrichment={top['conditional_enrichment']:.2f}x"
            f" concentration={top['market_concentration']:.2f}"
            f" markets={int(top['market_count'])}"
        )

    if (
        "LEGITIMATE_SPECIALIZATION"
        in scenario_max
    ):
        legit = scenario_max[
            "LEGITIMATE_SPECIALIZATION"
        ]

        print()
        print(
            f"LEGITIMATE SPECIALIZATION "
            f"max={legit:.3f}"
        )

        suspicious = [
            value
            for name, value in scenario_max.items()
            if name != "LEGITIMATE_SPECIALIZATION"
        ]

        if suspicious:
            strongest_suspicious = max(
                suspicious
            )

            print(
                f"STRONGEST PLANTED SCENARIO "
                f"max={strongest_suspicious:.3f}"
            )

            if strongest_suspicious > legit:
                print(
                    "[PASS] contextual relationship "
                    "model separates at least one "
                    "planted scenario from specialization"
                )
            else:
                print(
                    "[INFO] relationship layer still "
                    "needs additional evidence families"
                )


def main():

    print("=" * 72)
    print("VERITAS BEHAVIORAL RELATIONSHIP ENGINE")
    print("=" * 72)

    tenders, bids = load_data()

    print(
        f"Tenders: {tenders['tender_id'].nunique():,}"
        f" | Bids: {len(bids):,}"
    )

    print()
    print("[1/4] Market-conditioned relationships")

    relationships = conditional_relationships(
        tenders,
        bids,
    )

    print(
        f"       {len(relationships):,} relationships"
    )

    print("[2/4] Conditional win effects")

    win_effects = conditional_win_effect(
        tenders,
        bids,
    )

    print(
        f"       {len(win_effects):,} relationships"
    )

    print("[3/4] Relationship persistence")

    persistence = relationship_persistence(
        tenders,
        bids,
    )

    print(
        f"       {len(persistence):,} relationships"
    )

    print("[4/4] Evidence fusion")

    evidence = build_relationship_evidence(
        relationships,
        win_effects,
        persistence,
    )

    output = DATA_DIR / "behavioral_relationships.csv"

    evidence.to_csv(
        output,
        index=False,
    )

    self_test(
        evidence,
        relationships,
    )

    print()
    print(
        f"[OK] {output}"
    )

    print_scenarios(
        evidence,
    )

    print()
    print("=" * 72)
    print("TOP CONDITIONAL RELATIONSHIPS")
    print("=" * 72)

    columns = [
        "vendor_a",
        "vendor_b",
        "shared_tenders",
        "conditional_enrichment",
        "market_concentration",
        "market_count",
        "shrunk_win_uplift",
        "persistence",
        "context_adjusted_strength",
    ]

    print(
        evidence[
            columns
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

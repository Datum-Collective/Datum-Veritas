from __future__ import annotations

from itertools import combinations
from math import comb, log10
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/generated")


# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_dir: str | Path = DATA_DIR):
    data_dir = Path(data_dir)

    tenders = pd.read_csv(data_dir / "tenders.csv")
    bids = pd.read_csv(data_dir / "bids.csv")
    vendors = pd.read_csv(data_dir / "vendors.csv")

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

    return tenders, bids, vendors


# ============================================================
# 1. BIDDER CO-OCCURRENCE
# ============================================================

def bidder_cooccurrence(
    bids: pd.DataFrame,
    min_shared_tenders: int = 3,
) -> pd.DataFrame:
    """
    Detect repeated co-bidding between vendor pairs.

    The baseline assumes vendors participate independently.
    Enrichment measures how much more often a pair appears together
    than expected from their individual participation frequencies.

    No scenario_truth information is used.
    """

    participation = (
        bids[["tender_id", "vendor_id"]]
        .drop_duplicates()
        .groupby("vendor_id")["tender_id"]
        .nunique()
    )

    tender_count = bids["tender_id"].nunique()

    tender_vendors = (
        bids[["tender_id", "vendor_id"]]
        .drop_duplicates()
        .groupby("tender_id")["vendor_id"]
        .agg(list)
    )

    pair_counts: dict[tuple[str, str], int] = {}

    for vendor_list in tender_vendors:
        for a, b in combinations(sorted(vendor_list), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = []

    for (vendor_a, vendor_b), observed in pair_counts.items():
        if observed < min_shared_tenders:
            continue

        n_a = int(participation[vendor_a])
        n_b = int(participation[vendor_b])

        expected = (n_a * n_b) / tender_count
        enrichment = observed / max(expected, 1e-9)

        max_overlap = min(n_a, n_b)
        probability = 0.0

        denominator = comb(tender_count, n_b)

        for k in range(observed, max_overlap + 1):
            if tender_count - n_a >= n_b - k:
                numerator = (
                    comb(n_a, k)
                    * comb(tender_count - n_a, n_b - k)
                )
                probability += numerator / denominator

        p_value = min(probability, 1.0)

        rows.append(
            {
                "vendor_a": vendor_a,
                "vendor_b": vendor_b,
                "observed_shared_tenders": observed,
                "vendor_a_tenders": n_a,
                "vendor_b_tenders": n_b,
                "expected_shared_tenders": round(expected, 3),
                "enrichment_ratio": round(enrichment, 3),
                "p_value": p_value,
                "surprise_score": round(
                    -log10(max(p_value, 1e-300)),
                    3,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["surprise_score", "enrichment_ratio", "observed_shared_tenders"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 2. BID PRICE BEHAVIOR
# ============================================================

def bid_spread_anomalies(
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
    min_bids: int = 3,
) -> pd.DataFrame:
    """
    Detect unusually tight bid-price clusters relative to comparable
    procurement peers.

    Raw bid spread is first calculated within each tender. The tender
    is then compared against its category + region + procurement-method
    peer group.

    The detector deliberately separates:
      - observed tightness
      - peer-relative unusualness
      - sample size

    This prevents a tiny tender with three bids from automatically
    dominating the evidence simply because 1 / spread is large.
    """

    df = bids.merge(
        tenders[
            [
                "tender_id",
                "category",
                "region",
                "estimated_value",
                "procurement_method",
            ]
        ],
        on="tender_id",
        how="left",
    )

    df["bid_ratio"] = (
        df["bid_amount"]
        / df["estimated_value"].clip(lower=1e-9)
    )

    rows = []

    for tender_id, group in df.groupby("tender_id"):
        if len(group) < min_bids:
            continue

        ratios = group["bid_ratio"]

        median = ratios.median()
        mad = (ratios - median).abs().median()

        minimum = ratios.min()
        maximum = ratios.max()
        spread = maximum - minimum

        relative_spread = spread / max(abs(median), 1e-9)

        cv = (
            ratios.std(ddof=0)
            / max(abs(ratios.mean()), 1e-9)
        )

        rows.append(
            {
                "tender_id": tender_id,
                "category": group["category"].iloc[0],
                "region": group["region"].iloc[0],
                "procurement_method": (
                    group["procurement_method"].iloc[0]
                ),
                "bid_count": len(group),
                "median_bid_ratio": round(median, 5),
                "mad": round(mad, 5),
                "relative_spread": round(
                    relative_spread,
                    5,
                ),
                "coefficient_of_variation": round(
                    cv,
                    5,
                ),
                "minimum_bid_ratio": round(
                    minimum,
                    5,
                ),
                "maximum_bid_ratio": round(
                    maximum,
                    5,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # --------------------------------------------------------
    # Compare each tender against its procurement peer group.
    # --------------------------------------------------------

    peer_columns = [
        "category",
        "region",
        "procurement_method",
    ]

    peer_stats = (
        result.groupby(peer_columns)["relative_spread"]
        .agg(
            peer_median="median",
            peer_mad=lambda x: (
                (x - x.median()).abs().median()
            ),
            peer_count="count",
        )
        .reset_index()
    )

    result = result.merge(
        peer_stats,
        on=peer_columns,
        how="left",
    )

    # Robust deviation from peer behavior.
    result["robust_z"] = (
        (
            result["peer_median"]
            - result["relative_spread"]
        )
        / (
            1.4826
            * result["peer_mad"].clip(lower=0.0001)
        )
    )

    # Empirical percentile inside the peer group.
    #
    # Higher percentile means the tender has a tighter spread than
    # more of its comparable peers. `transform` is deliberately used
    # instead of `groupby().apply()` so the resulting Series retains
    # exactly the same row index as `result`.
    result["tightness_percentile"] = (
        result.groupby(peer_columns)["relative_spread"]
        .transform(
            lambda x: x.rank(
                pct=True,
                method="average",
            )
        )
    )

    # More bidders provide more information about the shape of the
    # competitive distribution. Apply a bounded sample-size weight.
    sample_weight = (
        1.0
        - 1.0 / result["bid_count"].clip(lower=3)
    )

    # Evidence is strongest when:
    #   1. the spread is tighter than peers
    #   2. the deviation is robustly large
    #   3. there are enough competing bids
    #
    # This remains an evidence measure, not a probability of misconduct.
    robust_component = result["robust_z"].clip(lower=0)

    percentile_component = (
        result["tightness_percentile"]
        .clip(lower=0, upper=1)
    )

    result["price_evidence"] = (
        (
            robust_component
            + percentile_component
        )
        * sample_weight
    )

    # Retain the old intuitive quantity for diagnostics.
    result["tightness_score"] = (
        1
        / result["relative_spread"].clip(lower=0.0001)
    )

    return result.sort_values(
        [
            "price_evidence",
            "tightness_percentile",
            "bid_count",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 3. WINNER CONCENTRATION
# ============================================================

def winner_concentration(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
    min_awards: int = 8,
) -> pd.DataFrame:
    """
    Measure concentration of awards inside contextual procurement
    peer groups.

    Peer group = category + region.

    Outputs both winner share and HHI. Concentration alone is NOT
    treated as evidence of misconduct because specialized markets
    can legitimately be concentrated.
    """

    awards = tenders[
        [
            "tender_id",
            "category",
            "region",
            "winner_id",
        ]
    ].dropna(subset=["winner_id"])

    rows = []

    for (category, region), group in awards.groupby(
        ["category", "region"]
    ):
        total = len(group)

        if total < min_awards:
            continue

        counts = group["winner_id"].value_counts()

        shares = counts / total

        top_vendor = counts.index[0]
        top_wins = int(counts.iloc[0])
        top_share = float(shares.iloc[0])

        hhi = float((shares ** 2).sum())

        unique_winners = len(counts)

        rows.append(
            {
                "category": category,
                "region": region,
                "tender_count": total,
                "unique_winners": unique_winners,
                "top_vendor": top_vendor,
                "top_vendor_wins": top_wins,
                "top_vendor_share": round(top_share, 4),
                "hhi": round(hhi, 4),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["top_vendor_share", "hhi"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 4. WINNER ROTATION
# ============================================================

def winner_rotation(
    tenders: pd.DataFrame,
    min_sequence_length: int = 6,
) -> pd.DataFrame:
    """
    Detect repeated alternation between winners.

    A simple turn-taking pattern can be meaningful when it persists
    inside the same procurement market.

    This detector does not assume that rotation itself is improper.
    """

    df = tenders[
        [
            "tender_id",
            "category",
            "region",
            "publication_date",
            "winner_id",
        ]
    ].dropna(
        subset=["winner_id", "publication_date"]
    ).sort_values(
        ["category", "region", "publication_date"]
    )

    rows = []

    for (category, region), group in df.groupby(
        ["category", "region"]
    ):
        if len(group) < min_sequence_length:
            continue

        winners = group["winner_id"].tolist()
        dates = group["publication_date"].tolist()

        pair_counts: dict[tuple[str, str], int] = {}
        transitions = 0

        for previous, current in zip(winners, winners[1:]):
            if previous == current:
                continue

            transitions += 1

            pair = tuple(sorted((previous, current)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            continue

        top_pair, top_transitions = max(
            pair_counts.items(),
            key=lambda item: item[1],
        )

        rotation_rate = top_transitions / max(transitions, 1)

        rows.append(
            {
                "category": category,
                "region": region,
                "tender_count": len(group),
                "distinct_winners": group["winner_id"].nunique(),
                "transition_count": transitions,
                "vendor_a": top_pair[0],
                "vendor_b": top_pair[1],
                "pair_transitions": top_transitions,
                "pair_transition_share": round(rotation_rate, 4),
                "period_start": dates[0],
                "period_end": dates[-1],
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["pair_transitions", "pair_transition_share"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 5. TEMPORAL BEHAVIOR
# ============================================================

def temporal_behavior(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
    min_total_participation: int = 10,
) -> pd.DataFrame:
    """
    Detect meaningful changes in vendor behavior over time.

    The vendor timeline is divided into early and late periods. The
    detector measures changes in participation and win rate while
    applying a sample-size confidence factor.

    Small samples therefore cannot produce the same evidence strength
    as sustained behavioral changes.
    """

    tender_dates = tenders[
        [
            "tender_id",
            "publication_date",
            "category",
            "region",
        ]
    ]

    participation = (
        bids[
            ["tender_id", "vendor_id"]
        ]
        .drop_duplicates()
        .merge(
            tender_dates,
            on="tender_id",
            how="left",
        )
        .dropna(subset=["publication_date"])
    )

    winner_lookup = tenders[
        ["tender_id", "winner_id"]
    ]

    participation = participation.merge(
        winner_lookup,
        on="tender_id",
        how="left",
    )

    rows = []

    for vendor_id, group in participation.groupby("vendor_id"):
        if len(group) < min_total_participation:
            continue

        midpoint = group["publication_date"].median()

        early = group[group["publication_date"] <= midpoint]
        late = group[group["publication_date"] > midpoint]

        if len(early) < 4 or len(late) < 4:
            continue

        early_wins = int(
            (early["winner_id"] == vendor_id).sum()
        )
        late_wins = int(
            (late["winner_id"] == vendor_id).sum()
        )

        early_n = len(early)
        late_n = len(late)

        early_win_rate = early_wins / early_n
        late_win_rate = late_wins / late_n

        win_rate_shift = (
            late_win_rate - early_win_rate
        )

        early_share = early_n / len(group)
        late_share = late_n / len(group)

        participation_shift = (
            late_share - early_share
        )

        raw_shift = (
            abs(win_rate_shift)
            + abs(participation_shift)
        )

        # Approximate uncertainty for the difference between two
        # binomial proportions. This is used as an evidence-strength
        # modifier rather than as a formal causal test.
        pooled_rate = (
            (early_wins + late_wins)
            / max(early_n + late_n, 1)
        )

        variance = pooled_rate * (1 - pooled_rate)

        standard_error = (
            (
                variance / max(early_n, 1)
                + variance / max(late_n, 1)
            )
            ** 0.5
        )

        standardized_shift = (
            abs(win_rate_shift)
            / max(standard_error, 0.05)
        )

        # Effective sample-size weight.
        # Approaches 1 as both periods become well populated.
        sample_weight = min(
            1.0,
            (
                min(early_n, late_n) / 20.0
            ) ** 0.5,
        )

        behavioral_shift = (
            raw_shift
            * sample_weight
        )

        rows.append(
            {
                "vendor_id": vendor_id,
                "total_tenders": len(group),
                "early_tenders": early_n,
                "late_tenders": late_n,
                "early_wins": early_wins,
                "late_wins": late_wins,
                "early_win_rate": round(
                    early_win_rate,
                    4,
                ),
                "late_win_rate": round(
                    late_win_rate,
                    4,
                ),
                "win_rate_shift": round(
                    win_rate_shift,
                    4,
                ),
                "early_participation_share": round(
                    early_share,
                    4,
                ),
                "late_participation_share": round(
                    late_share,
                    4,
                ),
                "participation_shift": round(
                    participation_shift,
                    4,
                ),
                "standardized_win_shift": round(
                    standardized_shift,
                    4,
                ),
                "sample_weight": round(
                    sample_weight,
                    4,
                ),
                "behavioral_shift": round(
                    behavioral_shift,
                    4,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "behavioral_shift",
            "standardized_win_shift",
            "total_tenders",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 6. GEOGRAPHIC BEHAVIOR
# ============================================================

def geographic_behavior(
    tenders: pd.DataFrame,
    bids: pd.DataFrame,
    min_participation: int = 8,
) -> pd.DataFrame:
    """
    Compare vendor participation and winning behavior across regions.

    This identifies unusually strong geographic allocation patterns.
    Geographic specialization is deliberately represented as evidence,
    not automatically treated as suspicious.
    """

    tender_context = tenders[
        [
            "tender_id",
            "region",
            "category",
            "winner_id",
        ]
    ]

    participation = (
        bids[
            ["tender_id", "vendor_id"]
        ]
        .drop_duplicates()
        .merge(
            tender_context,
            on="tender_id",
            how="left",
        )
    )

    rows = []

    for vendor_id, group in participation.groupby("vendor_id"):
        if len(group) < min_participation:
            continue

        region_counts = group["region"].value_counts()

        dominant_region = region_counts.index[0]
        dominant_participation = int(region_counts.iloc[0])
        dominant_share = dominant_participation / len(group)

        wins = group[
            group["winner_id"] == vendor_id
        ]

        if len(wins) == 0:
            dominant_win_share = 0.0
        else:
            dominant_win_share = (
                wins["region"]
                .value_counts(normalize=True)
                .get(dominant_region, 0.0)
            )

        cross_region = group["region"].nunique()

        rows.append(
            {
                "vendor_id": vendor_id,
                "total_participation": len(group),
                "regions_active": cross_region,
                "dominant_region": dominant_region,
                "dominant_region_tenders": dominant_participation,
                "dominant_region_share": round(
                    dominant_share,
                    4,
                ),
                "dominant_region_win_share": round(
                    dominant_win_share,
                    4,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "dominant_region_share",
            "dominant_region_win_share",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 7. CONTEXTUAL CO-OCCURRENCE
# ============================================================

def contextualize_cooccurrence(
    cooccurrence: pd.DataFrame,
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add procurement context to repeated bidder relationships.

    Repeated co-bidding concentrated inside one narrow category/region
    is discounted relative to a relationship spanning unrelated
    procurement environments.
    """

    if cooccurrence.empty:
        return pd.DataFrame()

    tender_context = tenders[
        [
            "tender_id",
            "category",
            "region",
        ]
    ]

    bid_context = bids[
        [
            "tender_id",
            "vendor_id",
        ]
    ].merge(
        tender_context,
        on="tender_id",
        how="left",
    ).drop_duplicates()

    rows = []

    for _, relationship in cooccurrence.iterrows():
        a = relationship["vendor_a"]
        b = relationship["vendor_b"]

        shared = bid_context[
            bid_context["vendor_id"].isin([a, b])
        ]

        shared_tenders = (
            shared.groupby("tender_id")["vendor_id"]
            .nunique()
        )

        shared_ids = set(
            shared_tenders[
                shared_tenders >= 2
            ].index
        )

        if not shared_ids:
            continue

        shared_context = bid_context[
            bid_context["tender_id"].isin(shared_ids)
        ]

        category_counts = (
            shared_context["category"]
            .value_counts(normalize=True)
        )

        region_counts = (
            shared_context["region"]
            .value_counts(normalize=True)
        )

        dominant_category = category_counts.index[0]
        dominant_category_share = category_counts.iloc[0]

        dominant_region = region_counts.index[0]
        dominant_region_share = region_counts.iloc[0]

        context_concentration = (
            dominant_category_share * 0.6
            + dominant_region_share * 0.4
        )

        contextual_adjustment = 1.0 - (
            0.45 * context_concentration
        )

        contextual_enrichment = (
            relationship["enrichment_ratio"]
            * contextual_adjustment
        )

        rows.append(
            {
                **relationship.to_dict(),
                "dominant_category": dominant_category,
                "category_share": round(
                    dominant_category_share,
                    3,
                ),
                "dominant_region": dominant_region,
                "region_share": round(
                    dominant_region_share,
                    3,
                ),
                "context_concentration": round(
                    context_concentration,
                    3,
                ),
                "contextual_adjustment": round(
                    contextual_adjustment,
                    3,
                ),
                "contextual_enrichment": round(
                    contextual_enrichment,
                    3,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "contextual_enrichment",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# UNIFIED DETECTOR PIPELINE
# ============================================================

def run_all_detectors(
    data_dir: str | Path = DATA_DIR,
):
    """
    Execute every evidence detector and persist normalized outputs.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tenders, bids, vendors = load_data(data_dir)

    print("=" * 72)
    print("VERITAS EVIDENCE ENGINE")
    print("=" * 72)
    print(
        f"Tenders: {len(tenders):,} | "
        f"Bids: {len(bids):,} | "
        f"Vendors: {len(vendors):,}"
    )
    print()

    # --------------------------------------------------------
    print("[1/7] Bidder co-occurrence")
    cooccurrence = bidder_cooccurrence(bids)
    cooccurrence.to_csv(
        data_dir / "cooccurrence.csv",
        index=False,
    )
    print(f"       {len(cooccurrence):,} relationships")

    # --------------------------------------------------------
    print("[2/7] Bid-price behavior")
    spread = bid_spread_anomalies(bids, tenders)
    spread.to_csv(
        data_dir / "bid_spread.csv",
        index=False,
    )
    print(f"       {len(spread):,} tenders analyzed")

    # --------------------------------------------------------
    print("[3/7] Winner concentration")
    concentration = winner_concentration(tenders, bids)
    concentration.to_csv(
        data_dir / "winner_concentration.csv",
        index=False,
    )
    print(f"       {len(concentration):,} peer groups")

    # --------------------------------------------------------
    print("[4/7] Winner rotation")
    rotation = winner_rotation(tenders)
    rotation.to_csv(
        data_dir / "winner_rotation.csv",
        index=False,
    )
    print(f"       {len(rotation):,} peer groups analyzed")

    # --------------------------------------------------------
    print("[5/7] Temporal behavior")
    temporal = temporal_behavior(tenders, bids)
    temporal.to_csv(
        data_dir / "temporal_behavior.csv",
        index=False,
    )
    print(f"       {len(temporal):,} vendors analyzed")

    # --------------------------------------------------------
    print("[6/7] Geographic behavior")
    geographic = geographic_behavior(tenders, bids)
    geographic.to_csv(
        data_dir / "geographic_behavior.csv",
        index=False,
    )
    print(f"       {len(geographic):,} vendors analyzed")

    # --------------------------------------------------------
    print("[7/7] Contextualizing bidder relationships")
    contextual = contextualize_cooccurrence(
        cooccurrence,
        bids,
        tenders,
    )
    contextual.to_csv(
        data_dir / "contextual_cooccurrence.csv",
        index=False,
    )
    print(f"       {len(contextual):,} relationships contextualized")

    print()
    print("=" * 72)
    print("EVIDENCE ENGINE COMPLETE")
    print("=" * 72)

    outputs = [
        "cooccurrence.csv",
        "bid_spread.csv",
        "winner_concentration.csv",
        "winner_rotation.csv",
        "temporal_behavior.csv",
        "geographic_behavior.csv",
        "contextual_cooccurrence.csv",
    ]

    for filename in outputs:
        path = data_dir / filename

        if not path.exists():
            raise RuntimeError(
                f"Expected detector output was not created: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Detector output is empty: {path}"
            )

        print(f"[OK] {path}")

    return {
        "cooccurrence": cooccurrence,
        "bid_spread": spread,
        "winner_concentration": concentration,
        "winner_rotation": rotation,
        "temporal_behavior": temporal,
        "geographic_behavior": geographic,
        "contextual_cooccurrence": contextual,
    }


# ============================================================
# SELF-TEST
# ============================================================

def self_test():
    """
    Structural smoke test for the detector layer.
    """

    data_dir = DATA_DIR

    required = [
        "tenders.csv",
        "bids.csv",
        "vendors.csv",
    ]

    for filename in required:
        path = data_dir / filename
        assert path.exists(), f"Missing input dataset: {path}"

    tenders, bids, vendors = load_data(data_dir)

    assert len(tenders) > 0
    assert len(bids) > 0
    assert len(vendors) > 0

    required_tender_columns = {
        "tender_id",
        "category",
        "region",
        "estimated_value",
        "winner_id",
        "publication_date",
    }

    required_bid_columns = {
        "bid_id",
        "tender_id",
        "vendor_id",
        "bid_amount",
    }

    assert required_tender_columns.issubset(tenders.columns)
    assert required_bid_columns.issubset(bids.columns)

    results = run_all_detectors(data_dir)

    assert not results["cooccurrence"].empty
    assert not results["bid_spread"].empty
    assert not results["winner_concentration"].empty
    assert not results["temporal_behavior"].empty
    assert not results["geographic_behavior"].empty

    print()
    print("[PASS] detector self-test completed")


if __name__ == "__main__":
    self_test()

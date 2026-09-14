from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import pandas as pd


def load_data(data_dir: str | Path = "data/generated"):
    data_dir = Path(data_dir)

    tenders = pd.read_csv(data_dir / "tenders.csv")
    bids = pd.read_csv(data_dir / "bids.csv")
    vendors = pd.read_csv(data_dir / "vendors.csv")

    return tenders, bids, vendors


def bidder_cooccurrence(
    bids: pd.DataFrame,
    min_shared_tenders: int = 3,
) -> pd.DataFrame:
    """
    Detect unusually strong repeated co-bidding between vendor pairs.

    Expected overlap is based on each vendor's participation frequency
    and the total number of tenders.

    No planted scenario information is used.
    """

    tender_count = bids["tender_id"].nunique()

    participation = (
        bids[["tender_id", "vendor_id"]]
        .drop_duplicates()
        .groupby("vendor_id")["tender_id"]
        .nunique()
    )

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

        # Hypergeometric probability:
        # probability of observing at least `observed` overlap
        # if the two vendors participated independently.
        #
        # We calculate the tail explicitly because this is more
        # interpretable than using a simple ratio alone.
        max_overlap = min(n_a, n_b)

        probability = 0.0

        denominator = comb(tender_count, n_b)

        for k in range(observed, max_overlap + 1):
            if (
                n_a <= tender_count
                and n_b <= tender_count
                and tender_count - n_a >= n_b - k
            ):
                numerator = (
                    comb(n_a, k)
                    * comb(tender_count - n_a, n_b - k)
                )
                probability += numerator / denominator

        rows.append(
            {
                "vendor_a": vendor_a,
                "vendor_b": vendor_b,
                "observed_shared_tenders": observed,
                "vendor_a_tenders": n_a,
                "vendor_b_tenders": n_b,
                "expected_shared_tenders": round(expected, 3),
                "enrichment_ratio": round(enrichment, 3),
                "p_value": min(probability, 1.0),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["surprise_score"] = (
        -result["p_value"].clip(lower=1e-300).apply(__import__("math").log10)
    )

    result = result.sort_values(
        [
            "enrichment_ratio",
            "observed_shared_tenders",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return result


def main():
    tenders, bids, vendors = load_data()

    print("VERITAS — BIDDER CO-OCCURRENCE DETECTOR")
    print("=" * 55)
    print(f"Tenders : {len(tenders)}")
    print(f"Bids    : {len(bids)}")
    print(f"Vendors : {len(vendors)}")
    print()

    results = bidder_cooccurrence(bids)

    if results.empty:
        print("No repeated bidder relationships detected.")
        return

    print("TOP REPEATED BIDDER RELATIONSHIPS")
    print("=" * 55)

    print(
        results[
            [
                "vendor_a",
                "vendor_b",
                "observed_shared_tenders",
                "expected_shared_tenders",
                "enrichment_ratio",
                "p_value",
            ]
        ].head(20).to_string(index=False)
    )

    output = Path("data/generated/cooccurrence.csv")
    results.to_csv(output, index=False)

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()


def bid_spread_anomalies(
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
    min_bids: int = 3,
) -> pd.DataFrame:
    """
    Detect tenders where competing bids are unusually tightly clustered.

    Raw prices are normalized by the tender estimate so that a
    ₹1 crore tender and a ₹100 crore tender are comparable.
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
        df["bid_amount"] / df["estimated_value"]
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

        # Relative spread around the median.
        relative_spread = spread / max(abs(median), 1e-9)

        # Coefficient of variation.
        cv = ratios.std(ddof=0) / max(abs(ratios.mean()), 1e-9)

        rows.append(
            {
                "tender_id": tender_id,
                "category": group["category"].iloc[0],
                "region": group["region"].iloc[0],
                "bid_count": len(group),
                "median_bid_ratio": round(median, 5),
                "mad": round(mad, 5),
                "relative_spread": round(relative_spread, 5),
                "coefficient_of_variation": round(cv, 5),
                "minimum_bid_ratio": round(minimum, 5),
                "maximum_bid_ratio": round(maximum, 5),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # Lower spread = tighter bidding.
    # Convert to a ranking score where higher means more unusual.
    result["tightness_score"] = (
        1 / result["relative_spread"].clip(lower=0.0001)
    )

    result = result.sort_values(
        "tightness_score",
        ascending=False,
    ).reset_index(drop=True)

    return result


def main():
    tenders, bids, vendors = load_data()

    print("VERITAS — BIDDER CO-OCCURRENCE DETECTOR")
    print("=" * 55)
    print(f"Tenders : {len(tenders)}")
    print(f"Bids    : {len(bids)}")
    print(f"Vendors : {len(vendors)}")
    print()

    results = bidder_cooccurrence(bids)

    if results.empty:
        print("No repeated bidder relationships detected.")
    else:
        print("TOP REPEATED BIDDER RELATIONSHIPS")
        print("=" * 55)

        print(
            results[
                [
                    "vendor_a",
                    "vendor_b",
                    "observed_shared_tenders",
                    "expected_shared_tenders",
                    "enrichment_ratio",
                    "p_value",
                ]
            ].head(20).to_string(index=False)
        )

        results.to_csv(
            "data/generated/cooccurrence.csv",
            index=False,
        )

    print()
    print("TOP TENDERS BY BID TIGHTNESS")
    print("=" * 55)

    spread = bid_spread_anomalies(bids, tenders)

    print(
        spread[
            [
                "tender_id",
                "category",
                "bid_count",
                "median_bid_ratio",
                "relative_spread",
                "coefficient_of_variation",
                "minimum_bid_ratio",
                "maximum_bid_ratio",
            ]
        ].head(20).to_string(index=False)
    )

    spread.to_csv(
        "data/generated/bid_spread.csv",
        index=False,
    )

    print()
    print("Saved:")
    print("  data/generated/cooccurrence.csv")
    print("  data/generated/bid_spread.csv")


if __name__ == "__main__":
    main()


def contextualize_cooccurrence(
    cooccurrence: pd.DataFrame,
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add procurement context to repeated bidder relationships.

    A strong co-occurrence signal is not automatically suspicious.
    This function determines whether the relationship is concentrated
    inside a narrow procurement environment.
    """

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

        # Tenders where both vendors participated.
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

        # Contextual concentration:
        #
        # If the repeated relationship only exists inside one
        # specialized category/region, that is weaker evidence than
        # the same relationship appearing across unrelated markets.
        context_concentration = (
            dominant_category_share * 0.6
            + dominant_region_share * 0.4
        )

        contextual_adjustment = 1.0 - (
            0.45 * context_concentration
        )

        contextual_score = (
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
                    contextual_score,
                    3,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_context_results(
    cooccurrence: pd.DataFrame,
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
):
    contextual = contextualize_cooccurrence(
        cooccurrence,
        bids,
        tenders,
    )

    if contextual.empty:
        print("No contextual relationships available.")
        return

    contextual = contextual.sort_values(
        "contextual_enrichment",
        ascending=False,
    )

    print()
    print("CONTEXTUALIZED BIDDER RELATIONSHIPS")
    print("=" * 80)

    print(
        contextual[
            [
                "vendor_a",
                "vendor_b",
                "observed_shared_tenders",
                "expected_shared_tenders",
                "enrichment_ratio",
                "dominant_category",
                "category_share",
                "dominant_region",
                "region_share",
                "contextual_adjustment",
                "contextual_enrichment",
            ]
        ].head(20).to_string(index=False)
    )

    contextual.to_csv(
        "data/generated/contextual_cooccurrence.csv",
        index=False,
    )

    print()
    print("Saved: data/generated/contextual_cooccurrence.csv")


# Run the contextual analysis after the existing detector.
if __name__ == "__main__":
    tenders, bids, vendors = load_data()

    cooccurrence = bidder_cooccurrence(bids)

    print_context_results(
        cooccurrence,
        bids,
        tenders,
    )

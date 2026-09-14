
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"

BIDS = DATA / "bids.csv"
TENDERS = DATA / "tenders.csv"
OUTPUT = DATA / "economic_evidence.csv"


EPS = 1e-9


def robust_scale(values: pd.Series) -> float:
    x = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(x) < 3:
        return 1.0

    median = float(x.median())
    mad = float(
        np.median(
            np.abs(x.to_numpy() - median)
        )
    )

    iqr = float(
        x.quantile(0.75)
        - x.quantile(0.25)
    )

    # Multiple robust scale estimators protect us against a degenerate
    # distribution where MAD happens to be zero.
    scale = max(
        1.4826 * mad,
        iqr / 1.349,
        0.01,
    )

    return scale


def load_data():
    bids = pd.read_csv(BIDS)
    tenders = pd.read_csv(TENDERS)

    bid_required = {
        "bid_id",
        "tender_id",
        "vendor_id",
        "bid_amount",
    }

    tender_required = {
        "tender_id",
        "category",
        "region",
        "procurement_method",
        "estimated_value",
    }

    missing_bids = bid_required - set(bids.columns)
    missing_tenders = tender_required - set(tenders.columns)

    if missing_bids:
        raise RuntimeError(
            f"Missing bid columns: {sorted(missing_bids)}"
        )

    if missing_tenders:
        raise RuntimeError(
            f"Missing tender columns: {sorted(missing_tenders)}"
        )

    bids["bid_amount"] = pd.to_numeric(
        bids["bid_amount"],
        errors="coerce",
    )

    tenders["estimated_value"] = pd.to_numeric(
        tenders["estimated_value"],
        errors="coerce",
    )

    bids = bids.dropna(
        subset=[
            "tender_id",
            "vendor_id",
            "bid_amount",
        ]
    )

    tenders = tenders.dropna(
        subset=[
            "tender_id",
            "estimated_value",
        ]
    )

    return bids, tenders


def build_tender_economics(
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
) -> pd.DataFrame:

    b = bids.merge(
        tenders[
            [
                "tender_id",
                "category",
                "region",
                "procurement_method",
                "estimated_value",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    b = b[
        b["bid_amount"] > 0
    ].copy()

    grouped = (
        b.groupby(
            "tender_id",
            as_index=False,
        )
        .agg(
            bid_count=(
                "bid_amount",
                "size",
            ),
            min_bid=(
                "bid_amount",
                "min",
            ),
            median_bid=(
                "bid_amount",
                "median",
            ),
            max_bid=(
                "bid_amount",
                "max",
            ),
            mean_bid=(
                "bid_amount",
                "mean",
            ),
            std_bid=(
                "bid_amount",
                "std",
            ),
        )
    )

    grouped["std_bid"] = grouped[
        "std_bid"
    ].fillna(0.0)

    grouped["relative_spread"] = (
        (
            grouped["max_bid"]
            - grouped["min_bid"]
        )
        / grouped["median_bid"].clip(
            lower=EPS
        )
    )

    grouped["coefficient_variation"] = (
        grouped["std_bid"]
        / grouped["mean_bid"].clip(
            lower=EPS
        )
    )

    grouped["bid_to_estimate"] = (
        grouped["median_bid"]
        / grouped["estimated_value"]
        if "estimated_value" in grouped
        else np.nan
    )

    meta = tenders[
        [
            "tender_id",
            "category",
            "region",
            "procurement_method",
            "estimated_value",
        ]
    ].drop_duplicates(
        "tender_id"
    )

    grouped = grouped.merge(
        meta,
        on="tender_id",
        how="left",
    )

    grouped["median_to_estimate"] = (
        grouped["median_bid"]
        / grouped["estimated_value"].clip(
            lower=EPS
        )
    )

    return grouped


def peer_residuals(
    tender_economics: pd.DataFrame,
) -> pd.DataFrame:

    df = tender_economics.copy()

    # Procurement context is part of the expected economic baseline.
    df["peer_key"] = (
        df["category"].astype(str)
        + "|"
        + df["region"].astype(str)
        + "|"
        + df["procurement_method"].astype(str)
    )

    peer = (
        df.groupby("peer_key")
        .agg(
            peer_count=(
                "tender_id",
                "size",
            ),
            expected_spread=(
                "relative_spread",
                "median",
            ),
            spread_scale=(
                "relative_spread",
                robust_scale,
            ),
            expected_cv=(
                "coefficient_variation",
                "median",
            ),
            cv_scale=(
                "coefficient_variation",
                robust_scale,
            ),
            expected_price_ratio=(
                "median_to_estimate",
                "median",
            ),
            price_ratio_scale=(
                "median_to_estimate",
                robust_scale,
            ),
        )
        .reset_index()
    )

    df = df.merge(
        peer,
        on="peer_key",
        how="left",
    )

    df["spread_residual"] = (
        df["expected_spread"]
        - df["relative_spread"]
    )

    df["spread_z"] = (
        df["spread_residual"]
        / df["spread_scale"].clip(
            lower=0.01
        )
    ).clip(
        -8.0,
        8.0,
    )

    df["cv_residual"] = (
        df["expected_cv"]
        - df["coefficient_variation"]
    )

    df["cv_z"] = (
        df["cv_residual"]
        / df["cv_scale"].clip(
            lower=0.01
        )
    ).clip(
        -8.0,
        8.0,
    )

    # A tender whose bids are unusually compressed relative to its
    # actual peer universe receives stronger economic evidence.
    df["compression_signal"] = (
        (
            df["spread_z"].clip(
                lower=0.0
            )
            + df["cv_z"].clip(
                lower=0.0
            )
        )
        / 2.0
    ).clip(
        0.0,
        8.0,
    )

    # Estimate-relative price movement is useful supporting evidence,
    # not a direct collusion indicator.
    df["price_ratio_z"] = (
        (
            df["median_to_estimate"]
            - df["expected_price_ratio"]
        )
        / df["price_ratio_scale"].clip(
            lower=0.01
        )
    ).clip(
        -8.0,
        8.0,
    )

    # Confidence increases with bidder count but saturates quickly.
    #
    # 2 bids should carry evidence, but much less confidence than
    # 10+ bids.
    df["sample_confidence"] = (
        1.0
        - np.exp(
            -0.28
            * np.maximum(
                df["bid_count"] - 1,
                0,
            )
        )
    ).clip(
        0.0,
        1.0,
    )

    # Require a meaningful peer universe before treating a residual
    # as strongly contextualized.
    df["peer_confidence"] = (
        1.0
        - np.exp(
            -0.18
            * np.maximum(
                df["peer_count"] - 3,
                0,
            )
        )
    ).clip(
        0.0,
        1.0,
    )

    df["economic_strength"] = (
        (
            0.82
            * (
                1.0
                - np.exp(
                    -0.55
                    * df["compression_signal"]
                )
            )
            + 0.18
            * (
                1.0
                - np.exp(
                    -0.35
                    * df["price_ratio_z"].clip(
                        lower=0.0
                    )
                )
            )
        )
        * df["sample_confidence"]
        * (
            0.55
            + 0.45
            * df["peer_confidence"]
        )
    ).clip(
        0.0,
        1.0,
    )

    # Tail strength is kept separately for later convergence modeling.
    df["economic_tail_strength"] = (
        1.0
        - np.exp(
            -0.65
            * df["compression_signal"]
        )
    ).clip(
        0.0,
        1.0,
    )

    return df


def persistence_by_market(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # Economic anomalies become considerably more interesting when the
    # same vendor repeatedly appears in unusually compressed tenders.
    #
    # This stage intentionally does NOT require the vendor to be the
    # winner. That prevents circular reasoning.
    b, tenders = load_data()

    joined = b.merge(
        df[
            [
                "tender_id",
                "economic_strength",
                "economic_tail_strength",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    vendor = (
        joined.groupby("vendor_id")
        .agg(
            tenders_observed=(
                "tender_id",
                "nunique",
            ),
            mean_economic_strength=(
                "economic_strength",
                "mean",
            ),
            high_economic_events=(
                "economic_strength",
                lambda x: int(
                    (x >= 0.55).sum()
                ),
            ),
            max_economic_strength=(
                "economic_strength",
                "max",
            ),
        )
        .reset_index()
    )

    vendor["economic_persistence"] = (
        1.0
        - np.exp(
            -0.35
            * vendor["high_economic_events"]
        )
    ).clip(
        0.0,
        1.0,
    )

    return vendor


def build_evidence(
    df: pd.DataFrame,
    vendor_persistence: pd.DataFrame,
) -> pd.DataFrame:

    out = df.merge(
        vendor_persistence[
            [
                "vendor_id",
                "economic_persistence",
            ]
        ],
        how="cross",
    )

    # The cross join above is intentionally replaced below by a proper
    # tender-vendor mapping. Keeping this explicit prevents accidentally
    # attributing tender evidence to unrelated vendors.
    b, _ = load_data()

    out = b[
        [
            "tender_id",
            "vendor_id",
        ]
    ].drop_duplicates().merge(
        df[
            [
                "tender_id",
                "economic_strength",
                "economic_tail_strength",
                "bid_count",
                "peer_count",
                "spread_z",
                "cv_z",
                "compression_signal",
                "sample_confidence",
                "peer_confidence",
                "category",
                "region",
                "procurement_method",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    out = out.merge(
        vendor_persistence[
            [
                "vendor_id",
                "economic_persistence",
            ]
        ],
        on="vendor_id",
        how="left",
    )

    out["economic_persistence"] = (
        out["economic_persistence"]
        .fillna(0.0)
    )

    # Persistence strengthens repeated evidence, but never creates
    # evidence where the individual tender itself is normal.
    out["economic_evidence"] = (
        out["economic_strength"]
        * (
            0.75
            + 0.25
            * out["economic_persistence"]
        )
    ).clip(
        0.0,
        1.0,
    )

    out["evidence_family"] = "economic"

    out["explanation"] = (
        "bid dispersion is "
        "unusually compressed relative to "
        "comparable procurements"
    )

    out["evidence"] = out.apply(
        lambda r: (
            f"compression={r['compression_signal']:.2f}; "
            f"spread_z={r['spread_z']:.2f}; "
            f"bids={int(r['bid_count'])}; "
            f"peer_count={int(r['peer_count'])}; "
            f"persistence={r['economic_persistence']:.2f}"
        ),
        axis=1,
    )

    return out


def self_test(
    tender_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
):

    required_tender = {
        "tender_id",
        "relative_spread",
        "spread_z",
        "compression_signal",
        "economic_strength",
        "sample_confidence",
        "peer_confidence",
    }

    required_evidence = {
        "tender_id",
        "vendor_id",
        "economic_evidence",
        "evidence_family",
    }

    assert not (
        required_tender
        - set(tender_df.columns)
    )

    assert not (
        required_evidence
        - set(evidence_df.columns)
    )

    for col in [
        "spread_z",
        "cv_z",
        "compression_signal",
        "economic_strength",
        "economic_tail_strength",
        "sample_confidence",
        "peer_confidence",
    ]:
        values = tender_df[col].to_numpy(
            dtype=float
        )

        assert np.isfinite(values).all()

    assert (
        tender_df["economic_strength"]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        tender_df["sample_confidence"]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        evidence_df["economic_evidence"]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        evidence_df["vendor_id"]
        .nunique()
        >= 400
    )

    # More bidders must never reduce sample confidence.
    ordered = tender_df.sort_values(
        "bid_count"
    )

    confidence_by_count = (
        ordered.groupby(
            "bid_count"
        )["sample_confidence"]
        .median()
    )

    counts = confidence_by_count.index.to_list()

    for a, b in zip(
        counts,
        counts[1:],
    ):
        assert (
            confidence_by_count.loc[b]
            >= confidence_by_count.loc[a]
            - 1e-9
        )

    print(
        "[PASS] economic engine structural self-test"
    )


def scenario_audit(
    evidence: pd.DataFrame,
):

    scenarios = {
        "COVER_BIDDING": [
            "V0042",
            "V0043",
        ],
        "TIGHT_PRICE_CLUSTER": [
            "V0071",
            "V0072",
            "V0073",
            "V0074",
        ],
        "LEGITIMATE_SPECIALIZATION": [
            "V0010",
            "V0020",
        ],
        "REPEATED_NETWORK": [
            "V0001",
            "V0024",
            "V0091",
        ],
    }

    print()
    print("=" * 72)
    print("ECONOMIC SCENARIO AUDIT")
    print("=" * 72)

    maxima = {}

    for name, vendors in scenarios.items():
        subset = evidence[
            evidence["vendor_id"].isin(
                vendors
            )
        ]

        if subset.empty:
            print(
                f"{name:<28} no evidence"
            )
            continue

        grouped = (
            subset.groupby(
                "vendor_id"
            )["economic_evidence"]
            .max()
        )

        value = float(
            grouped.max()
        )

        vendor = str(
            grouped.idxmax()
        )

        maxima[name] = value

        print(
            f"{name:<28} "
            f"max={value:.3f} "
            f"vendor={vendor}"
        )

    if (
        "COVER_BIDDING" in maxima
        and "LEGITIMATE_SPECIALIZATION"
        in maxima
    ):
        if (
            maxima["COVER_BIDDING"]
            > maxima[
                "LEGITIMATE_SPECIALIZATION"
            ]
        ):
            print(
                "[PASS] cover-bidding economic "
                "evidence exceeds legitimate specialization"
            )
        else:
            print(
                "[WARN] economic separation "
                "needs further calibration"
            )


def main():

    print("=" * 72)
    print("VERITAS ECONOMIC ANOMALY ENGINE")
    print("=" * 72)

    bids, tenders = load_data()

    print(
        f"Bids: {len(bids):,} | "
        f"Tenders: {len(tenders):,}"
    )

    print()
    print(
        "[1/4] Building tender-level economic features"
    )

    tender_df = build_tender_economics(
        bids,
        tenders,
    )

    print(
        f"       {len(tender_df):,} tenders"
    )

    print(
        "[2/4] Conditioning on procurement peers"
    )

    tender_df = peer_residuals(
        tender_df
    )

    print(
        "[3/4] Measuring repeated vendor exposure"
    )

    vendor_persistence = (
        persistence_by_market(
            tender_df
        )
    )

    print(
        f"       {len(vendor_persistence):,} vendors"
    )

    print(
        "[4/4] Building economic evidence"
    )

    evidence = build_evidence(
        tender_df,
        vendor_persistence,
    )

    self_test(
        tender_df,
        evidence,
    )

    evidence.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"[OK] {OUTPUT}"
    )

    scenario_audit(
        evidence
    )

    print()
    print("=" * 72)
    print("TOP ECONOMIC EVIDENCE")
    print("=" * 72)

    cols = [
        "tender_id",
        "vendor_id",
        "economic_evidence",
        "compression_signal",
        "spread_z",
        "bid_count",
        "peer_count",
        "economic_persistence",
        "category",
        "region",
    ]

    print(
        evidence.sort_values(
            "economic_evidence",
            ascending=False,
        )[cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

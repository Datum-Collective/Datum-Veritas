
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"

INPUT = DATA / "vendor_regimes.csv"
OUTPUT = DATA / "vendor_regime_risk.csv"


def load_regimes() -> pd.DataFrame:
    df = pd.read_csv(INPUT)

    required = {
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
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing regime columns: {sorted(missing)}"
        )

    numeric = [
        "regime_signal",
        "mahalanobis_distance",
        "cusum_score",
        "persistence_score",
        "z_participation_intensity",
        "z_win_rate",
        "z_bid_position",
        "z_market_breadth",
        "z_network_exposure",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)

    return df


def bounded_positive_z(x: pd.Series, scale: float = 3.0) -> pd.Series:
    """
    Convert a positive directional z-score into [0, 1].

    We deliberately do NOT treat every behavioral deviation as risky.
    This function only measures magnitude in the positive direction.
    """
    return (
        x.clip(lower=0.0)
        / (x.clip(lower=0.0) + scale)
    ).clip(0.0, 1.0)


def risk_relevance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ------------------------------------------------------------------
    # PROCUREMENT-RELEVANCE COMPONENTS
    #
    # The regime engine answers "what changed?"
    # This layer answers "how relevant is that change to procurement?"
    #
    # We intentionally preserve the underlying behavioral anomaly signal.
    # Relevance is a modifier, not a hard gate.
    # ------------------------------------------------------------------

    win = bounded_positive_z(
        out["z_win_rate"],
        scale=2.5,
    )

    participation = bounded_positive_z(
        out["z_participation_intensity"],
        scale=3.0,
    )

    network = bounded_positive_z(
        out["z_network_exposure"],
        scale=3.0,
    )

    bid_position = bounded_positive_z(
        -out["z_bid_position"],
        scale=3.0,
    )

    # A shrinking market footprint is not intrinsically suspicious.
    # It therefore contributes zero direct risk relevance.
    breadth = pd.Series(
        0.0,
        index=out.index,
        dtype=float,
    )

    # --------------------------------------------------------------
    # INTERACTION EVIDENCE
    #
    # The interesting events are combinations:
    #
    #   win ↑ + participation ↑
    #   win ↑ + network ↑
    #
    # Geometric means prevent either component from dominating when
    # the other is essentially absent.
    # --------------------------------------------------------------

    win_participation = np.sqrt(
        win * participation
    )

    win_network = np.sqrt(
        win * network
    )

    network_participation = np.sqrt(
        network * participation
    )

    interaction_strength = (
        0.45 * win_participation
        + 0.40 * win_network
        + 0.15 * network_participation
    ).clip(0.0, 1.0)

    # Direct procurement relevance.
    #
    # We keep the individual components available rather than hiding
    # everything inside one opaque score.
    direct_relevance = (
        0.55 * win
        + 0.12 * participation
        + 0.08 * network
        + 0.05 * bid_position
        + 0.20 * interaction_strength
    ).clip(0.0, 1.0)

    out["win_change_relevance"] = win
    out["participation_change_relevance"] = participation
    out["network_change_relevance"] = network
    out["bid_position_relevance"] = bid_position
    out["interaction_relevance"] = interaction_strength
    out["risk_relevance"] = direct_relevance

    # --------------------------------------------------------------
    # PERSISTENCE
    #
    # Persistent changes are more informative than isolated changes,
    # but persistence must never manufacture a signal by itself.
    # --------------------------------------------------------------

    persistence = out[
        "persistence_score"
    ].clip(0.0, 1.0)

    persistence_modifier = (
        0.85
        + 0.15 * persistence
    )

    # --------------------------------------------------------------
    # PRESERVE THE BEHAVIORAL SIGNAL
    #
    # Rather than:
    #
    #     regime_signal * relevance
    #
    # use relevance to modulate the signal around its original value.
    #
    # relevance=0   -> retain 55% of anomaly
    # relevance=.5  -> retain 82.5%
    # relevance=1   -> retain 100%
    #
    # This makes the subsystem useful as temporal evidence without
    # allowing it to independently accuse a vendor.
    # --------------------------------------------------------------

    relevance_modifier = (
        0.55
        + 0.45 * direct_relevance
    )

    out["risk_adjusted_regime_signal"] = (
        out["regime_signal"]
        * relevance_modifier
        * persistence_modifier
    ).clip(0.0, 1.0)

    def classify(row):
        signal = float(
            row["risk_adjusted_regime_signal"]
        )
        relevance = float(
            row["risk_relevance"]
        )

        if signal < 0.25:
            return "LOW_RELEVANCE"

        if relevance < 0.25:
            return "BEHAVIORAL_ONLY"

        if signal < 0.50:
            return "PROCUREMENT_RELEVANT"

        return "HIGH_RELEVANCE"

    out["risk_class"] = out.apply(
        classify,
        axis=1,
    )

    def explain(row):
        reasons = []

        if row["z_win_rate"] >= 1.5:
            reasons.append(
                f"win rate increased "
                f"({row['z_win_rate']:.1f} robust deviations)"
            )

        if row["z_participation_intensity"] >= 1.5:
            reasons.append(
                f"participation increased "
                f"({row['z_participation_intensity']:.1f} robust deviations)"
            )

        if row["z_network_exposure"] >= 1.5:
            reasons.append(
                f"network exposure increased "
                f"({row['z_network_exposure']:.1f} robust deviations)"
            )

        if row["z_bid_position"] <= -1.5:
            reasons.append(
                f"bid positioning shifted toward stronger positions "
                f"({abs(row['z_bid_position']):.1f} robust deviations)"
            )

        if (
            row["z_win_rate"] >= 1.0
            and row["z_participation_intensity"] >= 1.0
        ):
            reasons.append(
                "winning increased alongside participation"
            )

        if (
            row["z_win_rate"] >= 1.0
            and row["z_network_exposure"] >= 1.0
        ):
            reasons.append(
                "winning increased alongside network exposure"
            )

        if not reasons:
            return (
                "behavioral deviation detected; "
                "procurement relevance is weak"
            )

        return "; ".join(reasons)

    out["risk_explanation"] = out.apply(
        explain,
        axis=1,
    )

    columns = [
        "vendor_id",
        "regime_signal",
        "risk_relevance",
        "risk_adjusted_regime_signal",
        "risk_class",
        "win_change_relevance",
        "participation_change_relevance",
        "network_change_relevance",
        "bid_position_relevance",
        "interaction_relevance",
        "mahalanobis_distance",
        "cusum_score",
        "persistence_score",
        "z_participation_intensity",
        "z_win_rate",
        "z_bid_position",
        "z_market_breadth",
        "z_network_exposure",
        "risk_explanation",
    ]

    return out[columns].sort_values(
        [
            "risk_adjusted_regime_signal",
            "risk_relevance",
            "persistence_score",
        ],
        ascending=False,
    ).reset_index(drop=True)

def scenario_audit(df: pd.DataFrame):
    scenarios = {
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

    print()
    print("=" * 72)
    print("RISK-AWARE REGIME SCENARIO AUDIT")
    print("=" * 72)

    maxima = {}

    for name, vendors in scenarios.items():
        subset = df[
            df["vendor_id"].isin(vendors)
        ]

        if subset.empty:
            print(
                f"{name:<26} no matching vendors"
            )
            continue

        row = subset.loc[
            subset["risk_adjusted_regime_signal"].idxmax()
        ]

        maxima[name] = float(
            row["risk_adjusted_regime_signal"]
        )

        print(
            f"{name:<26} "
            f"risk={row['risk_adjusted_regime_signal']:.3f} "
            f"relevance={row['risk_relevance']:.3f} "
            f"vendor={row['vendor_id']} "
            f"class={row['risk_class']}"
        )

    if "LEGITIMATE_SPECIALIZATION" in maxima:
        legitimate = maxima[
            "LEGITIMATE_SPECIALIZATION"
        ]

        planted = [
            value
            for key, value in maxima.items()
            if key != "LEGITIMATE_SPECIALIZATION"
        ]

        if planted:
            strongest = max(planted)

            if strongest > legitimate:
                print()
                print(
                    "[PASS] at least one planted scenario "
                    "has stronger procurement relevance "
                    "than legitimate specialization"
                )
            else:
                print()
                print(
                    "[WARN] legitimate specialization remains "
                    "competitive with planted scenarios"
                )


def self_test(df: pd.DataFrame):
    required = {
        "vendor_id",
        "regime_signal",
        "risk_relevance",
        "risk_adjusted_regime_signal",
        "risk_class",
        "risk_explanation",
    }

    missing = required - set(df.columns)

    assert not missing, (
        f"Missing output columns: {sorted(missing)}"
    )

    assert df["vendor_id"].nunique() >= 400

    for col in [
        "risk_relevance",
        "risk_adjusted_regime_signal",
    ]:
        values = df[col].to_numpy(dtype=float)

        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()

    assert df["risk_class"].notna().all()
    assert df["risk_explanation"].notna().all()

    # Market breadth reduction must not automatically become
    # procurement risk.
    breadth_only = (
        (df["z_market_breadth"] < -1.5)
        & (df["z_win_rate"] < 1.0)
        & (df["z_network_exposure"] < 1.0)
        & (df["z_participation_intensity"] < 1.5)
    )

    if breadth_only.any():
        assert (
            df.loc[
                breadth_only,
                "risk_relevance",
            ] < 0.65
        ).all()

    # The modifier must preserve meaningful behavioral anomalies.
    assert (
        df["risk_adjusted_regime_signal"].max()
        > 0.30
    )

    # Risk relevance must remain independently inspectable.
    for col in [
        "win_change_relevance",
        "participation_change_relevance",
        "network_change_relevance",
        "bid_position_relevance",
        "interaction_relevance",
    ]:
        assert col in df.columns
        values = df[col].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()

    print(
        "[PASS] risk-aware regime self-test"
    )


def main():
    print("=" * 72)
    print("VERITAS RISK-AWARE REGIME ENGINE")
    print("=" * 72)

    df = load_regimes()

    print(
        f"Regime records: {len(df):,}"
    )

    result = risk_relevance(df)

    self_test(result)

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"[OK] {OUTPUT}"
    )

    scenario_audit(result)

    print()
    print("=" * 72)
    print("TOP PROCUREMENT-RELEVANT REGIME SHIFTS")
    print("=" * 72)

    display_cols = [
        "vendor_id",
        "risk_adjusted_regime_signal",
        "risk_relevance",
        "risk_class",
        "regime_signal",
        "persistence_score",
        "risk_explanation",
    ]

    print(
        result[display_cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

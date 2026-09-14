
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"

OUTPUT = DATA / "investigation_cases.csv"
EVIDENCE_OUTPUT = DATA / "converged_evidence.csv"


FAMILY_WEIGHTS = {
    "cooccurrence": 1.00,
    "economic": 0.95,
    "temporal": 0.90,
    "winner_rotation": 0.82,
    "geographic": 0.65,
    "winner_concentration": 0.45,
}

# Secondary observations from the SAME family are highly correlated.
#
# The strongest observation carries most of the information.
# Additional observations provide only a small corroborating increment.
SECONDARY_DECAY = 0.12

# A second genuinely distinct mechanism matters substantially more.
FAMILY_DIVERSITY = {
    1: 0.35,
    2: 0.68,
    3: 0.86,
    4: 0.94,
    5: 0.98,
    6: 1.00,
}

SINGLE_FAMILY_CAP = 58.0


def load_csv(name: str) -> pd.DataFrame:

    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required output: {path}"
        )

    return pd.read_csv(path)


def robust_unit(
    series: pd.Series,
    low: float,
    high: float,
) -> pd.Series:

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    if high <= low:
        return pd.Series(
            0.0,
            index=x.index,
        )

    return (
        (x - low)
        / (high - low)
    ).clip(
        0.0,
        1.0,
    )


def add_evidence(
    records: list[dict],
    vendor_id: str,
    family: str,
    strength: float,
    explanation: str,
    source_id: str,
    context: float = 1.0,
    persistence: float = 0.0,
    partner_vendor_id: str | None = None,
    enrichment: float | None = None,
):

    if family not in FAMILY_WEIGHTS:
        raise ValueError(
            f"Unknown evidence family: {family}"
        )

    strength = float(
        np.clip(
            strength,
            0.0,
            1.0,
        )
    )

    context = float(
        np.clip(
            context,
            0.0,
            1.0,
        )
    )

    persistence = float(
        np.clip(
            persistence,
            0.0,
            1.0,
        )
    )

    # Context and persistence modify the observation itself.
    adjusted = (
        strength
        * (
            0.55
            + 0.45 * context
        )
        * (
            0.80
            + 0.20 * persistence
        )
    )

    if adjusted <= 0:
        return

    records.append(
        {
            "vendor_id": str(vendor_id),
            "evidence_family": family,
            "source_id": str(source_id),
            "raw_strength": strength,
            "context_factor": context,
            "persistence": persistence,
            "adjusted_strength": float(
                np.clip(
                    adjusted,
                    0.0,
                    1.0,
                )
            ),
            "family_weight": FAMILY_WEIGHTS[family],
            "explanation": explanation,
            "partner_vendor_id": (
                str(partner_vendor_id)
                if partner_vendor_id is not None
                else None
            ),
            "enrichment": (
                float(enrichment)
                if enrichment is not None
                else None
            ),
        }
    )


def build_evidence() -> pd.DataFrame:

    records: list[dict] = []

    # ================================================================
    # NETWORK
    # ================================================================

    path = DATA / "behavioral_relationships.csv"

    if path.exists():

        df = pd.read_csv(path)

        required = {
            "vendor_a",
            "vendor_b",
            "context_adjusted_strength",
        }

        if required.issubset(df.columns):

            for idx, row in df.iterrows():

                # Use the raw relationship strength here.
                # The convergence layer applies the contextual discount
                # exactly once through add_evidence().
                strength = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "relationship_strength",
                                0.0,
                            ),
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                if strength < 0.15:
                    continue

                persistence = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "persistence",
                                0.0,
                            ),
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                enrichment = float(
                    pd.to_numeric(
                        row.get(
                            "conditional_enrichment",
                            0.0,
                        ),
                        errors="coerce",
                    )
                )

                relationship_strength = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "relationship_strength",
                                strength,
                            ),
                            errors="coerce",
                        ),
                        1e-6,
                        1.0,
                    )
                )

                context_adjusted_strength = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "context_adjusted_strength",
                                strength,
                            ),
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                # Actual contextual discount produced by the
                # behavioral relationship engine.
                context = float(
                    np.clip(
                        context_adjusted_strength
                        / relationship_strength,
                        0.0,
                        1.0,
                    )
                )

                explanation = (
                    f"conditional co-bidding "
                    f"{enrichment:.2f}x expected; "
                    f"persistence={persistence:.2f}"
                )

                for vendor, partner in (
                    (
                        row["vendor_a"],
                        row["vendor_b"],
                    ),
                    (
                        row["vendor_b"],
                        row["vendor_a"],
                    ),
                ):

                    add_evidence(
                        records,
                        vendor,
                        "cooccurrence",
                        strength,
                        explanation,
                        f"relationship:{idx}",
                        context,
                        persistence,
                        partner_vendor_id=str(partner),
                        enrichment=float(enrichment),
                    )

    # ================================================================
    # ECONOMIC
    # ================================================================

    path = DATA / "economic_evidence.csv"

    if path.exists():

        df = pd.read_csv(path)

        required = {
            "vendor_id",
            "tender_id",
            "economic_evidence",
        }

        if required.issubset(df.columns):

            # Collapse the same tender to ONE observation per vendor.
            event = (
                df.groupby(
                    [
                        "vendor_id",
                        "tender_id",
                    ],
                    as_index=False,
                )
                .agg(
                    strength=(
                        "economic_evidence",
                        "max",
                    ),
                    compression=(
                        "compression_signal",
                        "max",
                    ),
                    bid_count=(
                        "bid_count",
                        "max",
                    ),
                )
            )

            for idx, row in event.iterrows():

                strength = float(
                    np.clip(
                        row["strength"],
                        0.0,
                        1.0,
                    )
                )

                # Economic evidence is corroboration unless the
                # compression signal is genuinely strong.
                if strength < 0.22:
                    continue

                add_evidence(
                    records,
                    row["vendor_id"],
                    "economic",
                    strength,
                    (
                        f"unusually compressed bid "
                        f"dispersion; "
                        f"compression="
                        f"{float(row['compression']):.2f}; "
                        f"bids="
                        f"{int(row['bid_count'])}"
                    ),
                    f"economic:{row['tender_id']}",
                )

    # ================================================================
    # TEMPORAL
    # ================================================================

    path = DATA / "vendor_regime_risk.csv"

    if path.exists():

        df = pd.read_csv(path)

        required = {
            "vendor_id",
            "risk_adjusted_regime_signal",
        }

        if required.issubset(df.columns):

            for idx, row in df.iterrows():

                strength = float(
                    np.clip(
                        pd.to_numeric(
                            row[
                                "risk_adjusted_regime_signal"
                            ],
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                if strength < 0.15:
                    continue

                relevance = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "risk_relevance",
                                0.0,
                            ),
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                persistence = float(
                    np.clip(
                        pd.to_numeric(
                            row.get(
                                "persistence_score",
                                0.0,
                            ),
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                add_evidence(
                    records,
                    row["vendor_id"],
                    "temporal",
                    strength,
                    (
                        f"vendor behavioral regime "
                        f"shift; procurement relevance="
                        f"{relevance:.2f}; "
                        f"persistence="
                        f"{persistence:.2f}"
                    ),
                    f"regime:{idx}",
                    relevance,
                    persistence,
                )

    # ================================================================
    # WINNER ROTATION
    # ================================================================

    path = DATA / "winner_rotation.csv"

    if path.exists():

        df = pd.read_csv(path)

        if {
            "vendor_a",
            "vendor_b",
            "pair_transition_share",
        }.issubset(df.columns):

            for idx, row in df.iterrows():

                strength = float(
                    np.clip(
                        pd.to_numeric(
                            row[
                                "pair_transition_share"
                            ],
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                transition_count = float(
                    pd.to_numeric(
                        row.get(
                            "transition_count",
                            0,
                        ),
                        errors="coerce",
                    )
                )

                # Small-sample winner transitions are not
                # sufficient evidence of rotation.
                if (
                    strength < 0.18
                    or transition_count < 4
                ):
                    continue

                strength *= min(
                    transition_count / 8.0,
                    1.0,
                )

                if strength < 0.15:
                    continue

                # Transition share alone is not enough to call this
                # strong evidence. Require a meaningful relationship.
                if strength < 0.10:
                    continue

                for vendor in (
                    row["vendor_a"],
                    row["vendor_b"],
                ):

                    add_evidence(
                        records,
                        vendor,
                        "winner_rotation",
                        strength,
                        (
                            f"repeated winner "
                            f"transition share="
                            f"{strength:.2f}"
                        ),
                        f"rotation:{idx}",
                    )

    # ================================================================
    # GEOGRAPHIC
    # ================================================================

    path = DATA / "geographic_behavior.csv"

    if path.exists():

        df = pd.read_csv(path)

        if "vendor_id" in df.columns:

            for idx, row in df.iterrows():

                share = pd.to_numeric(
                    row.get(
                        "dominant_region_win_share",
                        np.nan,
                    ),
                    errors="coerce",
                )

                participation = pd.to_numeric(
                    row.get(
                        "dominant_region_share",
                        np.nan,
                    ),
                    errors="coerce",
                )

                if (
                    pd.isna(share)
                    or pd.isna(participation)
                ):
                    continue

                gap = abs(
                    float(share)
                    - float(participation)
                )

                strength = float(
                    robust_unit(
                        pd.Series([gap]),
                        0.25,
                        0.65,
                    ).iloc[0]
                )

                if strength < 0.30:
                    continue

                add_evidence(
                    records,
                    row["vendor_id"],
                    "geographic",
                    strength,
                    (
                        f"winning geography differs "
                        f"from participation geography "
                        f"by {gap:.2f}"
                    ),
                    f"geographic:{idx}",
                )

    # ================================================================
    # CONCENTRATION
    # ================================================================

    path = DATA / "winner_concentration.csv"

    if path.exists():

        df = pd.read_csv(path)

        if {
            "top_vendor",
            "top_vendor_share",
        }.issubset(df.columns):

            for idx, row in df.iterrows():

                share = float(
                    np.clip(
                        pd.to_numeric(
                            row[
                                "top_vendor_share"
                            ],
                            errors="coerce",
                        ),
                        0.0,
                        1.0,
                    )
                )

                # Concentration becomes meaningful only at materially
                # elevated levels. This avoids manufacturing evidence
                # from normal market concentration.
                tender_count = float(
                    pd.to_numeric(
                        row.get(
                            "tender_count",
                            0,
                        ),
                        errors="coerce",
                    )
                )

                unique_winners = float(
                    pd.to_numeric(
                        row.get(
                            "unique_winners",
                            0,
                        ),
                        errors="coerce",
                    )
                )

                winner_diversity = (
                    unique_winners / tender_count
                    if tender_count > 0
                    else 1.0
                )

                share_signal = float(
                    robust_unit(
                        pd.Series([share]),
                        0.25,
                        0.55,
                    ).iloc[0]
                )

                diversity_signal = float(
                    robust_unit(
                        pd.Series(
                            [1.0 - winner_diversity]
                        ),
                        0.50,
                        0.90,
                    ).iloc[0]
                )

                strength = (
                    share_signal
                    * diversity_signal
                )

                if strength < 0.25:
                    continue

                add_evidence(
                    records,
                    row["top_vendor"],
                    "winner_concentration",
                    strength,
                    (
                        f"elevated winner concentration="
                        f"{share:.2%} in peer group"
                    ),
                    f"concentration:{idx}",
                )

    evidence = pd.DataFrame(records)

    if evidence.empty:
        raise RuntimeError(
            "No evidence records were generated"
        )

    return evidence


def _family_score(
    group: pd.DataFrame,
) -> float:
    """
    Robust family aggregation.

    This is intentionally NOT a sum.

    The strongest observation carries 100% weight.
    Subsequent observations contribute only a small residual amount.

    This prevents:
        20 mediocre co-bidding relationships
    from becoming stronger than:
        1 extremely anomalous relationship.
    """

    values = (
        group[
            "adjusted_strength"
        ]
        .astype(float)
        .sort_values(
            ascending=False
        )
        .to_numpy()
    )

    if len(values) == 0:
        return 0.0

    score = float(values[0])

    multiplier = SECONDARY_DECAY

    for value in values[1:8]:

        score += (
            float(value)
            * multiplier
        )

        multiplier *= SECONDARY_DECAY

    # Saturating transform keeps the score bounded.
    return float(
        np.clip(
            1.0
            - math.exp(
                -score
            ),
            0.0,
            1.0,
        )
    )


def _family_context(
    group: pd.DataFrame,
) -> float:

    # Context is already applied to each evidence observation.
    # For the family-level diagnostic, use a strength-weighted mean
    # rather than letting one weak relationship suppress the entire
    # vendor's network evidence.
    strengths = pd.to_numeric(
        group["raw_strength"],
        errors="coerce",
    ).fillna(0.0)

    contexts = pd.to_numeric(
        group["context_factor"],
        errors="coerce",
    ).fillna(1.0)

    total = float(strengths.sum())

    if total <= 0:
        return 1.0

    value = float(
        (strengths * contexts).sum() / total
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _family_persistence(
    group: pd.DataFrame,
) -> float:

    return float(
        np.clip(
            group[
                "persistence"
            ]
            .astype(float)
            .max(),
            0.0,
            1.0,
        )
    )


def evidence_convergence(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine detector outputs into investigation cases.

    Important design principle:
    detector families are NOT assumed to be independent.

    Independent mechanisms:

        NETWORK
            cooccurrence + winner_rotation

        ECONOMIC
            economic

        TEMPORAL
            temporal

    Supporting/contextual mechanisms:

        geographic + winner_concentration

    This prevents several correlated detectors from manufacturing
    artificial convergence.
    """

    rows = []

    for vendor_id, vendor_group in evidence.groupby(
        "vendor_id"
    ):

        family_scores = {}
        family_contexts = {}
        family_persistence = {}
        family_sources = {}

        for family, family_group in vendor_group.groupby(
            "evidence_family"
        ):

            family_scores[family] = float(
                _family_score(
                    family_group
                )
            )

            family_contexts[family] = float(
                _family_context(
                    family_group
                )
            )

            family_persistence[family] = float(
                _family_persistence(
                    family_group
                )
            )

            family_sources[family] = (
                family_group.sort_values(
                    "adjusted_strength",
                    ascending=False,
                )[
                    "source_id"
                ]
                .head(8)
                .tolist()
            )

        families = sorted(
            family_scores
        )

        if not families:
            continue

        # ============================================================
        # MECHANISM GROUPS
        # ============================================================

        network_parts = [
            family_scores[f]
            for f in (
                "cooccurrence",
                "winner_rotation",
            )
            if f in family_scores
        ]

        network_score = (
            max(network_parts)
            if network_parts
            else 0.0
        )

        economic_score = float(
            family_scores.get(
                "economic",
                0.0,
            )
        )

        temporal_score = float(
            family_scores.get(
                "temporal",
                0.0,
            )
        )

        supporting_parts = [
            family_scores[f]
            for f in (
                "geographic",
                "winner_concentration",
            )
            if f in family_scores
        ]

        supporting_score = (
            max(supporting_parts)
            if supporting_parts
            else 0.0
        )

        # ============================================================
        # FAMILY WEIGHTS
        # ============================================================
        #
        # Weights reflect the evidentiary role of each family.
        # Relational, economic and temporal signals are the primary
        # mechanisms; geographic concentration is supporting evidence.

        family_weights = {
            "cooccurrence": 1.00,
            "economic": 0.95,
            "temporal": 0.85,
            "winner_rotation": 0.80,
            "geographic": 0.60,
            "winner_concentration": 0.45,
        }

        family_weight_values = np.array(
            [
                float(
                    family_weights.get(
                        family,
                        1.0,
                    )
                )
                for family in families
            ],
            dtype=float,
        )

        family_weight_values = np.clip(
            family_weight_values,
            1e-9,
            None,
        )

        # ============================================================
        # CASE-LEVEL PERSISTENCE
        # ============================================================
        #
        # Aggregate persistence across the evidence families present
        # for this vendor. This is metadata about how consistently the
        # observed signals persist; it is not another independent
        # evidence family.

        persistence_values = np.array(
            [
                float(
                    np.clip(
                        family_persistence.get(
                            family,
                            0.0,
                        ),
                        0.0,
                        1.0,
                    )
                )
                for family in families
            ],
            dtype=float,
        )

        if len(persistence_values) > 0:
            persistence = float(
                np.average(
                    persistence_values,
                    weights=family_weight_values,
                )
            )
        else:
            persistence = 0.0

        persistence = float(
            np.clip(
                persistence,
                0.0,
                1.0,
            )
        )

        mechanism_scores = {
            "network": network_score,
            "economic": economic_score,
            "temporal": temporal_score,
        }

        # ============================================================
        # CONTEXTUAL SUPPRESSION
        # ============================================================
        #
        # Context should reduce investigation priority when unusual
        # behaviour is well explained by legitimate market structure.
        #
        # contextual_adjustment is produced by the behavioural
        # relationship engine. Values near 1 indicate stronger
        # contextual support for the relationship being meaningful;
        # lower values indicate stronger legitimate-context suppression.
        #
        # The factor is bounded so context can suppress evidence without
        # completely erasing an otherwise strong independent signal.

        context_values = []

        for family in families:
            value = family_contexts.get(
                family,
                1.0,
            )

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 1.0

            context_values.append(
                float(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                )
            )

        if context_values:
            context_factor = float(
                np.mean(
                    context_values
                )
            )
        else:
            context_factor = 1.0

        # Case-level context metadata.
        #
        # This does not modify the convergence score. It records the
        # weakest contextual support among the evidence families
        # contributing to this case.

        context_values = np.array(
            [
                float(
                    np.clip(
                        family_contexts.get(
                            family,
                            0.0,
                        ),
                        0.0,
                        1.0,
                    )
                )
                for family in families
            ],
            dtype=float,
        )

        minimum_context = (
            float(np.min(context_values))
            if len(context_values) > 0
            else 0.0
        )


        context_factor = float(
            np.clip(
                context_factor,
                0.35,
                1.0,
            )
        )

        # ============================================================
        # CONTINUOUS MECHANISM EVIDENCE
        # ============================================================
        #
        # Mechanism evidence is continuous rather than binary.
        #
        # A score of 0.18 is weaker than 0.45, but it is still evidence.
        # Hard activation thresholds create discontinuities where very
        # different cases can receive the same priority merely because
        # they fall on opposite sides of a threshold.
        #
        # Network = co-occurrence + winner rotation.
        # Economic = procurement-price behaviour.
        # Temporal = vendor behavioural regime change.
        #
        # Geographic and winner concentration remain supporting evidence.

        core_scores = np.array(
            [
                float(score)
                for score in mechanism_scores.values()
            ],
            dtype=float,
        )

        core_scores = np.clip(
            core_scores,
            0.0,
            1.0,
        )

        strongest_core = float(
            core_scores.max()
        )

        # Tiny numerical signals should not count as meaningful
        # mechanisms, but there is deliberately no high activation
        # threshold.
        meaningful_core = core_scores[
            core_scores >= 0.08
        ]

        core_count = int(
            len(meaningful_core)
        )

        if core_count:
            core_mean = float(
                meaningful_core.mean()
            )

            # Strongest mechanism dominates; additional mechanisms
            # provide corroboration without simple additive inflation.
            core_convergence = (
                0.72
                * strongest_core
                + 0.28
                * core_mean
            )
        else:
            core_mean = 0.0
            core_convergence = 0.0

        # Independent mechanism bonus:
        #
        # 1 mechanism -> 1.00
        # 2 mechanisms -> 1.14
        # 3 mechanisms -> 1.28
        #
        # The bonus is intentionally modest.
        mechanism_bonus = (
            1.0
            + 0.14
            * min(
                max(core_count - 1, 0),
                2,
            )
        )

        supporting_strength = float(
            np.clip(
                supporting_score,
                0.0,
                1.0,
            )
        )

        # ============================================================
        # PRIORITY
        # ============================================================

        if core_count == 0:
            # Supporting evidence alone produces a continuously ranked
            # lead. There is no artificial 24-point pile-up.
            #
            # Supporting evidence cannot reach the priority tier of
            # a genuinely corroborated core case.
            base = (
                0.06
                + 0.34
                * supporting_strength
            )

            priority = (
                base
                * (
                    0.45
                    + 0.55
                    * context_factor
                )
                * 100.0
            )

            priority = min(
                priority,
                20.0,
            )

        elif core_count == 1:
            # A single core mechanism is meaningful, but receives less
            # convergence uplift than multiple independent mechanisms.
            base = (
                0.76
                * strongest_core
                + 0.10
                * core_mean
                + 0.08
                * supporting_strength
            )

            priority = (
                base
                * (
                    0.45
                    + 0.55
                    * context_factor
                )
                * 100.0
            )

            priority = min(
                priority,
                40.0,
            )

        else:
            # Multiple independent mechanisms.
            #
            # This is the principal convergence signal:
            # relational + economic,
            # relational + temporal,
            # economic + temporal,
            # or all three.
            base = (
                0.50
                * strongest_core
                + 0.38
                * core_convergence
                + 0.12
                * supporting_strength
            )

            base *= mechanism_bonus

            context_modifier = (
                0.42
                + 0.58
                * context_factor
            )

            priority = (
                base
                * context_modifier
                * 100.0
            )

        priority = float(
            np.clip(
                priority,
                0.0,
                100.0,
            )
        )

        # ============================================================
        # EVIDENCE DIVERSITY
        # ============================================================

        family_count = len(
            families
        )

        diversity = float(
            FAMILY_DIVERSITY.get(
                family_count,
                1.0,
            )
        )

        # ============================================================
        # EXPLANATIONS
        # ============================================================

        explanations = []

        for family in sorted(
            families,
            key=lambda x: family_scores[x],
            reverse=True,
        ):

            family_group = vendor_group[
                vendor_group[
                    "evidence_family"
                ] == family
            ]

            top = family_group.loc[
                family_group[
                    "adjusted_strength"
                ].idxmax()
            ]

            explanations.append(
                (
                    f"[{family}] "
                    f"{top['explanation']}"
                )
            )

        # ============================================================
        # HIERARCHY METADATA
        # ============================================================

        core_families = []

        if network_score >= 0.25:
            core_families.append(
                "network"
            )

        if economic_score >= 0.25:
            core_families.append(
                "economic"
            )

        if temporal_score >= 0.25:
            core_families.append(
                "temporal"
            )

        supporting_families = []

        if "geographic" in family_scores:
            if family_scores[
                "geographic"
            ] >= 0.30:
                supporting_families.append(
                    "geographic"
                )

        if "winner_concentration" in family_scores:
            if family_scores[
                "winner_concentration"
            ] >= 0.30:
                supporting_families.append(
                    "winner_concentration"
                )

        # ============================================================
        # RISK COMPONENTS
        # ============================================================

        risk_components = {
            family: round(
                float(
                    family_scores[
                        family
                    ]
                ),
                4,
            )
            for family in families
        }

        rows.append(
            {
                "vendor_id": vendor_id,

                "investigation_priority": priority,

                "evidence_family_count":
                    family_count,

                "evidence_families":
                    ",".join(families),

                "weighted_evidence_strength":
                    round(
                        float(
                            np.average(
                                np.array(
                                    [
                                        family_scores[f]
                                        for f in families
                                    ],
                                    dtype=float,
                                ),
                                weights=family_weight_values,
                            )
                        ),
                        6,
                    ),

                "strongest_family_strength":
                    round(
                        float(
                            max(
                                family_scores.values()
                            )
                        ),
                        6,
                    ),

                "convergence_strength":
                    round(
                        float(
                            np.clip(
                                core_convergence,
                                0.0,
                                1.0,
                            )
                        ),
                        6,
                    ),

                "evidence_diversity":
                    round(
                        diversity,
                        6,
                    ),

                "persistence_strength":
                    round(
                        persistence,
                        6,
                    ),

                "context_factor":
                    round(
                        context_factor,
                        6,
                    ),

                "minimum_family_context":
                    round(
                        minimum_context,
                        6,
                    ),

                "core_evidence_families":
                    ",".join(
                        core_families
                    ),

                "supporting_evidence_families":
                    ",".join(
                        supporting_families
                    ),

                "core_evidence_strength":
                    round(
                        strongest_core,
                        6,
                    ),

                "supporting_evidence_strength":
                    round(
                        supporting_score,
                        6,
                    ),

                "evidence_count":
                    int(
                        len(vendor_group)
                    ),

                "risk_components":
                    str(
                        risk_components
                    ),

                "explanations":
                    " | ".join(
                        explanations
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

def self_test(
    evidence: pd.DataFrame,
    cases: pd.DataFrame,
):

    assert not evidence.empty
    assert not cases.empty

    required_evidence = {
        "vendor_id",
        "evidence_family",
        "adjusted_strength",
        "context_factor",
        "persistence",
    }

    required_cases = {
        "vendor_id",
        "investigation_priority",
        "evidence_family_count",
        "evidence_families",
        "convergence_strength",
        "evidence_diversity",
    }

    assert required_evidence.issubset(
        evidence.columns
    )

    assert required_cases.issubset(
        cases.columns
    )

    assert np.isfinite(
        evidence[
            "adjusted_strength"
        ].to_numpy(
            dtype=float
        )
    ).all()

    assert np.isfinite(
        cases[
            "investigation_priority"
        ].to_numpy(
            dtype=float
        )
    ).all()

    assert (
        evidence[
            "adjusted_strength"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    assert (
        evidence[
            "context_factor"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    assert (
        evidence[
            "persistence"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    assert (
        cases[
            "investigation_priority"
        ]
        .between(
            0.0,
            100.0,
        )
        .all()
    )

    assert (
        cases[
            "evidence_family_count"
        ].ge(1)
        .all()
    )

    assert (
        cases[
            "evidence_diversity"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    assert (
        cases[
            "evidence_family_count"
        ]
        .ge(2)
        .any()
    )

    assert (
        cases[
            "evidence_family_count"
        ]
        .eq(1)
        .any()
    )

    # No family may exceed 1 after robust aggregation.
    for vendor_id, group in evidence.groupby(
        "vendor_id"
    ):
        for family, family_group in group.groupby(
            "evidence_family"
        ):
            score = _family_score(
                family_group
            )
            assert 0.0 <= score <= 1.0

    print(
        "[PASS] hardened convergence self-test"
    )


def scenario_audit(
    cases: pd.DataFrame,
):

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
    print("HARDENED CONVERGENCE SCENARIO AUDIT")
    print("=" * 72)

    maxima = {}

    for name, vendors in scenarios.items():

        subset = cases[
            cases["vendor_id"].isin(
                vendors
            )
        ]

        if subset.empty:
            print(
                f"{name:<28} no case"
            )
            continue

        row = subset.loc[
            subset[
                "investigation_priority"
            ].idxmax()
        ]

        priority = float(
            row[
                "investigation_priority"
            ]
        )

        maxima[name] = priority

        print(
            f"{name:<28} "
            f"priority={priority:6.2f} "
            f"families="
            f"{row['evidence_families']} "
            f"diversity="
            f"{float(row['evidence_diversity']):.2f}"
        )

    print()

    if (
        "REPEATED_NETWORK" in maxima
        and "LEGITIMATE_SPECIALIZATION" in maxima
    ):

        if (
            maxima["REPEATED_NETWORK"]
            > maxima[
                "LEGITIMATE_SPECIALIZATION"
            ]
        ):
            print(
                "[PASS] repeated network "
                "out-ranks legitimate specialization"
            )
        else:
            print(
                "[WARN] repeated network separation "
                "still requires calibration"
            )

    if (
        "COVER_BIDDING" in maxima
        and "LEGITIMATE_SPECIALIZATION" in maxima
    ):

        if (
            maxima["COVER_BIDDING"]
            > maxima[
                "LEGITIMATE_SPECIALIZATION"
            ]
        ):
            print(
                "[PASS] cover bidding "
                "out-ranks legitimate specialization"
            )
        else:
            print(
                "[WARN] cover bidding separation "
                "still requires calibration"
            )


def main():

    print("=" * 72)
    print("VERITAS HARDENED EVIDENCE CONVERGENCE ENGINE")
    print("=" * 72)

    print()
    print(
        "[1/3] Collecting independent evidence"
    )

    evidence = build_evidence()

    print(
        f"       {len(evidence):,} evidence records"
    )

    print()
    print(
        "[2/3] Collapsing correlated evidence"
    )

    cases = evidence_convergence(
        evidence
    )

    print(
        f"       {len(cases):,} investigation cases"
    )

    print()
    print(
        "[3/3] Running hardened self-test"
    )

    self_test(
        evidence,
        cases,
    )

    evidence.to_csv(
        EVIDENCE_OUTPUT,
        index=False,
    )

    cases.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"[OK] {EVIDENCE_OUTPUT}"
    )

    print(
        f"[OK] {OUTPUT}"
    )

    scenario_audit(
        cases
    )

    print()
    print("=" * 72)
    print("TOP INVESTIGATION CASES")
    print("=" * 72)

    cols = [
        "vendor_id",
        "investigation_priority",
        "evidence_family_count",
        "evidence_families",
        "weighted_evidence_strength",
        "strongest_family_strength",
        "convergence_strength",
        "evidence_diversity",
        "persistence_strength",
        "context_factor",
        "evidence_count",
    ]

    print(
        cases.sort_values(
            [
                "investigation_priority",
                "evidence_family_count",
                "weighted_evidence_strength",
            ],
            ascending=False,
        )[cols]
        .head(25)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

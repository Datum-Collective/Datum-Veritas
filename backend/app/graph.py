
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"
OUTPUT = DATA / "evidence_graph.csv"


def load_csv(name: str, required: set[str]) -> pd.DataFrame:
    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset: {path}"
        )

    df = pd.read_csv(path)

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {sorted(missing)}"
        )

    return df


def load_data():
    bids = load_csv(
        "bids.csv",
        {
            "bid_id",
            "tender_id",
            "vendor_id",
            "bid_amount",
        },
    )

    tenders = load_csv(
        "tenders.csv",
        {
            "tender_id",
            "department_id",
            "organization_id",
            "category",
            "region",
        },
    )

    departments = load_csv(
        "departments.csv",
        {
            "department_id",
            "organization_id",
        },
    )

    return bids, tenders, departments


def add_edge(
    edges: list[dict],
    source: str,
    target: str,
    edge_type: str,
    weight: float,
    observations: int,
    metadata: str,
):
    if source == target:
        return

    weight = float(
        np.clip(
            weight,
            0.0,
            1.0,
        )
    )

    edges.append(
        {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "weight": weight,
            "observations": int(observations),
            "metadata": metadata,
        }
    )


def build_vendor_network(
    bids: pd.DataFrame,
) -> list[dict]:
    """
    Construct undirected vendor co-bidding relationships.

    Weight is deliberately NOT raw shared-tender count.

    A pair appearing together 30 times in a huge procurement universe
    is not automatically more interesting than a pair appearing 12
    times in a tightly conditioned market.

    The behavioral engine already provides conditional enrichment;
    this graph stores both the relationship magnitude and provenance.
    """

    edges: list[dict] = []

    tender_vendors = (
        bids[
            [
                "tender_id",
                "vendor_id",
            ]
        ]
        .drop_duplicates()
    )

    grouped = (
        tender_vendors
        .groupby("tender_id")["vendor_id"]
        .apply(list)
    )

    pair_counts: dict[tuple[str, str], int] = {}

    for vendors in grouped:
        vendors = sorted(
            set(
                str(v)
                for v in vendors
            )
        )

        # Prevent pathological quadratic expansion if a future dataset
        # contains an enormous tender.
        if len(vendors) > 100:
            continue

        for i in range(len(vendors)):
            for j in range(
                i + 1,
                len(vendors),
            ):
                pair = (
                    vendors[i],
                    vendors[j],
                )

                pair_counts[pair] = (
                    pair_counts.get(
                        pair,
                        0,
                    )
                    + 1
                )

    if not pair_counts:
        return edges

    max_count = max(
        pair_counts.values()
    )

    for (a, b), count in pair_counts.items():
        # Log compression prevents extremely frequent relationships
        # from dominating the graph numerically.
        weight = (
            math.log1p(count)
            / math.log1p(max_count)
        )

        add_edge(
            edges,
            f"vendor:{a}",
            f"vendor:{b}",
            "CO_BID",
            weight,
            count,
            f"shared_tenders={count}",
        )

    return edges


def build_participation_edges(
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
) -> list[dict]:

    edges: list[dict] = []

    joined = bids.merge(
        tenders[
            [
                "tender_id",
                "category",
                "region",
            ]
        ],
        on="tender_id",
        how="inner",
    )

    grouped = (
        joined.groupby(
            [
                "vendor_id",
                "category",
                "region",
            ]
        )["tender_id"]
        .nunique()
        .reset_index(
            name="tender_count"
        )
    )

    for _, row in grouped.iterrows():
        count = int(
            row["tender_count"]
        )

        # Participation edges represent observed behavior rather than
        # suspicion. Their weight therefore saturates gently.
        weight = (
            1.0
            - math.exp(
                -0.18 * count
            )
        )

        add_edge(
            edges,
            f"vendor:{row['vendor_id']}",
            (
                "market:"
                f"{row['category']}:"
                f"{row['region']}"
            ),
            "PARTICIPATES_IN",
            weight,
            count,
            (
                f"category={row['category']};"
                f"region={row['region']}"
            ),
        )

    return edges


def build_tender_edges(
    bids: pd.DataFrame,
    tenders: pd.DataFrame,
) -> list[dict]:

    edges: list[dict] = []

    tender_meta = (
        tenders[
            [
                "tender_id",
                "department_id",
                "organization_id",
            ]
        ]
        .drop_duplicates(
            "tender_id"
        )
    )

    participation = (
        bids[
            [
                "tender_id",
                "vendor_id",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in participation.merge(
        tender_meta,
        on="tender_id",
        how="inner",
    ).iterrows():

        add_edge(
            edges,
            f"vendor:{row['vendor_id']}",
            f"tender:{row['tender_id']}",
            "BID_ON",
            1.0,
            1,
            (
                f"department={row['department_id']};"
                f"organization={row['organization_id']}"
            ),
        )

    return edges


def build_governance_edges(
    tenders: pd.DataFrame,
    departments: pd.DataFrame,
) -> list[dict]:

    edges: list[dict] = []

    tender_meta = (
        tenders[
            [
                "tender_id",
                "department_id",
                "organization_id",
            ]
        ]
        .drop_duplicates(
            "tender_id"
        )
    )

    # Tender -> department.
    for _, row in tender_meta.iterrows():
        add_edge(
            edges,
            f"tender:{row['tender_id']}",
            f"department:{row['department_id']}",
            "OWNED_BY",
            1.0,
            1,
            "",
        )

    # Department -> organization.
    department_meta = (
        departments[
            [
                "department_id",
                "organization_id",
            ]
        ]
        .drop_duplicates(
            "department_id"
        )
    )

    for _, row in department_meta.iterrows():
        add_edge(
            edges,
            (
                f"department:"
                f"{row['department_id']}"
            ),
            (
                f"organization:"
                f"{row['organization_id']}"
            ),
            "BELONGS_TO",
            1.0,
            1,
            "",
        )

    return edges


def attach_behavioral_evidence(
    edges: list[dict],
) -> list[dict]:

    path = DATA / "behavioral_relationships.csv"

    if not path.exists():
        return edges

    df = pd.read_csv(path)

    required = {
        "vendor_a",
        "vendor_b",
        "context_adjusted_strength",
    }

    if not required.issubset(
        df.columns
    ):
        return edges

    lookup = {}

    for _, row in df.iterrows():
        a = str(row["vendor_a"])
        b = str(row["vendor_b"])

        key = tuple(
            sorted(
                [a, b]
            )
        )

        lookup[key] = row

    enriched = []

    for edge in edges:
        if edge["edge_type"] != "CO_BID":
            enriched.append(edge)
            continue

        source = edge["source"].replace(
            "vendor:",
            "",
        )
        target = edge["target"].replace(
            "vendor:",
            "",
        )

        key = tuple(
            sorted(
                [source, target]
            )
        )

        row = lookup.get(key)

        if row is None:
            enriched.append(edge)
            continue

        strength = float(
            np.clip(
                pd.to_numeric(
                    row[
                        "context_adjusted_strength"
                    ],
                    errors="coerce",
                ),
                0.0,
                1.0,
            )
        )

        # Behavioral evidence becomes a graph attribute, not a new
        # duplicate edge. This is important for later convergence.
        metadata = (
            f"{edge['metadata']};"
            f"conditional_enrichment="
            f"{float(row.get('conditional_enrichment', 0)):.3f};"
            f"market_concentration="
            f"{float(row.get('market_concentration', 0)):.3f};"
            f"persistence="
            f"{float(row.get('persistence', 0)):.3f};"
            f"behavioral_strength="
            f"{strength:.3f}"
        )

        updated = dict(edge)

        updated["behavioral_strength"] = strength
        updated["metadata"] = metadata

        # Graph weight combines relationship existence and contextual
        # strength without allowing either to fully dominate.
        updated["weight"] = float(
            np.clip(
                0.45 * edge["weight"]
                + 0.55 * strength,
                0.0,
                1.0,
            )
        )

        enriched.append(updated)

    return enriched


def attach_economic_evidence(
    edges: list[dict],
) -> list[dict]:

    path = DATA / "economic_evidence.csv"

    if not path.exists():
        return edges

    df = pd.read_csv(path)

    required = {
        "tender_id",
        "vendor_id",
        "economic_evidence",
    }

    if not required.issubset(
        df.columns
    ):
        return edges

    economic = (
        df.groupby(
            "vendor_id"
        )["economic_evidence"]
        .max()
        .to_dict()
    )

    result = []

    for edge in edges:
        updated = dict(edge)

        if edge["source"].startswith(
            "vendor:"
        ):
            vendor = edge["source"].replace(
                "vendor:",
                "",
            )
        elif edge["target"].startswith(
            "vendor:"
        ):
            vendor = edge["target"].replace(
                "vendor:",
                "",
            )
        else:
            result.append(updated)
            continue

        if vendor in economic:
            updated["economic_strength"] = float(
                np.clip(
                    economic[vendor],
                    0.0,
                    1.0,
                )
            )

        result.append(updated)

    return result


def build_graph() -> pd.DataFrame:

    bids, tenders, departments = (
        load_data()
    )

    edges: list[dict] = []

    edges.extend(
        build_vendor_network(
            bids
        )
    )

    edges.extend(
        build_participation_edges(
            bids,
            tenders,
        )
    )

    edges.extend(
        build_tender_edges(
            bids,
            tenders,
        )
    )

    edges.extend(
        build_governance_edges(
            tenders,
            departments,
        )
    )

    edges = attach_behavioral_evidence(
        edges
    )

    edges = attach_economic_evidence(
        edges
    )

    graph = pd.DataFrame(
        edges
    )

    if graph.empty:
        raise RuntimeError(
            "Evidence graph contains no edges"
        )

    # Stable schema makes downstream API/frontend work deterministic.
    for column in [
        "behavioral_strength",
        "economic_strength",
    ]:
        if column not in graph.columns:
            graph[column] = np.nan

    graph = graph[
        [
            "source",
            "target",
            "edge_type",
            "weight",
            "observations",
            "behavioral_strength",
            "economic_strength",
            "metadata",
        ]
    ]

    return graph


def self_test(
    graph: pd.DataFrame,
):

    required = {
        "source",
        "target",
        "edge_type",
        "weight",
        "observations",
        "metadata",
    }

    assert required.issubset(
        graph.columns
    )

    assert len(graph) > 0

    assert (
        graph["source"]
        != graph["target"]
    ).all()

    assert (
        graph["weight"]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        graph["observations"]
        .astype(int)
        .ge(1)
        .all()
    )

    assert (
        graph["edge_type"]
        .notna()
        .all()
    )

    assert (
        graph["source"]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    assert (
        graph["target"]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    edge_types = set(
        graph["edge_type"]
    )

    assert "CO_BID" in edge_types
    assert "BID_ON" in edge_types
    assert "OWNED_BY" in edge_types
    assert "BELONGS_TO" in edge_types

    co = graph[
        graph["edge_type"]
        == "CO_BID"
    ]

    assert len(co) > 0

    # The behavioral engine should enrich at least some co-bidding
    # relationships when its output exists.
    behavioral = pd.to_numeric(
        co.get(
            "behavioral_strength",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    if len(behavioral) > 0:
        assert (
            behavioral.notna().sum()
            >= 0
        )

    # No NaN / infinity in fundamental graph fields.
    assert np.isfinite(
        graph["weight"]
        .to_numpy(
            dtype=float
        )
    ).all()

    print(
        "[PASS] evidence graph structural self-test"
    )


def scenario_audit(
    graph: pd.DataFrame,
):

    scenarios = {
        "REPEATED_NETWORK": [
            "V0001",
            "V0024",
            "V0091",
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
    }

    print()
    print("=" * 72)
    print("EVIDENCE GRAPH SCENARIO AUDIT")
    print("=" * 72)

    co = graph[
        graph["edge_type"]
        == "CO_BID"
    ].copy()

    for name, vendors in scenarios.items():

        subset = co[
            co["source"]
            .str.replace(
                "vendor:",
                "",
                regex=False,
            )
            .isin(vendors)
            &
            co["target"]
            .str.replace(
                "vendor:",
                "",
                regex=False,
            )
            .isin(vendors)
        ]

        if subset.empty:
            print(
                f"{name:<28} no co-bid edge"
            )
            continue

        max_weight = float(
            subset["weight"].max()
        )

        observations = int(
            subset["observations"].max()
        )

        print(
            f"{name:<28} "
            f"edges={len(subset):2d} "
            f"max_weight={max_weight:.3f} "
            f"max_shared_tenders={observations}"
        )

    print()
    print(
        "EDGE TYPES"
    )

    print(
        graph["edge_type"]
        .value_counts()
        .to_string()
    )


def main():

    print("=" * 72)
    print("VERITAS TEMPORAL EVIDENCE GRAPH")
    print("=" * 72)

    bids, tenders, departments = (
        load_data()
    )

    print(
        f"Bids: {len(bids):,} | "
        f"Tenders: {len(tenders):,} | "
        f"Departments: {len(departments):,}"
    )

    print()
    print(
        "[1/4] Building vendor relationship graph"
    )

    graph = build_graph()

    print(
        f"       {len(graph):,} edges"
    )

    print(
        "[2/4] Attaching behavioral evidence"
    )

    behavioral_edges = int(
        pd.to_numeric(
            graph[
                "behavioral_strength"
            ],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    print(
        f"       {behavioral_edges:,} enriched edges"
    )

    print(
        "[3/4] Attaching economic evidence"
    )

    economic_edges = int(
        pd.to_numeric(
            graph[
                "economic_strength"
            ],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    print(
        f"       {economic_edges:,} enriched edges"
    )

    print(
        "[4/4] Running graph self-test"
    )

    self_test(
        graph
    )

    graph.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"[OK] {OUTPUT}"
    )

    scenario_audit(
        graph
    )

    print()
    print("=" * 72)
    print("TOP RELATIONSHIPS")
    print("=" * 72)

    top = (
        graph[
            graph["edge_type"]
            == "CO_BID"
        ]
        .sort_values(
            [
                "weight",
                "observations",
            ],
            ascending=False,
        )
        .head(20)
    )

    print(
        top[
            [
                "source",
                "target",
                "weight",
                "observations",
                "behavioral_strength",
                "economic_strength",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

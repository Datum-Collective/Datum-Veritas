from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "generated"


app = FastAPI(
    title="Veritas Procurement Intelligence API",
    version="0.1.0",
    description=(
        "Investigator-facing API for procurement anomaly evidence, "
        "investigation priority, and relationship context."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_CACHE: dict[str, pd.DataFrame] = {}


def _load(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        path = DATA_DIR / f"{name}.csv"

        if not path.exists():
            raise RuntimeError(f"Required data file missing: {path}")

        _CACHE[name] = pd.read_csv(path)

    return _CACHE[name]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    clean = df.copy()

    clean = clean.replace(
        {
            float("inf"): None,
            float("-inf"): None,
        }
    )

    clean = clean.where(pd.notna(clean), None)

    return clean.to_dict(orient="records")


def _case(vendor_id: str) -> dict[str, Any]:
    cases = _load("investigation_cases")

    if "vendor_id" not in cases.columns:
        raise RuntimeError("investigation_cases.csv has no vendor_id column")

    match = cases[
        cases["vendor_id"].astype(str).str.upper()
        == vendor_id.upper()
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation case found for vendor {vendor_id}",
        )

    return _records(match.head(1))[0]


def _vendor(vendor_id: str) -> dict[str, Any]:
    vendors = _load("vendors")

    if "vendor_id" not in vendors.columns:
        raise RuntimeError("vendors.csv has no vendor_id column")

    match = vendors[
        vendors["vendor_id"].astype(str).str.upper()
        == vendor_id.upper()
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Vendor {vendor_id} not found",
        )

    return _records(match.head(1))[0]


@app.get("/api/health")
def health() -> dict[str, Any]:
    required = [
        "vendors",
        "tenders",
        "bids",
        "investigation_cases",
        "converged_evidence",
        "evidence_graph",
    ]

    status = {}

    for name in required:
        path = DATA_DIR / f"{name}.csv"
        status[name] = path.exists()

    return {
        "status": "ok" if all(status.values()) else "degraded",
        "data_directory": str(DATA_DIR),
        "artifacts": status,
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    vendors = _load("vendors")
    tenders = _load("tenders")
    bids = _load("bids")
    cases = _load("investigation_cases")
    evidence = _load("converged_evidence")

    summary_data: dict[str, Any] = {
        "vendors": len(vendors),
        "tenders": len(tenders),
        "bids": len(bids),
        "investigation_cases": len(cases),
        "evidence_records": len(evidence),
    }

    if "investigation_priority" in cases.columns:
        priorities = pd.to_numeric(
            cases["investigation_priority"],
            errors="coerce",
        )

        summary_data["priority"] = {
            "min": round(float(priorities.min()), 2),
            "median": round(float(priorities.median()), 2),
            "p90": round(float(priorities.quantile(0.90)), 2),
            "max": round(float(priorities.max()), 2),
        }

    if "evidence_family_count" in cases.columns:
        family_counts = pd.to_numeric(
            cases["evidence_family_count"],
            errors="coerce",
        )

        summary_data["multi_family_cases"] = int(
            (family_counts >= 2).sum()
        )

    if "evidence_families" in cases.columns:
        families: dict[str, int] = {}

        for value in cases["evidence_families"].dropna():
            for family in str(value).split(","):
                family = family.strip()

                if family:
                    families[family] = families.get(family, 0) + 1

        summary_data["evidence_families"] = dict(
            sorted(
                families.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    return summary_data


@app.get("/api/cases")
def cases(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_priority: float | None = Query(default=None, ge=0),
    family: str | None = Query(default=None),
) -> dict[str, Any]:
    df = _load("investigation_cases").copy()

    if "investigation_priority" in df.columns:
        df["investigation_priority"] = pd.to_numeric(
            df["investigation_priority"],
            errors="coerce",
        )
        df = df.sort_values(
            "investigation_priority",
            ascending=False,
            na_position="last",
        )

    if min_priority is not None:
        df = df[
            df["investigation_priority"] >= min_priority
        ]

    if family:
        if "evidence_families" not in df.columns:
            raise RuntimeError(
                "investigation_cases.csv has no evidence_families column"
            )

        needle = family.strip().lower()

        df = df[
            df["evidence_families"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(needle, regex=False)
        ]

    total = len(df)

    page = df.iloc[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "cases": _records(page),
    }


@app.get("/api/cases/{vendor_id}")
def case_detail(vendor_id: str) -> dict[str, Any]:
    case = _case(vendor_id)

    try:
        vendor = _vendor(vendor_id)
    except HTTPException:
        vendor = {"vendor_id": vendor_id}

    return {
        "case": case,
        "vendor": vendor,
    }


@app.get("/api/cases/{vendor_id}/evidence")
def case_evidence(vendor_id: str) -> dict[str, Any]:
    case = _case(vendor_id)

    evidence = _load("converged_evidence").copy()

    possible_columns = [
        "vendor_id",
        "case_id",
        "entity_id",
    ]

    matching_column = next(
        (
            column
            for column in possible_columns
            if column in evidence.columns
        ),
        None,
    )

    if matching_column == "vendor_id":
        filtered = evidence[
            evidence["vendor_id"].astype(str).str.upper()
            == vendor_id.upper()
        ]
    elif matching_column in {"case_id", "entity_id"}:
        filtered = evidence[
            evidence[matching_column].astype(str).str.upper()
            == vendor_id.upper()
        ]
    else:
        filtered = pd.DataFrame()

    return {
        "vendor_id": vendor_id,
        "investigation_priority": case.get(
            "investigation_priority"
        ),
        "evidence": _records(filtered),
        "count": len(filtered),
    }


@app.get("/api/cases/{vendor_id}/graph")
def case_graph(vendor_id: str) -> dict[str, Any]:
    _case(vendor_id)

    graph = _load("evidence_graph").copy()

    if graph.empty:
        return {
            "vendor_id": vendor_id.upper(),
            "nodes": [],
            "edges": [],
        }

    canonical_vendor = vendor_id.upper()
    graph_vendor = f"vendor:{canonical_vendor}"
    graph_vendor_normalized = graph_vendor.upper()

    required = {"source", "target"}

    if not required.issubset(graph.columns):
        raise RuntimeError(
            "evidence_graph.csv must contain source and target columns"
        )

    source = graph["source"].fillna("").astype(str)
    target = graph["target"].fillna("").astype(str)

    mask = (
        source.str.upper().eq(graph_vendor_normalized)
        | target.str.upper().eq(graph_vendor_normalized)
    )

    edges = graph[mask].copy()

    if edges.empty:
        return {
            "vendor_id": canonical_vendor,
            "nodes": [
                {
                    "id": graph_vendor,
                    "type": "vendor",
                }
            ],
            "edges": [],
        }

    edge_records = _records(edges)

    nodes: dict[str, dict[str, Any]] = {}

    for record in edge_records:
        source_id = record.get("source")
        target_id = record.get("target")

        for node_id in (source_id, target_id):
            if not node_id:
                continue

            node_id = str(node_id)

            if node_id in nodes:
                continue

            normalized = node_id.lower()

            if normalized.startswith("vendor:"):
                node_type = "vendor"
            elif normalized.startswith("tender:"):
                node_type = "tender"
            elif normalized.startswith("department:"):
                node_type = "department"
            elif normalized.startswith("organization:"):
                node_type = "organization"
            elif normalized.startswith("person:"):
                node_type = "person"
            else:
                node_type = "entity"

            nodes[node_id] = {
                "id": node_id,
                "type": node_type,
            }

    return {
        "vendor_id": canonical_vendor,
        "nodes": list(nodes.values()),
        "edges": edge_records,
    }


@app.get("/api/vendors")
def vendors(
    limit: int = Query(default=50, ge=1, le=500),
    search: str | None = Query(default=None),
) -> dict[str, Any]:
    df = _load("vendors").copy()

    if search:
        needle = search.strip().lower()

        text_columns = [
            column
            for column in [
                "vendor_id",
                "legal_name",
                "vendor_type",
                "headquarters_region",
            ]
            if column in df.columns
        ]

        if text_columns:
            mask = pd.Series(False, index=df.index)

            for column in text_columns:
                mask = (
                    mask
                    | df[column]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        needle,
                        regex=False,
                    )
                )

            df = df[mask]

    return {
        "total": len(df),
        "vendors": _records(df.head(limit)),
    }


@app.get("/api/vendors/{vendor_id}")
def vendor_detail(vendor_id: str) -> dict[str, Any]:
    vendor = _vendor(vendor_id)

    cases = _load("investigation_cases")

    if "vendor_id" in cases.columns:
        vendor_cases = cases[
            cases["vendor_id"].astype(str).str.upper()
            == vendor_id.upper()
        ]
    else:
        vendor_cases = pd.DataFrame()

    return {
        "vendor": vendor,
        "cases": _records(vendor_cases),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

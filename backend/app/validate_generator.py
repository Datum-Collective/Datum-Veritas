from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"

def fail(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[PASS] {msg}")

def section(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")

def load(name):
    path = DATA / name
    if not path.exists():
        fail(f"Missing {path}")
    return pd.read_csv(path)

section("VERITAS SYNTHETIC PROCUREMENT DATASET VALIDATION")

# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

organizations = load("organizations.csv")
departments = load("departments.csv")
vendors = load("vendors.csv")
market_profiles = load("market_profiles.csv")
persons = load("persons.csv")
addresses = load("addresses.csv")
relationships = load("relationships.csv")
tenders = load("tenders.csv")
bids = load("bids.csv")
awards = load("awards.csv")
contracts = load("contracts.csv")
truth = load("scenario_truth.csv")

print("\nDATASET SIZE")
for name, df in [
    ("organizations", organizations),
    ("departments", departments),
    ("vendors", vendors),
    ("market_profiles", market_profiles),
    ("persons", persons),
    ("addresses", addresses),
    ("relationships", relationships),
    ("tenders", tenders),
    ("bids", bids),
    ("awards", awards),
    ("contracts", contracts),
    ("scenario_truth", truth),
]:
    print(f"{name:20s} {len(df):>8,}")

# ---------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------

section("1. SCHEMA INTEGRITY")

required = {
    "organizations": [
        "organization_id", "organization_name", "organization_type", "region"
    ],
    "departments": [
        "department_id", "organization_id", "region"
    ],
    "vendors": [
        "vendor_id"
    ],
    "tenders": [
        "tender_id", "department_id", "organization_id",
        "region", "category", "estimated_value"
    ],
    "bids": [
        "bid_id", "tender_id", "vendor_id", "bid_amount"
    ],
    "awards": [
        "award_id", "tender_id", "vendor_id", "award_value"
    ],
    "contracts": [
        "contract_id", "tender_id", "vendor_id", "contract_value"
    ],
}

loaded = {
    "organizations": organizations,
    "departments": departments,
    "vendors": vendors,
    "tenders": tenders,
    "bids": bids,
    "awards": awards,
    "contracts": contracts,
}

for table, cols in required.items():
    df = loaded[table]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{table}.csv missing columns: {missing}")

ok("Required schema columns exist")

# ---------------------------------------------------------------------
# PRIMARY KEY UNIQUENESS
# ---------------------------------------------------------------------

for table, key in [
    ("organizations", "organization_id"),
    ("departments", "department_id"),
    ("vendors", "vendor_id"),
    ("tenders", "tender_id"),
    ("bids", "bid_id"),
    ("awards", "award_id"),
    ("contracts", "contract_id"),
]:
    df = loaded[table]

    if df[key].isna().any():
        fail(f"{table}.{key} contains NULL values")

    if df[key].duplicated().any():
        dupes = df.loc[df[key].duplicated(), key].head().tolist()
        fail(f"{table}.{key} contains duplicates: {dupes}")

ok("Primary keys are unique and non-null")

# ---------------------------------------------------------------------
# REFERENTIAL INTEGRITY
# ---------------------------------------------------------------------

section("2. REFERENTIAL INTEGRITY")

def check_fk(child, child_col, parent, parent_col):
    valid = set(parent[parent_col].dropna())
    bad = child[~child[child_col].isin(valid)]

    if len(bad):
        sample = bad[child_col].head().tolist()
        fail(
            f"Broken FK {child_col} -> {parent_col}: "
            f"{len(bad)} invalid rows; sample={sample}"
        )

    ok(f"{child_col} -> {parent_col}")

check_fk(
    departments, "organization_id",
    organizations, "organization_id"
)

check_fk(
    tenders, "department_id",
    departments, "department_id"
)

check_fk(
    tenders, "organization_id",
    organizations, "organization_id"
)

check_fk(
    bids, "tender_id",
    tenders, "tender_id"
)

check_fk(
    bids, "vendor_id",
    vendors, "vendor_id"
)

check_fk(
    awards, "tender_id",
    tenders, "tender_id"
)

check_fk(
    awards, "vendor_id",
    vendors, "vendor_id"
)

check_fk(
    contracts, "tender_id",
    tenders, "tender_id"
)

check_fk(
    contracts, "vendor_id",
    vendors, "vendor_id"
)

# ---------------------------------------------------------------------
# TENDER ORGANIZATION / DEPARTMENT CONSISTENCY
# ---------------------------------------------------------------------

section("3. PROCUREMENT HIERARCHY")

dept_org = departments.set_index("department_id")["organization_id"]

expected_org = tenders["department_id"].map(dept_org)

bad = tenders[
    expected_org.notna()
    & (expected_org != tenders["organization_id"])
]

if len(bad):
    fail(
        f"{len(bad)} tenders have department/organization mismatch"
    )

ok("Every tender belongs to the correct department organization")

dept_region = departments.set_index("department_id")["region"]
expected_region = tenders["department_id"].map(dept_region)

bad = tenders[
    expected_region.notna()
    & (expected_region != tenders["region"])
]

if len(bad):
    fail(
        f"{len(bad)} tenders have department/region mismatch"
    )

ok("Every tender has the correct regional assignment")

# ---------------------------------------------------------------------
# BASIC NUMERIC SANITY
# ---------------------------------------------------------------------

section("4. NUMERIC / TEMPORAL SANITY")

for col in ["estimated_value"]:
    values = pd.to_numeric(tenders[col], errors="coerce")

    if values.isna().any():
        fail(f"tenders.{col} contains non-numeric values")

    if (values <= 0).any():
        fail(f"tenders.{col} contains non-positive values")

for col in ["bid_amount"]:
    values = pd.to_numeric(bids[col], errors="coerce")

    if values.isna().any():
        fail(f"bids.{col} contains non-numeric values")

    if (values <= 0).any():
        fail(f"bids.{col} contains non-positive values")

ok("Tender and bid monetary values are positive and numeric")

# Every tender must have bids.
bid_counts = bids.groupby("tender_id").size()

missing_bid_tenders = tenders[
    ~tenders["tender_id"].isin(bid_counts.index)
]

if len(missing_bid_tenders):
    fail(
        f"{len(missing_bid_tenders)} tenders have zero bids"
    )

ok("Every tender has at least one bid")

# ---------------------------------------------------------------------
# BIDDER COUNT SANITY
# ---------------------------------------------------------------------

section("5. BIDDER STRUCTURE")

distinct_bidders = (
    bids.groupby("tender_id")["vendor_id"]
    .nunique()
)

if (distinct_bidders < 2).any():
    bad = distinct_bidders[distinct_bidders < 2]
    fail(
        f"{len(bad)} tenders have fewer than 2 distinct bidders"
    )

reported = tenders.set_index("tender_id")["number_of_tenderers"]

if "number_of_tenderers" in tenders.columns:
    mismatch = distinct_bidders[
        distinct_bidders.index.isin(reported.index)
    ] != reported.loc[
        distinct_bidders.index.intersection(reported.index)
    ]

    if mismatch.any():
        fail(
            "number_of_tenderers does not match distinct bidder counts"
        )

ok("Every tender has a valid competitive bidder set")

print("\nBIDDER COUNT DISTRIBUTION")
print(distinct_bidders.describe().round(2).to_string())

# ---------------------------------------------------------------------
# MARKET STRUCTURE
# ---------------------------------------------------------------------

section("6. MARKET STRUCTURE")

print("\nTENDERS BY CATEGORY")
print(
    tenders["category"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTENDERS BY REGION")
print(
    tenders["region"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nPROCUREMENT METHOD")
if "procurement_method" in tenders.columns:
    print(
        tenders["procurement_method"]
        .value_counts(normalize=True)
        .round(3)
        .to_string()
    )

categories = tenders["category"].nunique()

if categories < 5:
    fail(
        f"Only {categories} procurement categories exist; "
        "benchmark lacks market diversity"
    )

regions = tenders["region"].nunique()

if regions < 4:
    fail(
        f"Only {regions} regions exist; benchmark lacks geographic diversity"
    )

ok(f"Market contains {categories} categories across {regions} regions")

# ---------------------------------------------------------------------
# CATEGORY BIDDER DIVERSITY
# ---------------------------------------------------------------------

section("7. CATEGORY COMPETITION")

tender_category = tenders.set_index("tender_id")["category"]

bid_check = bids.copy()
bid_check["category"] = bid_check["tender_id"].map(tender_category)

category_stats = (
    bid_check.groupby("category")
    .agg(
        tenders=("tender_id", "nunique"),
        unique_bidders=("vendor_id", "nunique"),
        avg_bidders_per_tender=("tender_id", "count"),
    )
)

category_stats["avg_bids_per_tender"] = (
    bid_check.groupby("category")
    .size()
    /
    bid_check.groupby("category")["tender_id"].nunique()
)

print(category_stats.round(2).to_string())

if (category_stats["unique_bidders"] < 3).any():
    bad = category_stats[
        category_stats["unique_bidders"] < 3
    ]
    fail(
        "Some categories have fewer than 3 participating vendors: "
        f"{bad.index.tolist()}"
    )

ok("Every market category has meaningful supplier diversity")

# ---------------------------------------------------------------------
# ORGANIZATION COVERAGE
# ---------------------------------------------------------------------

section("8. ORGANIZATION / REGIONAL COVERAGE")

org_tenders = (
    tenders.groupby("organization_id")
    .size()
)

missing_orgs = organizations[
    ~organizations["organization_id"].isin(org_tenders.index)
]

if len(missing_orgs):
    fail(
        "Organizations with zero tenders: "
        f"{missing_orgs['organization_id'].tolist()}"
    )

print("\nTENDERS BY ORGANIZATION")
print(org_tenders.to_string())

ok("Every organization participates in procurement")

# ---------------------------------------------------------------------
# SCENARIO TRUTH
# ---------------------------------------------------------------------

section("9. SCENARIO GROUND TRUTH")

print("\nSCENARIO TRUTH COLUMNS:")
print(list(truth.columns))

print("\nSCENARIO TRUTH:")
print(truth.to_string(index=False))

if len(truth) < 5:
    fail(
        f"Only {len(truth)} planted scenarios found; "
        "benchmark should contain multiple anomaly families"
    )

ok(f"{len(truth)} planted scenarios are present")

# ---------------------------------------------------------------------
# KNOWN SCENARIO VENDOR CHECKS
# ---------------------------------------------------------------------

section("10. PLANTED SCENARIO EMBEDDING")

scenario_groups = {
    "REPEATED_BIDDER_NETWORK": ["V0001", "V0024", "V0091"],
    "LEGITIMATE_SPECIALIZED_MARKET": ["V0010", "V0020"],
    "COVER_BIDDING": ["V0042", "V0043"],
    "BID_ROTATION": ["V0051", "V0052", "V0053"],
    "GEOGRAPHIC_MARKET_ALLOCATION": ["V0061", "V0062", "V0063"],
    "TIGHT_PRICE_CLUSTER": ["V0071", "V0072", "V0073", "V0074"],
}

vendor_ids = set(vendors["vendor_id"])

for scenario, group in scenario_groups.items():

    missing = [v for v in group if v not in vendor_ids]

    if missing:
        fail(
            f"{scenario}: missing vendors {missing}"
        )

    scenario_bids = bids[
        bids["vendor_id"].isin(group)
    ]

    if scenario_bids.empty:
        fail(
            f"{scenario}: scenario vendors never submitted bids"
        )

    tender_counts = (
        scenario_bids.groupby("vendor_id")["tender_id"]
        .nunique()
    )

    print(
        f"\n{scenario}:"
    )

    for vendor in group:
        print(
            f"  {vendor}: "
            f"{int(tender_counts.get(vendor, 0))} tenders"
        )

    ok(f"{scenario} vendors are embedded in procurement activity")

# ---------------------------------------------------------------------
# REPEATED CO-OCCURRENCE
# ---------------------------------------------------------------------

section("11. NETWORK SIGNAL VALIDATION")

network_group = scenario_groups["REPEATED_BIDDER_NETWORK"]

network_tenders = set(
    bids[
        bids["vendor_id"].isin(network_group)
    ]["tender_id"]
)

joint = 0

for tender_id, group in bids.groupby("tender_id"):
    bidders = set(group["vendor_id"])

    if set(network_group).issubset(bidders):
        joint += 1

print(
    f"Repeated bidder network joint tenders: {joint}"
)

if joint < 3:
    fail(
        "Repeated bidder network does not produce enough "
        "joint observations for network detection"
    )

ok("Repeated bidder network produces repeated co-occurrence")

# ---------------------------------------------------------------------
# LEGITIMATE SPECIALIZATION CHECK
# ---------------------------------------------------------------------

section("12. LEGITIMATE SPECIALIZATION NEGATIVE CONTROL")

specialized = scenario_groups[
    "LEGITIMATE_SPECIALIZED_MARKET"
]

specialized_bids = bids[
    bids["vendor_id"].isin(specialized)
].copy()

specialized_tenders = tenders[
    tenders["tender_id"].isin(
        specialized_bids["tender_id"]
    )
]

print(
    "\nSpecialized-market tenders:",
    len(specialized_tenders)
)

if len(specialized_tenders) < 3:
    fail(
        "Legitimate specialized-market scenario has too few observations"
    )

print("\nSpecialized-market categories:")
print(
    specialized_tenders["category"]
    .value_counts()
    .to_string()
)

ok("Legitimate specialization has enough observations to act as a negative control")

# ---------------------------------------------------------------------
# PRICE SANITY
# ---------------------------------------------------------------------

section("13. PRICE DISTRIBUTION")

bid_values = pd.to_numeric(
    bids["bid_amount"],
    errors="coerce"
)

print(
    bid_values.describe(
        percentiles=[0.01, 0.10, 0.50, 0.90, 0.99]
    ).round(2).to_string()
)

if bid_values.nunique() < 100:
    fail(
        "Bid amounts have insufficient diversity"
    )

ok("Bid distribution has meaningful variation")

# ---------------------------------------------------------------------
# AWARDS / CONTRACTS
# ---------------------------------------------------------------------

section("14. PROCUREMENT LIFECYCLE")

if len(awards) != len(tenders):
    fail(
        f"Expected one award per tender; "
        f"got {len(awards)} awards for {len(tenders)} tenders"
    )

if len(contracts) != len(tenders):
    fail(
        f"Expected one contract per tender; "
        f"got {len(contracts)} contracts for {len(tenders)} tenders"
    )

award_tenders = set(awards["tender_id"])
contract_tenders = set(contracts["tender_id"])
tender_ids = set(tenders["tender_id"])

if award_tenders != tender_ids:
    fail("Award lifecycle does not cover exactly the tender universe")

if contract_tenders != tender_ids:
    fail("Contract lifecycle does not cover exactly the tender universe")

ok("Tender -> award -> contract lifecycle is complete")

# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------

section("FINAL DATASET VERDICT")

print("""
The generator passed all structural and benchmark-integrity checks.

This validates:
  - relational integrity
  - organization/department hierarchy
  - regional coverage
  - market diversity
  - bidder diversity
  - procurement lifecycle
  - planted scenario embedding
  - repeated-network behavior
  - legitimate specialized-market negative control
  - price diversity
""")

print("[PASS] VERITAS DATA GENERATOR VALIDATION COMPLETE")

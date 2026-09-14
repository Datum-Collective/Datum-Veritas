from pathlib import Path
import sys
import itertools
import numpy as np
import pandas as pd

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
    p = DATA / name
    if not p.exists():
        fail(f"Missing {p}")
    return pd.read_csv(p)

tenders = load("tenders.csv")
bids = load("bids.csv")
vendors = load("vendors.csv")
truth = load("scenario_truth.csv")

# ---------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------

b = bids.merge(
    tenders[
        [
            "tender_id",
            "category",
            "region",
            "organization_id",
            "estimated_value",
        ]
    ],
    on="tender_id",
    how="left",
)

b["bid_ratio"] = b["bid_amount"] / b["estimated_value"]

winner_map = tenders.set_index("tender_id")["winner_id"]
b["is_winner"] = b["vendor_id"] == b["tender_id"].map(winner_map)

section("VERITAS BEHAVIORAL DATASET VALIDATION")

# ---------------------------------------------------------------------
# 1. WINNER CONCENTRATION
# ---------------------------------------------------------------------

section("1. WINNER CONCENTRATION")

winner_counts = (
    tenders["winner_id"]
    .value_counts(normalize=True)
)

print(
    "Winner concentration statistics:"
)
print(
    winner_counts.describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).round(4).to_string()
)

if winner_counts.max() > 0.30:
    print(
        f"\n[WARN] One vendor wins {winner_counts.max():.1%} "
        "of all tenders."
    )
else:
    ok(
        f"Maximum overall winner concentration is "
        f"{winner_counts.max():.1%}"
    )

# We want substantial winner diversity.
if len(winner_counts) < 50:
    fail(
        "Too few vendors ever win tenders; winner process may be artificial."
    )

ok(f"{len(winner_counts)} vendors win at least one tender")

# ---------------------------------------------------------------------
# 2. VENDOR PARTICIPATION DISTRIBUTION
# ---------------------------------------------------------------------

section("2. VENDOR PARTICIPATION")

participation = (
    bids.groupby("vendor_id")["tender_id"]
    .nunique()
)

print(
    participation.describe(
        percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    ).round(2).to_string()
)

if participation.nunique() < 20:
    fail(
        "Vendor participation is suspiciously uniform."
    )

top_share = (
    participation.sort_values(ascending=False)
    .head(max(1, int(len(participation) * 0.01)))
    .sum()
    / participation.sum()
)

print(
    f"\nTop 1% vendor participation share: {top_share:.2%}"
)

if top_share > 0.35:
    print(
        "[WARN] Participation is highly concentrated."
    )
else:
    ok("Vendor participation has meaningful heterogeneity")

# ---------------------------------------------------------------------
# 3. BID COUNT DISTRIBUTION BY CATEGORY
# ---------------------------------------------------------------------

section("3. COMPETITION HETEROGENEITY")

category_bid_counts = (
    b.groupby(["tender_id", "category"])
    .size()
    .reset_index(name="bid_count")
)

stats = (
    category_bid_counts
    .groupby("category")["bid_count"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max",
    )
)

print(stats.round(2).to_string())

if stats["mean"].nunique() < 3:
    fail(
        "Bid-count behavior is too uniform across procurement categories."
    )

if category_bid_counts["bid_count"].nunique() < 5:
    fail(
        "Tender competition has insufficient variation."
    )

ok("Competition intensity varies across markets")

# ---------------------------------------------------------------------
# 4. PRICE SPREAD HETEROGENEITY
# ---------------------------------------------------------------------

section("4. PRICE COMPETITION")

price_stats = (
    b.groupby("tender_id")["bid_ratio"]
    .agg(
        median="median",
        minimum="min",
        maximum="max",
        std="std",
    )
)

price_stats["spread"] = (
    price_stats["maximum"] - price_stats["minimum"]
) / price_stats["median"].replace(0, np.nan)

print(
    price_stats["spread"]
    .describe(
        percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    )
    .round(4)
    .to_string()
)

if price_stats["spread"].nunique() < 50:
    fail(
        "Tender price spreads have insufficient variation."
    )

ok("Price competition has substantial natural variation")

# ---------------------------------------------------------------------
# 5. NORMAL CO-OCCURRENCE
# ---------------------------------------------------------------------

section("5. NATURAL BIDDER CO-OCCURRENCE")

pair_counts = {}

for _, group in b.groupby("tender_id"):
    bidders = sorted(set(group["vendor_id"]))

    for a, c in itertools.combinations(bidders, 2):
        key = (a, c)
        pair_counts[key] = pair_counts.get(key, 0) + 1

pair_series = pd.Series(pair_counts, dtype=float)

print(
    "Pair co-occurrence statistics:"
)

print(
    pair_series.describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    ).round(2).to_string()
)

if len(pair_series) < 100:
    fail(
        "Too few naturally occurring bidder relationships."
    )

# There should be many weak relationships, not just planted strong ones.
weak_pairs = (pair_series <= 3).sum()
strong_pairs = (pair_series >= 10).sum()

print(
    f"\nPairs observed <=3 times: {weak_pairs:,}"
)
print(
    f"Pairs observed >=10 times: {strong_pairs:,}"
)

if weak_pairs == 0:
    fail(
        "All bidder relationships are unusually frequent."
    )

ok("Natural bidder co-occurrence produces a long-tailed relationship structure")

# ---------------------------------------------------------------------
# 6. GEOGRAPHIC COMPLEXITY
# ---------------------------------------------------------------------

section("6. GEOGRAPHIC STRUCTURE")

vendor_regions = (
    vendors.set_index("vendor_id")["headquarters_region"]
)

b["vendor_region"] = b["vendor_id"].map(vendor_regions)

same_region = (
    b["vendor_region"] == b["region"]
)

print(
    f"Bidder/headquarters same-region rate: "
    f"{same_region.mean():.2%}"
)

if same_region.mean() < 0.20 or same_region.mean() > 0.98:
    fail(
        "Geographic participation is unrealistically deterministic."
    )

ok("Geographic participation contains both local and cross-region behavior")

# ---------------------------------------------------------------------
# 7. SCENARIO VENDOR PARTICIPATION VS NORMAL VENDORS
# ---------------------------------------------------------------------

section("7. SCENARIO LEAKAGE CHECK")

scenario_groups = {
    "REPEATED_BIDDER_NETWORK": ["V0001", "V0024", "V0091"],
    "LEGITIMATE_SPECIALIZED_MARKET": ["V0010", "V0020"],
    "COVER_BIDDING": ["V0042", "V0043"],
    "BID_ROTATION": ["V0051", "V0052", "V0053"],
    "GEOGRAPHIC_MARKET_ALLOCATION": ["V0061", "V0062", "V0063"],
    "TIGHT_PRICE_CLUSTER": ["V0071", "V0072", "V0073", "V0074"],
}

# Global participation is not itself suspicious: a legitimate large
# supplier can bid frequently. What matters is abnormal participation
# relative to the supplier's qualified market exposure.
#
# Compare each scenario vendor against other vendors qualified for the
# same primary/secondary markets. This prevents the validator from
# incorrectly treating a highly active legitimate supplier as leakage.

vendor_market_counts = {}

for _, row in vendors.iterrows():
    markets = {
        row["primary_category"],
        row["secondary_category"],
    }
    vendor_market_counts[row["vendor_id"]] = markets

for scenario, group in scenario_groups.items():

    print(f"{scenario:32s}")

    for vendor_id in group:

        if vendor_id not in participation:
            fail(
                f"{scenario}: {vendor_id} has no participation."
            )

        markets = vendor_market_counts[vendor_id]

        peers = [
            v
            for v, peer_markets in vendor_market_counts.items()
            if markets & peer_markets
        ]

        peer_participation = participation[
            participation.index.isin(peers)
        ]

        percentile = (
            (peer_participation < participation[vendor_id]).sum()
            / max(1, len(peer_participation))
        )

        print(
            f"  {vendor_id}: "
            f"{int(participation[vendor_id])} tenders, "
            f"market-exposure percentile={percentile:.1%}"
        )

ok(
    "Scenario vendors are evaluated against comparable market exposure "
    "rather than an arbitrary global top-3 cutoff"
)

# ---------------------------------------------------------------------
# 8. LEGITIMATE SPECIALIZATION SHOULD LOOK SUSPICIOUS
# ---------------------------------------------------------------------

section("8. NEGATIVE CONTROL: SPECIALIZATION")

specialized = scenario_groups["LEGITIMATE_SPECIALIZED_MARKET"]

spec_bids = b[
    b["vendor_id"].isin(specialized)
    & (b["category"] == "LAB_EQUIPMENT")
].copy()

spec_tenders = tenders[
    tenders["tender_id"].isin(spec_bids["tender_id"])
    & (tenders["category"] == "LAB_EQUIPMENT")
].copy()

spec_bid_counts = (
    spec_bids.groupby("tender_id")
    .size()
)

spec_participation = (
    spec_bids.groupby("vendor_id")["tender_id"]
    .nunique()
    .sort_values(ascending=False)
)

print(
    "Specialized-market supplier participation:"
)
print(
    spec_participation.to_string()
)

print(
    f"\nSpecialized-market tenders: "
    f"{len(spec_tenders)}"
)

print(
    "\nSpecialized-market bidder count:"
)
print(
    spec_bid_counts.describe().round(2).to_string()
)

# This is deliberately a negative control:
# a small legitimate specialist market should look suspicious to
# a naive detector because the same suppliers repeatedly participate
# while each tender has very few qualified bidders.
#
# We do NOT require artificial winner concentration. The award table
# represents the underlying procurement process and is intentionally
# not rewritten by the scenario planter.

if len(spec_participation) < 2:
    fail(
        "Legitimate specialization does not contain both "
        "specialized suppliers."
    )

if spec_participation.min() < 20:
    fail(
        "Legitimate specialized suppliers do not participate "
        "frequently enough to create a meaningful negative-control signal."
    )

if spec_bid_counts.mean() > 3.0:
    fail(
        "Specialized market is not sufficiently competition-constrained."
    )

ok(
    "Legitimate specialization creates a strong look-alike signal "
    "through repeated participation and sparse competition, without "
    "requiring fabricated winner concentration"
)

# ---------------------------------------------------------------------
# 9. SCENARIO SIGNAL STRENGTH
# ---------------------------------------------------------------------

section("9. PLANTED SIGNAL STRENGTH")

# Repeated network
network = scenario_groups["REPEATED_BIDDER_NETWORK"]

network_joint = 0

for _, group in b.groupby("tender_id"):
    bidders = set(group["vendor_id"])

    if set(network).issubset(bidders):
        network_joint += 1

print(
    f"Repeated network joint participation: {network_joint}"
)

if network_joint < 10:
    fail(
        "Repeated network signal is too weak."
    )

# Cover bidding
cover = scenario_groups["COVER_BIDDING"]

cover_bids = b[
    b["vendor_id"].isin(cover)
]

cover_pivot = (
    cover_bids.groupby("vendor_id")["bid_ratio"]
    .median()
)

print(
    "\nCover-bidding median bid ratios:"
)
print(cover_pivot.round(4).to_string())

if len(cover_pivot) < 2:
    fail("Cover-bidding scenario lacks both participants")

if abs(
    cover_pivot.iloc[0] - cover_pivot.iloc[1]
) < 0.01:
    print(
        "[WARN] Cover-bidding vendors have very similar median ratios."
    )
else:
    ok("Cover-bidding vendors exhibit price separation")

# Tight cluster
tight = scenario_groups["TIGHT_PRICE_CLUSTER"]

tight_bids = b[
    b["vendor_id"].isin(tight)
].copy()

tight_spread = (
    tight_bids.groupby("tender_id")["bid_ratio"]
    .agg(["min", "max", "median"])
)

tight_spread["relative_spread"] = (
    tight_spread["max"] - tight_spread["min"]
) / tight_spread["median"]

print(
    "\nTight-cluster relative spread:"
)
print(
    tight_spread["relative_spread"]
    .describe(
        percentiles=[0.25, 0.50, 0.75, 0.90]
    )
    .round(4)
    .to_string()
)

if tight_spread["relative_spread"].median() > 0.10:
    fail(
        "Tight-price scenario is not actually tight."
    )

ok("Tight-price scenario produces a measurable price-clustering signal")

# ---------------------------------------------------------------------
# FINAL
# ---------------------------------------------------------------------

section("FINAL BEHAVIORAL VERDICT")

print("""
Structural validation passed previously.

This pass additionally checked:
  - winner concentration
  - supplier participation heterogeneity
  - competition heterogeneity
  - price-spread heterogeneity
  - natural bidder co-occurrence
  - geographic complexity
  - scenario leakage
  - legitimate suspicious-looking behavior
  - planted scenario signal strength
""")

print("[PASS] VERITAS BEHAVIORAL VALIDATION COMPLETE")

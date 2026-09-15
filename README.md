# VERITAS

## Procurement Intelligence for Finding Evidence, Not Making Accusations

Veritas is an investigator-facing public procurement anomaly detection system.

Instead of trying to predict whether a vendor is "fraudulent", Veritas searches for procurement behavior that is unusually difficult to explain given its surrounding context.

It analyzes:

- Vendors
- Tenders
- Bids
- Prices
- Winners
- Procurement markets
- Geography
- Supplier relationships
- Historical behavior
- Temporal patterns

The output is an **Investigation Priority**, backed by inspectable evidence.

> **An anomaly is not an accusation. It is a reason to look closer.**

The central question is not:

> "Is this vendor fraudulent?"

It is:

> **"What behavior is unusually difficult to explain given the context?"**

> **Technical deep dive:** [Backend Algorithms & Mathematics](backend/README.md) — detailed formulas, statistical models, scoring logic, regime detection, convergence and graph construction.


---

# How Veritas Works

    PROCUREMENT DATA
           |
           v
    DATA NORMALIZATION
           |
           v
    CONTEXTUAL PEER UNIVERSE
           |
      +----+----+----+
      |    |    |    |
      v    v    v    v
    NETWORK ECONOMIC TEMPORAL
      |    |    |
      +----+----+
           |
    SUPPORTING SIGNALS
      /           \
     v             v
    GEOGRAPHY   CONCENTRATION
           |
           v
    EVIDENCE CONVERGENCE
           |
           v
    INVESTIGATION PRIORITY
           |
      +----+----+
      |         |
      v         v
    EVIDENCE   EVIDENCE
     CASE       GRAPH

The core idea is that procurement anomalies are usually **relational and contextual**, rather than isolated events.

A single unusual observation may be completely legitimate.

Multiple independent mechanisms pointing toward the same behavior are much more useful to an investigator.

---

# Detection Algorithms

Veritas currently uses six main evidence families.

---

## 1. Conditional Co-Bidding

### What it asks

> Do two suppliers bid together substantially more often than we would expect given the markets in which they operate?

Simply counting shared tenders is misleading.

For example:

    Vendor A and Vendor B bid together 20 times

is not necessarily suspicious if they operate in the same specialist market.

Veritas therefore builds a contextual expectation using procurement characteristics such as:

- Category
- Region
- Procurement method
- Vendor participation frequency
- Market size

For vendors `i` and `j` in market `m`:

    E[i,j,m] = n[i,m] * n[j,m] / N[m]

where:

- `n[i,m]` = tenders where vendor `i` participated
- `n[j,m]` = tenders where vendor `j` participated
- `N[m]` = total tenders in the market

The observed-to-expected enrichment is:

    R[i,j] = Observed co-bidding / Expected co-bidding

Example:

    1.00x   approximately expected
    2.00x   notable
    5.00x   strong
    8.02x   highly unusual

The relationship engine additionally considers:

- Shared-market overlap
- Market concentration
- Conditional win effects
- Persistence over time

### Statistical surprise

The detector also evaluates the probability of observing the overlap under an independence baseline using a hypergeometric model.

The resulting surprise is represented as:

    Surprise = -log10(p)

This provides a statistical measure of how unexpected the observed relationship is.

---

# 2. Economic / Bid-Price Compression

### What it asks

> Are the bids unusually close together compared with comparable procurements?

For every tender, Veritas examines the distribution of submitted bids.

Important measurements include:

- Minimum bid
- Maximum bid
- Median bid
- Bid spread
- Relative spread
- Coefficient of variation
- Bid / estimated-value relationships

Relative bid spread:

    S = (max(bids) - min(bids)) / median(bids)

Coefficient of variation:

    CV = standard deviation(bids) / mean(bids)

The tender is then compared with a contextual peer group rather than the entire dataset.

The peer group is based on procurement characteristics such as:

    Category + Region + Procurement Method

### Robust statistics

Procurement data can contain extreme values, so Veritas uses robust statistics rather than relying only on mean and standard deviation.

The robust deviation is based on median absolute deviation:

    robust_z =
        (peer_median - observed_value)
        --------------------------------
               1.4826 * MAD

The detector combines:

1. Robust deviation from peers
2. Empirical percentile within the peer group
3. A bounded bidder-count factor

More bidders provide more information about the shape of the competitive distribution.

A tight bid cluster is therefore treated as an **economic signal**, not proof of collusion.

---

# 3. Temporal Vendor Behavior

### What it asks

> Has a vendor's behavior changed materially over time?

Veritas constructs vendor-level historical behavior from procurement participation.

It tracks signals such as:

- Participation
- Win rate
- Behavioral changes
- Network exposure
- Market activity

The vendor timeline is divided into earlier and later periods.

    HISTORICAL BEHAVIOR
            |
            v
       EARLY PERIOD
            |
            | compare
            v
        LATE PERIOD
            |
            v
       BEHAVIORAL SHIFT

The system measures changes in:

- Win rate
- Participation
- Market behavior
- Network exposure

### Sample-size protection

Small samples can produce misleading percentages.

For example:

    1 win / 1 tender = 100%

should not be treated the same as:

    100 wins / 100 tenders = 100%

Veritas therefore applies a sample-size confidence factor and empirical-Bayes-style shrinkage to win-rate estimates.

Conceptually:

    adjusted_rate =
        (observed_wins + prior_strength * prior_rate)
        ----------------------------------------------
             observations + prior_strength

This prevents tiny samples from producing extreme evidence.

---

# 4. Winner Rotation

### What it asks

> Do the same suppliers repeatedly alternate as winners within a procurement market?

Tenders are ordered chronologically within:

    Category + Region

Winner transitions are then examined.

Example:

    A -> B
    B -> A
    A -> B
    B -> A

For every vendor pair, Veritas measures how frequently that pair accounts for winner transitions.

The detector requires a sufficiently long sequence and meaningful number of transitions.

Rotation is therefore treated as:

    potential evidence

rather than:

    automatic collusion

---

# 5. Geographic Behavior

### What it asks

> Does a supplier's geographic winning behavior differ significantly from where it participates?

Veritas compares:

    Regional participation share

against:

    Regional win share

The geographic divergence is:

    Delta_geo =
        |regional win share - regional participation share|

A large difference can indicate unusual geographic allocation or other market structure.

Geographic evidence is intentionally treated as **supporting evidence**, because geographic specialization can be legitimate.

---

# 6. Winner Concentration

### What it asks

> Is a procurement market unusually dominated by one supplier?

For every:

    Category + Region

peer group, Veritas calculates winner concentration.

Winner share:

    Share[i] =
        Wins[i] / Total Tenders

It also calculates the Herfindahl-Hirschman Index:

    HHI = sum(Share[i]^2)

High concentration alone is not considered proof of misconduct.

A specialist procurement market can legitimately have one dominant supplier.

Therefore concentration is a **low-weight supporting signal**.

---

# Behavioral Relationship Engine

The behavioral engine turns raw supplier interactions into contextual relationships.

Instead of asking:

    Did A and B bid together?

Veritas asks:

    Given where A operates,
    where B operates,
    how frequently they participate,
    how many tenders exist,
    and what the procurement context is,

    how often should A and B
    reasonably be expected to appear together?

This produces contextual relationship features including:

- Conditional co-bidding enrichment
- Shared-market overlap
- Market concentration
- Number of shared markets
- Conditional win effects
- Relationship persistence

---

# Conditional Win Effect

Veritas also examines whether the presence of one supplier changes another supplier's probability of winning.

For vendor `A` and partner `B`:

    U(A|B) =
        P(A wins | B participates)
        -
        P(A wins | B does not participate)

This helps distinguish:

    "These suppliers frequently participate together"

from:

    "The relationship is associated with a meaningful
     change in competitive outcomes."

Small samples are down-weighted.

---

# Relationship Persistence

A relationship appearing once is different from a relationship repeatedly appearing over time.

Veritas therefore tracks relationship persistence using:

- Time span
- Active calendar months
- Repeated observations

Persistence strengthens an existing observation.

It does **not** create independent evidence by itself.

---

# Evidence Convergence

This is the core of the Veritas scoring system.

A naive anomaly detector might do:

    features -> anomaly score -> alert

Veritas instead does:

    detectors
        |
        v
      context
        |
        v
     evidence
        |
        v
    mechanisms
        |
        v
    convergence
        |
        v
    investigation priority

The system deliberately does **not** simply add every alert together.

Why?

Because evidence can be correlated.

For example:

    20 co-bidding observations

are not necessarily:

    20 independent pieces of evidence

---

# Evidence Hierarchy

Veritas groups evidence into mechanisms.

## Core mechanisms

    NETWORK
        |
        +-- Co-bidding
        +-- Winner rotation

    ECONOMIC
        |
        +-- Bid-price behavior

    TEMPORAL
        |
        +-- Vendor behavioral change

## Supporting mechanisms

    GEOGRAPHIC
        |
        +-- Geographic divergence

    CONCENTRATION
        |
        +-- Winner concentration

The primary mechanisms are:

    Network
    Economic
    Temporal

Geographic behavior and winner concentration provide contextual support.

---

# Evidence Normalization

Each detector produces evidence strength on a common bounded scale:

    0 <= strength <= 1

This allows fundamentally different signals to participate in the same scoring framework.

Detector-specific statistics are converted into comparable evidence strength before convergence.

---

# Contextual Suppression

Context is used to reduce false positives.

If unusual behavior has a strong legitimate explanation, its contribution can be reduced.

Conceptually:

    Adjusted Evidence
        =
    Raw Evidence
        * Context Factor
        * Persistence Factor

Context factors are bounded so contextualization can suppress an observation without automatically deleting strong independent evidence.

This is especially important for legitimate specialist markets.

---

# Diminishing Returns

Evidence within the same family is aggregated using diminishing returns.

Suppose a vendor has:

    Relationship A = very strong
    Relationship B = medium
    Relationship C = medium
    Relationship D = medium

The strongest observation carries the majority of the information.

Subsequent observations contribute progressively less.

Conceptually:

    Family Score =
        strongest evidence
        + small contribution from #2
        + smaller contribution from #3
        + smaller contribution from #4

This prevents:

    20 mediocre observations

from automatically overpowering:

    1 extremely strong observation

---

# Independent Mechanism Convergence

The system then asks:

> How many different mechanisms are pointing in the same direction?

The core mechanisms are:

    Network
    Economic
    Temporal

The convergence bonus is intentionally modest:

    1 mechanism  -> 1.00x
    2 mechanisms -> 1.14x
    3 mechanisms -> 1.28x

This means:

    Network

is useful.

But:

    Network + Economic

is more compelling.

And:

    Network + Economic + Temporal

is stronger still.

The system does not allow convergence to grow without bound.

---

# Investigation Priority

The final output is:

    INVESTIGATION PRIORITY

It is **not**:

    FRAUD PROBABILITY

and it is not a legal determination.

The scoring hierarchy is:

### No core mechanism

Supporting evidence can produce a low-level investigation lead.

### One core mechanism

A meaningful:

    Network
    Economic
    Temporal

signal can produce a case.

### Multiple core mechanisms

The main convergence model is activated.

Examples:

    Network + Economic
    Network + Temporal
    Economic + Temporal
    Network + Economic + Temporal

The strongest mechanism dominates, additional mechanisms corroborate it, and supporting evidence contributes without overwhelming the primary signal.

The final priority is bounded:

    0 <= Investigation Priority <= 100

---

# What Every Case Contains

A case can expose:

- Investigation priority
- Evidence family count
- Evidence families
- Weighted evidence strength
- Strongest family strength
- Convergence strength
- Evidence diversity
- Persistence strength
- Context factor
- Core evidence families
- Supporting evidence families
- Evidence explanations
- Source identifiers

This makes the result traceable back to the underlying observations.

---

# Evidence Graph

Veritas converts procurement relationships into an evidence graph.

Core relationships include:

    VENDOR
       |
       +-- CO_BID --> VENDOR
       |
       +-- BID_ON --> TENDER
       |
       +-- PARTICIPATES_IN --> MARKET

    TENDER
       |
       +-- OWNED_BY --> DEPARTMENT

    DEPARTMENT
       |
       +-- BELONGS_TO --> ORGANIZATION

Co-bidding relationships are enriched with behavioral information.

An investigator can therefore move from:

    Investigation Priority
            |
            v
         Evidence
            |
            v
     Supplier Relationship
            |
            v
      Tender / Market Context

rather than receiving a black-box score.

---

# End-to-End Example

Consider supplier `V0071`.

The system may observe:

    V0071 <-> V0074
    8.02x expected co-bidding

and:

    V0071 <-> V0073
    6.59x expected co-bidding

and:

    V0071 <-> V0072
    5.21x expected co-bidding

These relationships are grouped into a network rather than treated as unrelated alerts.

The system can then independently identify economic or temporal evidence.

Conceptually:

                 V0074
                   |
                 8.02x
                   |
                   v
    V0073 -----> V0071 <----- V0072
     6.59x                    5.21x
                   |
                   v
            Economic evidence
                   |
                   v
            Temporal evidence
                   |
                   v
        INVESTIGATION PRIORITY

The investigator can then inspect the evidence behind the priority.

---

# Why Veritas Is Different

A generic anomaly detector usually asks:

    "What is statistically unusual?"

Veritas asks:

    "What is unusual relative to the right context,
     does it persist,
     what relationships explain it,
     and do independent mechanisms converge?"

This distinction matters.

A legitimate specialist supplier may:

- Win frequently
- Operate in a narrow geography
- Bid repeatedly with the same suppliers
- Submit similar prices

Those facts alone should not automatically generate a serious investigation.

Veritas attempts to distinguish:

    UNUSUAL

from:

    UNUSUAL + CONTEXTUALLY UNEXPLAINED

---

# Data

The current prototype uses synthetic procurement data.

Current dataset:

    550 vendors
    2,500 tenders
    13,512 bids

The data is synthetic and exists to exercise the analytical pipeline.

It does not represent real procurement records or real-world allegations.

---

# Generated Analytical Artifacts

The pipeline produces analytical datasets under:

    data/generated/

Important artifacts include:

    vendors.csv
    tenders.csv
    bids.csv

    contextual_cooccurrence.csv
    behavioral_relationships.csv

    bid_spread.csv
    economic_evidence.csv

    temporal_behavior.csv
    vendor_regime_risk.csv

    winner_rotation.csv
    geographic_behavior.csv
    winner_concentration.csv

    evidence_graph.csv
    converged_evidence.csv
    investigation_cases.csv

The final application primarily consumes:

    converged_evidence.csv
    investigation_cases.csv
    evidence_graph.csv

---

# Backend Architecture

    backend/app/

    generator.py
        Synthetic procurement data generation

    detectors.py
        Core procurement anomaly detectors

    behavioral.py
        Contextual supplier relationships

    regimes.py
        Vendor time-series regime analysis

    regime_risk.py
        Procurement regime-risk features

    economics.py
        Bid-price and economic analysis

    scoring.py
        Evidence normalization/scoring utilities

    graph.py
        Evidence graph construction

    convergence.py
        Evidence fusion and investigation cases

    main.py
        FastAPI investigation API

---

# API

The backend exposes:

    GET /api/health

    GET /api/summary

    GET /api/cases

    GET /api/cases/{vendor_id}

    GET /api/cases/{vendor_id}/evidence

    GET /api/cases/{vendor_id}/graph

    GET /api/vendors

    GET /api/vendors/{vendor_id}

The cases endpoint supports:

- Pagination
- Minimum investigation priority
- Evidence-family filtering

---

# Running Veritas

## Backend

    cd /home/dan/Datum-Veritas
    source .venv/bin/activate
    uvicorn backend.app.main:app --reload

## Frontend

Open another terminal:

    cd /home/dan/Datum-Veritas/frontend
    npm install
    npm run dev

Then open the Vite URL, normally:

    http://localhost:5173

The application uses the analytical artifacts already generated in:

    data/generated/

---

# Rebuilding the Pipeline

From the repository root:

    cd /home/dan/Datum-Veritas
    source .venv/bin/activate

    python -m backend.app.detectors
    python -m backend.app.behavioral
    python -m backend.app.regimes
    python -m backend.app.regime_risk
    python -m backend.app.economics
    python -m backend.app.graph
    python -m backend.app.convergence

The analytical modules contain validation/self-test logic to catch malformed outputs and pipeline regressions.

---

# Technology

## Backend

- Python
- FastAPI
- Pandas
- NumPy
- SciPy
- NetworkX

## Frontend

- React
- TypeScript
- Vite

---

# Design Principles

### Context before suspicion

Compare behavior against the correct procurement population.

### Relationships matter

Supplier behavior is often relational rather than isolated.

### Time matters

Persistent relationships and behavioral changes are more informative than single events.

### Correlated evidence is not independent evidence

Repeated observations from one detector family receive diminishing returns.

### Legitimate explanations matter

Specialization and normal market structure should suppress false positives.

### Scores must be inspectable

Every investigation priority should lead back to concrete evidence.

### Humans remain in the loop

Veritas prioritizes where investigators should look.

It does not decide whether misconduct occurred.

---

# Core Idea

    Don't automate the verdict.

    Automate the search for the evidence.

**VERITAS** turns procurement data from a haystack into an investigation queue while keeping the underlying evidence visible to the investigator.

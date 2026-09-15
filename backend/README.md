# Veritas Backend — Algorithms & Mathematics

This document explains the statistical, behavioral, economic, temporal, and graph algorithms implemented by the Veritas backend.

Veritas is not a black-box fraud classifier.

It is a multi-stage evidence system:

    PROCUREMENT DATA
          |
          v
    CONTEXTUALIZATION
          |
          +----------------+----------------+
          |                |                |
          v                v                v
       NETWORK          ECONOMIC         TEMPORAL
          |                |                |
          +----------------+----------------+
                           |
                    SUPPORTING SIGNALS
                           |
                           v
                 EVIDENCE CONVERGENCE
                           |
                           v
                 INVESTIGATION PRIORITY
                           |
                    +------+------+
                    |             |
                    v             v
                EVIDENCE      EVIDENCE
                   CASE          GRAPH

The core principle is:

    UNUSUAL != FRAUD

The system searches for behavior that is unusually difficult to explain given its procurement context.

---

# 1. Procurement Context

The main contextual market key is:

    category | region | procurement_method

This creates peer groups so that vendors are compared against relevant procurement behavior rather than against the entire dataset.

For example, a supplier operating exclusively in a specialist regional market should not be compared directly against a supplier bidding across every category.

Context is therefore applied before evidence is interpreted.

---

# 2. Bidder Co-Occurrence

Implemented in:

    backend/app/detectors.py

## Objective

Find vendor pairs that appear together more frequently than expected.

For vendor i:

    n_i = number of tenders containing vendor i

For vendor j:

    n_j = number of tenders containing vendor j

Let:

    N = total number of tenders

Under an independence baseline:

    E[i,j] = (n_i * n_j) / N

where E is the expected number of shared tenders.

The enrichment ratio is:

    R[i,j] = Observed[i,j] / E[i,j]

Interpretation:

    R ~= 1       approximately expected
    R > 1        more frequent than expected
    R >> 1       unusually frequent

The detector requires a minimum number of shared tenders.

---

# 3. Hypergeometric Co-Bidding Significance

The same detector also calculates an exact overlap probability under the independence model.

The probability of observing at least k shared tenders is:

    p = sum(k=O to min(n_i,n_j))
        C(n_i,k) C(N-n_i,n_j-k)
        ---------------------------
              C(N,n_j)

where:

    O = observed shared tenders
    C(a,b) = binomial coefficient

The resulting statistical surprise is:

    Surprise = -log10(p)

Large surprise means the observed overlap is statistically difficult to explain under the independence baseline.

This is NOT a probability of collusion.

It is a probability under a specific statistical null model.

---

# 4. Contextual Supplier Relationships

Implemented in:

    backend/app/behavioral.py

Raw co-occurrence can be misleading because suppliers may naturally operate in the same markets.

For market m:

    N_m   = number of tenders in market m
    n_A,m = tenders where A participates
    n_B,m = tenders where B participates

Expected shared participation:

    E[A,B,m] = (n_A,m * n_B,m) / N_m

Observed shared participation:

    O[A,B,m]

Market-level enrichment:

    R[A,B,m] = O[A,B,m] / E[A,B,m]

Across markets:

    total_observed = sum(O[A,B,m])

    total_expected = sum(E[A,B,m])

Therefore:

    Conditional Enrichment =
        total_observed / total_expected

This is the main network statistic used by the behavioral engine.

---

# 5. Market Overlap

The relationship engine also calculates a Jaccard-style overlap:

    overlap =
        shared / (n_A + n_B - shared)

This measures how much of the suppliers' combined activity overlaps.

Two suppliers can therefore have:

    high shared count
    but low proportional overlap

or:

    moderate shared count
    but extremely high proportional overlap

These describe different behaviors.

---

# 6. Market Concentration of Relationships

For each shared market:

    market_share_m =
        shared_m / total_shared

The relationship concentration is:

    concentration =
        max(market_share_m)

This helps distinguish:

    repeated interaction inside one specialist market

from:

    repeated interaction across multiple procurement markets.

A highly concentrated relationship may have a legitimate specialist-market explanation.

---

# 7. Conditional Win Effect

Implemented in:

    backend/app/behavioral.py

For vendor A and partner B:

    U(A|B) =
        P(A wins | B participates)
        -
        P(A wins | B does not participate)

This measures conditional association between a supplier relationship and competitive outcomes.

It is NOT interpreted causally.

The purpose is to distinguish:

    "A and B frequently bid together"

from:

    "A and B frequently bid together and
     A's competitive outcome changes when B is present."

Minimum-observation requirements prevent tiny samples from creating extreme effects.

---

# 8. Temporal Vendor Regime Detection

Implemented in:

    backend/app/regimes.py

The temporal engine asks:

    "Has this vendor's behavior changed relative to its own history?"

It builds a monthly vendor panel with:

    participation_intensity
    win_rate
    bid_position
    market_breadth
    network_exposure

The system compares earlier active observations with later active observations.

---

# 9. Participation Intensity

For vendor i in month t:

    participation_intensity =
        vendor_tenders_t / monthly_tenders_t

This measures the vendor's share of procurement activity during that month.

---

# 10. Normalized Bid Position

Bid rank is normalized by the number of bidders:

    normalized_position =
        (rank - 1) / (bidder_count - 1)

Therefore:

    0 = strongest position
    1 = weakest position

This makes bid position comparable across tenders with different numbers of bidders.

---

# 11. Market Breadth

For a vendor:

    market_breadth =
        distinct_markets / tenders

This describes how broadly the vendor operates across procurement markets.

A reduction in breadth is NOT automatically considered suspicious.

---

# 12. Temporal Network Exposure

For each month, the system builds the set of distinct suppliers that co-participated with a vendor.

Then:

    network_exposure =
        distinct_network_partners / vendor_tenders

This makes network behavior explicitly time-dependent.

A vendor can therefore experience a regime change in its relationship environment.

---

# 13. Empirical-Bayes-Style Win-Rate Shrinkage

Raw win rates are unstable for small samples.

Instead of trusting:

    wins / tenders

the system uses:

    p_hat =
        (s + k*p_0) / (n + k)

where:

    s   = observed wins
    n   = observations
    p_0 = long-run reference win rate
    k   = prior strength

The implementation uses:

    k = 6

Therefore:

    1 win / 1 tender

does not immediately become an unquestioned 100% behavioral rate.

This reduces small-sample volatility.

---

# 14. Robust Historical Deviations

For each temporal feature, Veritas compares recent behavior with the vendor's historical distribution.

The historical center is:

    median(x)

The primary robust scale is:

    MAD =
        median(|x - median(x)|)

Converted to a standard-deviation-like scale:

    robust_scale = 1.4826 * MAD

The implementation also considers:

    IQR / 1.349

and a feature-specific minimum scale.

The standardized deviation is:

    z =
        (recent_value - historical_median)
        / robust_scale

The result is capped:

    -8 <= z <= 8

The cap prevents zero-variance histories from creating absurd numerical values.

---

# 15. Directional Persistence

A single extreme month should not dominate the temporal signal.

For every feature, the system determines the dominant direction of the recent shift.

If:

    direction = sign(median(recent_z))

then persistence is approximately:

    persistence =
        mean((recent_z * direction) > 1)

The overall persistence score is the mean persistence across the temporal features.

This answers:

    "Is the behavior consistently shifted?"

rather than:

    "Was there one extreme observation?"

---

# 16. CUSUM-Like Persistent Shift

The regime engine also calculates a CUSUM-like statistic.

Recent standardized deviations are aligned to their dominant direction:

    aligned = z * direction

Only deviations above a 0.75 threshold contribute:

    signal_t =
        max(aligned_t - 0.75, 0)

The accumulated shift is:

    CUSUM =
        sum(signal_t) / sqrt(T)

This captures sustained directional movement.

It is deliberately described as CUSUM-like rather than a formal sequential hypothesis test.

---

# 17. Multivariate Mahalanobis Distance

Temporal behavior has multiple correlated dimensions.

For each vendor:

    z =
      [
        z_participation,
        z_win_rate,
        z_bid_position,
        z_market_breadth,
        z_network_exposure
      ]

Instead of scoring every dimension independently, Veritas measures the multivariate distance from the vendor's historical behavioral state.

The covariance matrix is estimated from historical standardized observations.

Because vendor histories can be short, covariance is shrunk toward its diagonal:

    Sigma_shrunk =
        0.80 * diag(Sigma)
        + 0.20 * Sigma
        + 0.05 * I

The Mahalanobis distance is:

    D_M =
        sqrt(
            z^T Sigma_shrunk^-1 z
        )

The pseudoinverse is used for numerical stability.

This allows combinations such as:

    win rate increased
    +
    participation increased
    +
    network exposure increased

to be recognized as a multivariate behavioral shift.

---

# 18. Temporal Regime Signal

The Mahalanobis and CUSUM values are converted to bounded signals.

Multivariate signal:

    M =
        1 - exp(-0.32 * D_M)

CUSUM signal:

    C =
        1 - exp(-0.55 * CUSUM)

The regime signal is:

    regime_signal =
        0.45*M
        + 0.30*C
        + 0.25*persistence

Historical coverage then modifies the signal.

History confidence:

    history_confidence =
        min(1, history_months / 10)

The signal is multiplied by:

    0.70 + 0.30*history_confidence

This prevents short histories from receiving the same confidence as long histories.

---

# 19. Procurement-Relevant Regime Risk

Implemented in:

    backend/app/regime_risk.py

The regime engine answers:

    "What changed?"

The risk layer asks:

    "How relevant is that change to procurement?"

Positive directional z-scores are converted into bounded relevance values:

    f(z) = z / (z + scale)

The main relevance components are:

    win-rate change
    participation change
    network exposure change
    movement toward stronger bid positions

A shrinking market footprint does not receive direct risk relevance.

---

# 20. Temporal Interaction Evidence

The system explicitly models combinations of changes.

For components a and b:

    interaction(a,b) =
        sqrt(a*b)

The main interactions are:

    win + participation       45%
    win + network             40%
    network + participation   15%

The resulting interaction strength is:

    interaction_strength =
        0.45*sqrt(win*participation)
        + 0.40*sqrt(win*network)
        + 0.15*sqrt(network*participation)

The geometric mean prevents one component from dominating when the other is essentially absent.

---

# 21. Procurement Relevance

The direct relevance score is:

    relevance =
        0.55*win
        + 0.12*participation
        + 0.08*network
        + 0.05*bid_position
        + 0.20*interaction_strength

Persistence modifies it with:

    persistence_modifier =
        0.85 + 0.15*persistence

The final temporal evidence signal is:

    risk_adjusted_regime_signal =
        regime_signal
        * (0.55 + 0.45*relevance)
        * persistence_modifier

The relevance layer therefore modifies the behavioral anomaly rather than replacing it.

---

# 22. Economic Bid Analysis

Implemented in:

    backend/app/economics.py

The economic engine analyzes the distribution of bids within each tender.

For bid amounts b:

    relative_spread =
        (max(b) - min(b)) / median(b)

Coefficient of variation:

    CV =
        standard_deviation(b) / mean(b)

It also calculates:

    median_bid / estimated_value

These describe the structure and level of the tender's bid distribution.

---

# 23. Economic Peer Groups

Each tender is compared against:

    category
    +
    region
    +
    procurement_method

rather than against every tender.

For every peer group, Veritas calculates robust baselines for:

    relative spread
    coefficient of variation
    median bid / estimated value

---

# 24. Robust Economic Residuals

The robust scale is:

    scale =
        max(
            1.4826*MAD,
            IQR/1.349,
            0.01
        )

For bid spread:

    spread_residual =
        peer_median - tender_spread

    spread_z =
        spread_residual / spread_scale

Positive spread_z means the tender is more compressed than its peers.

The same approach is used for coefficient of variation.

---

# 25. Economic Compression Signal

The compression signal is:

    compression_signal =
        (
            max(spread_z, 0)
            +
            max(cv_z, 0)
        ) / 2

It is bounded to:

    0 <= compression_signal <= 8

Estimate-relative movement is kept as supporting economic information rather than being treated as a direct collusion indicator.

---

# 26. Economic Sample Confidence

More bids provide more information about a bid distribution.

Bid-count confidence:

    sample_confidence =
        1 - exp(
            -0.28 * max(bid_count - 1, 0)
        )

Peer-group confidence:

    peer_confidence =
        1 - exp(
            -0.18 * max(peer_count - 3, 0)
        )

Both functions saturate toward one.

---

# 27. Economic Evidence Strength

Compression is converted into a bounded signal:

    compression_strength =
        1 - exp(
            -0.55 * compression_signal
        )

Estimate-relative price movement:

    price_strength =
        1 - exp(
            -0.35 * max(price_ratio_z, 0)
        )

Combined economic evidence:

    economic_strength =
        [
          0.82*compression_strength
          +
          0.18*price_strength
        ]
        *
        sample_confidence
        *
        [
          0.55 + 0.45*peer_confidence
        ]

The result is clipped to:

    [0,1]

Bid compression is therefore the primary economic signal.

---

# 28. Economic Persistence

Repeated economic anomalies for the same vendor receive additional contextual strength.

For a vendor:

    high_economic_events =
        count(economic_strength >= 0.55)

Persistence:

    economic_persistence =
        1 - exp(
            -0.35 * high_economic_events
        )

Tender-level evidence becomes:

    economic_evidence =
        economic_strength
        *
        (0.75 + 0.25*economic_persistence)

Persistence cannot create evidence when the underlying tender is normal.

---

# 29. Winner Rotation

Implemented in:

    backend/app/detectors.py

Tenders are ordered chronologically inside:

    category + region

Consecutive winners create transitions:

    A -> B
    B -> A
    A -> B
    B -> A

For each pair:

    pair_transition_share =
        pair_transitions
        /
        all_nonidentical_transitions

The detector requires a minimum sequence length and transition count.

Winner rotation is therefore supporting evidence rather than an automatic collusion conclusion.

---

# 30. Winner Concentration

For every:

    category + region

peer group:

    winner_share_i =
        wins_i / total_tenders

Veritas also calculates the Herfindahl-Hirschman Index:

    HHI =
        sum(winner_share_i^2)

Concentration is not automatically suspicious because specialized markets can legitimately have few winners.

It is therefore assigned a low evidentiary role.

---

# 31. Geographic Behavior

For a vendor, Veritas compares:

    dominant-region win share

against:

    dominant-region participation share

The divergence is:

    geographic_gap =
        |
          win_share
          -
          participation_share
        |

The gap is mapped into bounded evidence using explicit thresholds.

Geographic evidence is supporting evidence because legitimate geographic specialization is possible.

---

# 32. Evidence Normalization

Implemented in:

    backend/app/convergence.py

Every observation is converted into a common evidence record:

    vendor_id
    evidence_family
    source_id
    raw_strength
    context_factor
    persistence
    adjusted_strength
    family_weight
    explanation

Strength, context and persistence are bounded to `[0,1]`.

Observation-level adjustment:

    adjusted_strength =
        raw_strength
        *
        (0.55 + 0.45*context)
        *
        (0.80 + 0.20*persistence)

Context and persistence therefore modify an observation without becoming fake independent evidence families.

---

# 33. Evidence Families

Production family weights are:

    cooccurrence          1.00
    economic              0.95
    temporal              0.85
    winner_rotation       0.80
    geographic            0.60
    winner_concentration  0.45

These are evidentiary weights, not probabilities.

The independent mechanism groups are:

    NETWORK
        cooccurrence
        winner_rotation

    ECONOMIC
        economic

    TEMPORAL
        temporal

Supporting evidence:

    geographic
    winner_concentration

Co-bidding and winner rotation therefore do not automatically count as two independent mechanisms.

---

# 34. Diminishing Returns

Multiple observations from the same family are correlated.

Therefore family evidence is NOT summed linearly.

Sorted strengths:

    s1 >= s2 >= s3 >= ...

The production family aggregation is:

    raw_family_score =
        s1
        + 0.12*s2
        + 0.12^2*s3
        + 0.12^3*s4
        + ...

Only the strongest eight observations contribute.

The result is saturated:

    family_score =
        1 - exp(-raw_family_score)

This prevents a vendor with hundreds of ordinary observations from automatically dominating the investigation queue.

---

# 35. Family Context

Family context is calculated as a strength-weighted average:

    family_context =
        sum(raw_strength_i * context_i)
        /
        sum(raw_strength_i)

This means strong observations influence the family context more than weak observations.

---

# 36. Family Persistence

For each evidence family:

    family_persistence =
        max(observation_persistence)

This describes how persistent the strongest evidence is.

Persistence is metadata about evidence strength.

It is not another independent evidence family.

---

# 37. Core Mechanism Convergence

For each vendor:

    network_score
    economic_score
    temporal_score

are calculated.

Tiny mechanism scores below:

    0.08

are not counted as meaningful mechanisms.

Let the meaningful mechanism scores be:

    m1, m2, ..., mk

The strongest mechanism is:

    strongest = max(m_i)

The mean mechanism strength is:

    mean = average(m_i)

Core convergence is:

    core_convergence =
        0.72*strongest
        +
        0.28*mean

The strongest mechanism therefore dominates.

Additional mechanisms corroborate rather than simply adding their full score.

---

# 38. Independent Mechanism Bonus

The convergence multiplier is:

    1 mechanism  -> 1.00
    2 mechanisms -> 1.14
    3 mechanisms -> 1.28

Formally:

    mechanism_bonus =
        1 + 0.14*min(max(k-1,0),2)

The bonus is deliberately bounded.

This means:

    NETWORK

is meaningful.

But:

    NETWORK + ECONOMIC

is stronger.

And:

    NETWORK + ECONOMIC + TEMPORAL

is stronger again.

---

# 39. Contextual Suppression

The case-level context factor is derived from family context.

It is bounded:

    0.35 <= context_factor <= 1.00

For multiple-core cases:

    context_modifier =
        0.42 + 0.58*context_factor

Therefore legitimate market context can suppress an investigation priority, but cannot completely erase strong independent evidence.

---

# 40. Investigation Priority

The final score is:

    INVESTIGATION PRIORITY

It is NOT:

    fraud probability

It is a ranking mechanism for deciding where investigators should look first.

## Zero core mechanisms

Supporting evidence alone is intentionally limited.

    base =
        0.06 + 0.34*supporting_strength

    priority =
        base
        *
        (0.45 + 0.55*context_factor)
        *
        100

Then:

    priority <= 20

## One core mechanism

    base =
        0.76*strongest_core
        + 0.10*core_mean
        + 0.08*supporting_strength

    priority =
        base
        *
        (0.45 + 0.55*context_factor)
        *
        100

The result is capped at:

    priority <= 40

## Multiple core mechanisms

    base =
        0.50*strongest_core
        + 0.38*core_convergence
        + 0.12*supporting_strength

Then:

    base =
        base * mechanism_bonus

And:

    priority =
        base
        *
        (0.42 + 0.58*context_factor)
        *
        100

Finally:

    priority =
        clip(priority, 0, 100)

The result combines:

    strongest evidence
    +
    corroboration
    +
    supporting evidence
    +
    contextual adjustment

without treating every observation as independent.

---

# 41. Evidence Diversity

The number of evidence families is retained as an additional diagnostic.

Current mapping:

    1 family -> 0.35
    2 families -> 0.68
    3 families -> 0.86
    4 families -> 0.94
    5 families -> 0.98
    6 families -> 1.00

This represents breadth of evidence.

It is not a probability.

---

# 42. Evidence Graph

Implemented in:

    backend/app/graph.py

The graph represents procurement structure and supplier relationships.

Core edges:

    VENDOR -> VENDOR
        CO_BID

    VENDOR -> MARKET
        PARTICIPATES_IN

    VENDOR -> TENDER
        BID_ON

    TENDER -> DEPARTMENT
        OWNED_BY

    DEPARTMENT -> ORGANIZATION
        GOVERNANCE

The graph is an investigation-navigation layer rather than another fraud detector.

---

# 43. Graph Co-Bidding Weight

For a vendor pair with `c` shared tenders:

    weight =
        log(1+c)
        /
        log(1+C_max)

where:

    C_max = maximum observed pair count

The logarithm compresses extremely frequent relationships so they do not numerically dominate the graph.

The graph also retains the raw observation count.

---

# 44. Graph Participation Weight

For a vendor participating in `c` tenders in a market:

    weight =
        1 - exp(-0.18*c)

This saturates with participation.

A vendor participating in 100 tenders should not have an edge weight 100 times stronger than one participating in 1 tender.

---

# 45. Source Provenance

Every evidence record has a source identifier.

Examples:

    relationship:<index>
    economic:<tender_id>
    regime:<index>
    rotation:<index>
    geographic:<index>
    concentration:<index>

Co-bidding records additionally retain:

    partner_vendor_id
    enrichment

This allows the investigation UI to show the actual supplier relationship and trace the evidence back to its source observation.

---

# 46. Statistical Safeguards

Veritas deliberately includes protections against unstable or misleading evidence:

### Contextual peer groups

Behavior is compared against relevant procurement populations.

### Minimum observations

Tiny samples do not receive the same confidence as sustained observations.

### Robust statistics

Median, MAD and IQR reduce sensitivity to extreme values.

### Scale floors

Near-zero variance cannot create infinite deviations.

### Z-score caps

Temporal standardized deviations are bounded to `[-8,8]`.

### Win-rate shrinkage

Small-sample win rates are pulled toward a reference rate.

### Saturating functions

Evidence and confidence approach bounded limits instead of growing without bound.

### Diminishing returns

Repeated observations from one family have progressively smaller effects.

### Mechanism grouping

Related detectors are not falsely treated as independent mechanisms.

### Contextual suppression

Legitimate market structure can reduce evidence strength.

### Human-in-the-loop

The final result is an investigation priority, not an accusation.

---

# 47. Synthetic Scenario Validation

The benchmark contains synthetic scenarios used to test whether the analytical pipeline responds to known behavioral patterns.

Scenarios include:

    REPEATED_NETWORK
    LEGITIMATE_SPECIALIZATION
    COVER_BIDDING
    BID_ROTATION
    GEOGRAPHIC_ALLOCATION
    TIGHT_PRICE_CLUSTER

Scenario labels are used for validation/auditing.

They are not used by the anomaly detectors as evidence.

The objective is for the algorithms to discover the behavioral pattern from procurement observations.

---

# 48. Final Mathematical Model

At a high level, Veritas transforms raw procurement behavior into:

    raw observation
          |
          v
    contextual baseline
          |
          v
    statistical deviation
          |
          v
    bounded evidence
          |
          v
    contextual adjustment
          |
          v
    within-family aggregation
          |
          v
    independent mechanism convergence
          |
          v
    investigation priority

The philosophy can be summarized as:

    UNUSUAL
       +
    CONTEXTUALLY UNEXPLAINED
       +
    PERSISTENT / CORROBORATED
       =
    HIGHER INVESTIGATION PRIORITY

It does not follow:

    unusual
       =
    fraud

The backend is therefore designed to make suspicious-looking procurement behavior **searchable, explainable and challengeable** by a human investigator.

> Don't automate the verdict.
>
> Automate the search for the evidence.

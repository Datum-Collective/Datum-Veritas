import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { CaseDetail, EvidenceRecord } from "../types/api";

interface InvestigationItem {
  vendor_id: string;
  investigation_priority: number;
  evidence_families: string | string[];
  evidence_diversity: number;
  persistence_strength: number;
  context_factor: number;
}

interface InvestigationsProps {
  selectedVendor: string | null;
  onSelect: (vendorId: string) => void;
}

function families(item: InvestigationItem) {
  if (Array.isArray(item.evidence_families)) {
    return item.evidence_families;
  }

  return item.evidence_families
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

export function Investigations({
  selectedVendor,
  onSelect,
}: InvestigationsProps) {
  const [cases, setCases] = useState<InvestigationItem[]>([]);
  const [detail, setDetail] =
    useState<CaseDetail | null>(null);
  const [evidence, setEvidence] =
    useState<EvidenceRecord[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const result = await api.cases();

      if (cancelled) return;

      const payload = result as
        | InvestigationItem[]
        | { cases?: InvestigationItem[] };

      setCases(
        Array.isArray(payload)
          ? payload
          : payload.cases ?? [],
      );
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedVendor) {
      return;
    }

    const vendorId = selectedVendor;
    let cancelled = false;

    async function load() {
      try {
        const [caseData, evidenceResult] = await Promise.all([
          api.case(vendorId),
          api.evidence(vendorId),
        ]);

        if (cancelled) {
          return;
        }

        setDetail(caseData);
        setEvidence(evidenceResult.evidence ?? []);
      } catch (err) {
        if (!cancelled) {
          console.error(
            err instanceof Error
              ? err.message
              : "Unable to load case.",
          );
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [selectedVendor]);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();

    return cases
      .filter((item) =>
        !q
          ? true
          : item.vendor_id
              .toUpperCase()
              .includes(q),
      )
      .sort(
        (a, b) =>
          b.investigation_priority -
          a.investigation_priority,
      );
  }, [cases, query]);

  return (
    <section className="investigations-page">
      <div className="page-kicker">
        VERITAS / INVESTIGATIONS
      </div>

      <h1 className="page-title">
        Investigation Queue
      </h1>

      <p className="page-description">
        {cases.length} ranked leads. Selection does not
        constitute a finding of misconduct.
      </p>

      <div className="investigation-toolbar">
        <label className="search-box">
          <span className="search-label">
            SEARCH VENDOR / EVIDENCE
          </span>

          <input
            value={query}
            onChange={(event) =>
              setQuery(event.target.value)
            }
            placeholder="V0071"
            spellCheck={false}
          />
        </label>
      </div>

      <div className="investigation-layout">
        <section className="investigation-table">
          <div className="investigation-table-header">
            <span>Rank</span>
            <span>Vendor</span>
            <span>Evidence</span>
            <span>Priority</span>
          </div>

          {filtered.slice(0, 60).map((item, index) => (
            <button
              type="button"
              key={item.vendor_id}
              className={[
                "investigation-row",
                selectedVendor === item.vendor_id
                  ? "investigation-row-selected"
                  : "",
              ].join(" ")}
              onClick={() =>
                onSelect(item.vendor_id)
              }
            >
              <span className="investigation-rank">
                {String(index + 1).padStart(3, "0")}
              </span>

              <span className="investigation-vendor">
                {item.vendor_id}
              </span>

              <span className="investigation-evidence">
                {families(item).map((family) => (
                  <span
                    className="case-family"
                    key={family}
                  >
                    {evidenceFamilyLabel(family)}
                  </span>
                ))}
              </span>

              <span className="investigation-priority">
                {item.investigation_priority.toFixed(1)}
              </span>
            </button>
          ))}
        </section>

        <CaseDetailPanel
          detail={detail}
          evidence={evidence}
        />
      </div>
    </section>
  );
}

function evidenceFamilyLabel(family: string): string {
  const labels: Record<string, string> = {
    cooccurrence: "Co-bidding",
    economic: "Economic",
    temporal: "Temporal",
    geographic: "Geographic",
    winner_rotation: "Winner rotation",
    winner_concentration: "Winner concentration",
  };

  return (
    labels[family] ??
    evidenceFamilyLabel(family)
  );
}

function formatEvidence(record: EvidenceRecord) {
  const family = String(
    record.evidence_family ??
      record.family ??
      "evidence",
  );

  const explanation = String(
    record.explanation ??
      record.evidence ??
      "Evidence attached to case.",
  );

  if (family === "cooccurrence") {
    const enrichment =
      explanation.match(
        /([0-9.]+)x expected/i,
      )?.[1];

    const persistence =
      explanation.match(
        /persistence=([0-9.]+)/i,
      )?.[1];

    return {
      title: "Unusually frequent co-bidding",
      metric: enrichment
        ? `${enrichment}× expected`
        : null,
      detail:
        "This vendor participated with another supplier substantially more often than expected within comparable procurement activity.",
      persistence:
        persistence === "1.00"
          ? "Persistent relationship"
          : persistence
            ? `Persistence signal ${persistence}`
            : null,
    };
  }

  if (family === "economic") {
    const compression =
      explanation.match(
        /compression=([0-9.]+)/i,
      )?.[1];

    const bids =
      explanation.match(
        /bids=([0-9]+)/i,
      )?.[1];

    const tender =
      String(record.source_id ?? "")
        .replace(/^economic:/i, "");

    return {
      title: "Unusually tight bid dispersion",
      metric: bids
        ? `${bids} bids`
        : null,
      detail:
        "Bid prices were unusually close together relative to comparable procurements.",
      persistence: compression
        ? `Compression signal ${compression}`
        : null,
      tender: tender
        ? `Tender ${tender}`
        : null,
    };
  }

  return {
    title: explanation,
    metric: null,
    detail: null,
    persistence: null,
  };
}

function CaseDetailPanel({
  detail,
  evidence,
}: {
  detail: CaseDetail | null;
  evidence: EvidenceRecord[];
}) {
  if (!detail) {
    return (
      <aside className="case-panel case-panel-empty">
        Select an investigation case
      </aside>
    );
  }

  const investigation =
    detail.case as InvestigationItem;

  const familyList = families(investigation);

  return (
    <aside className="case-panel">
      <div className="case-panel-header">
        <div className="case-panel-case-id">
          CASE / {investigation.vendor_id}
        </div>

        <div className="case-panel-vendor">
          {detail.vendor?.legal_name ??
            investigation.vendor_id}
        </div>
      </div>

      <div className="case-panel-priority">
        <div className="case-panel-priority-label">
          Investigation priority
        </div>

        <div className="case-panel-priority-value">
          {investigation.investigation_priority.toFixed(1)}
        </div>
      </div>

      <div className="case-panel-stats">
        <Stat
          label="Diversity"
          value={`${Math.round(
            investigation.evidence_diversity * 100,
          )}%`}
        />

        <Stat
          label="Persistence"
          value={`${Math.round(
            investigation.persistence_strength * 100,
          )}%`}
        />

        <Stat
          label="Context"
          value={`${Math.round(
            investigation.context_factor * 100,
          )}%`}
        />
      </div>

      <div className="case-panel-body">
        <div className="case-panel-section">
          Evidence signals
        </div>

        {familyList.map((family) => (
          <div
            className="evidence-card"
            key={family}
          >
            <div className="evidence-card-family">
              {evidenceFamilyLabel(family)}
            </div>
          </div>
        ))}

        {evidence.length > 0 && (
          <>
            <div className="case-panel-section">
              Evidence
            </div>

            {evidence.slice(0, 5).map(
              (record, index) => {
                const formatted =
                  formatEvidence(record);

                return (
                  <div
                    className="evidence-card evidence-card-rich"
                    key={`${record.source_id ?? record.source ?? "evidence"}-${index}`}
                  >
                    <div className="evidence-card-family">
                      {evidenceFamilyLabel(
                        String(
                          record.evidence_family ??
                            record.family ??
                            "evidence",
                        ),
                      )}
                    </div>

                    {formatted.tender && (
                      <div className="evidence-card-tender">
                        {formatted.tender}
                      </div>
                    )}

                    <div className="evidence-card-title">
                      {formatted.title}
                    </div>

                    {formatted.metric && (
                      <div className="evidence-card-metric">
                        {formatted.metric}
                      </div>
                    )}

                    {formatted.detail && (
                      <div className="evidence-card-text">
                        {formatted.detail}
                      </div>
                    )}

                    {formatted.persistence && (
                      <div className="evidence-card-meta">
                        {formatted.persistence}
                      </div>
                    )}
                  </div>
                );
              },
            )}
          </>
        )}
      </div>
    </aside>
  );
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="case-panel-stat">
      <div className="case-panel-stat-label">
        {label}
      </div>

      <div className="case-panel-stat-value">
        {value}
      </div>
    </div>
  );
}

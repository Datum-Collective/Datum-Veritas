import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Summary } from "../types/api";

interface CommandCenterProps {
  onInvestigate: (vendorId: string) => void;
}


interface CaseItem {
  vendor_id: string;
  investigation_priority: number;
  evidence_families: string | string[];
}

function families(item: CaseItem) {
  if (Array.isArray(item.evidence_families)) {
    return item.evidence_families;
  }

  return item.evidence_families
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

export function CommandCenter({
  onInvestigate,
}: CommandCenterProps) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [summaryResult, casesResult] =
          await Promise.all([
            api.summary(),
            api.cases(),
          ]);

        if (cancelled) return;

        setSummary(summaryResult);

        const payload = casesResult as
          | CaseItem[]
          | { cases?: CaseItem[] };

        setCases(
          Array.isArray(payload)
            ? payload
            : payload.cases ?? [],
        );
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load command center.",
          );
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <section className="command-page">
        <div className="graph-error">{error}</div>
      </section>
    );
  }

  const ranked = [...cases]
    .sort(
      (a, b) =>
        b.investigation_priority -
        a.investigation_priority,
    )
    .slice(0, 8);

  const landscape: Array<[string, number]> = summary
    ? Object.entries(summary.evidence_families ?? {})
        .map(([family, value]) => [
          ({
            cooccurrence: "Co-bidding",
            economic: "Economic",
            temporal: "Temporal",
            geographic: "Geographic",
            winner_rotation: "Winner rotation",
          } as Record<string, string>)[family] ?? family,
          Number(value),
        ] as [string, number])
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
    : [];

  const maxLandscape =
    Math.max(...landscape.map(([, value]) => value), 1);

  return (
    <section className="command-page">
      <div className="page-kicker">
        VERITAS / COMMAND CENTER
      </div>

      <h1 className="page-title">
        Command Center
      </h1>

      <p className="page-description">
        Procurement evidence requiring closer examination.
      </p>

      <div className="metric-grid">
        <Metric
          label="Investigation cases"
          value={summary?.investigation_cases}
        />

        <Metric
          label="Multi-evidence cases"
          value={summary?.multi_family_cases}
        />

        <Metric
          label="Procurement tenders"
          value={summary?.tenders}
        />

        <Metric
          label="Evidence records"
          value={summary?.evidence_records}
        />
      </div>

      <div className="command-grid">
        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                Priority Queue
              </div>

              <div className="panel-subtitle">
                Highest-ranked investigative leads
              </div>
            </div>

            <button
              type="button"
              className="panel-action"
              onClick={() =>
                onInvestigate(
                  ranked[0]?.vendor_id ?? "",
                )
              }
            >
              Open queue
            </button>
          </div>

          <div className="case-list">
            {ranked.map((item, index) => {
              const priority =
                item.investigation_priority;

              return (
                <button
                  type="button"
                  className="command-case"
                  key={item.vendor_id}
                  onClick={() =>
                    onInvestigate(item.vendor_id)
                  }
                >
                  <span className="case-rank">
                    {String(index + 1).padStart(2, "0")}
                  </span>

                  <span>
                    <span className="case-vendor">
                      {item.vendor_id}
                    </span>

                    <span className="case-families">
                      {families(item).map((family) => (
                        <span
                          className="case-family"
                          key={family}
                        >
                          {family.replaceAll("_", " ")}
                        </span>
                      ))}
                    </span>
                  </span>

                  <span>
                    <span className="case-priority">
                      {priority.toFixed(1)}
                    </span>

                    <span className="priority-track">
                      <span
                        className="priority-fill"
                        style={{
                          display: "block",
                          width: `${Math.min(
                            priority / 40 * 100,
                            100,
                          )}%`,
                        }}
                      />
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                Evidence Landscape
              </div>

              <div className="panel-subtitle">
                Families contributing to current cases
              </div>
            </div>
          </div>

          <div className="landscape-list">
            {landscape.map(([family, value]) => (
              <div
                className="landscape-row"
                key={family}
              >
                <span className="landscape-label">
                  {family.replaceAll("_", " ")}
                </span>

                <span className="landscape-track">
                  <span
                    className="landscape-fill"
                    style={{
                      display: "block",
                      width: `${value / maxLandscape * 100}%`,
                    }}
                  />
                </span>

                <span className="landscape-value">
                  {value}
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value?: number;
}) {
  return (
    <div className="metric">
      <div className="metric-label">
        {label}
      </div>

      <div className="metric-value">
        {value === undefined
          ? "—"
          : value.toLocaleString()}
      </div>
    </div>
  );
}

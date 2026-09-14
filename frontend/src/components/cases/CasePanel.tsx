import {
  ArrowUpRight,
  Clock3,
  GitBranch,
  ShieldAlert,
} from "lucide-react";
import type { CaseDetail, EvidenceRecord } from "../../types/api";

interface CasePanelProps {
  detail: CaseDetail | null;
  evidence: EvidenceRecord[];
  onOpen: () => void;
}

export function CasePanel({
  detail,
  evidence,
  onOpen,
}: CasePanelProps) {
  if (!detail) {
    return (
      <div className="flex min-h-[420px] items-center justify-center rounded-lg border border-white/[0.07] bg-[#0c0d10] text-xs text-zinc-700">
        Select an investigation case
      </div>
    );
  }

  const { case: investigation, vendor } = detail;

  const rawInvestigation = investigation as unknown as Record<string, unknown>;

  const priority =
    typeof rawInvestigation.investigation_priority === "number"
      ? rawInvestigation.investigation_priority
      : typeof rawInvestigation.priority === "number"
        ? rawInvestigation.priority
        : 0;

  const evidenceFamilies = Array.isArray(investigation.evidence_families)
    ? investigation.evidence_families
    : typeof rawInvestigation.evidence_families === "string"
      ? rawInvestigation.evidence_families
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean)
      : [];

  const strongestFamily =
    typeof rawInvestigation.strongest_family === "string"
      ? rawInvestigation.strongest_family
      : evidenceFamilies[0];

  return (
    <div className="overflow-hidden rounded-lg border border-white/[0.07] bg-[#0c0d10]">
      <div className="flex items-start justify-between border-b border-white/[0.07] px-5 py-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] text-zinc-300">
              CASE / {investigation.vendor_id}
            </span>

            <span className="rounded border border-white/10 px-1.5 py-0.5 text-[7px] uppercase tracking-wider text-zinc-600">
              Investigation lead
            </span>
          </div>

          <div className="mt-1 text-[9px] text-zinc-700">
            Anomaly signals are investigative leads, not findings of misconduct.
          </div>
        </div>

        <button
          type="button"
          onClick={onOpen}
          className="flex items-center gap-1.5 rounded border border-white/10 px-3 py-2 text-[8px] uppercase tracking-wider text-zinc-500 hover:bg-white/[0.04] hover:text-zinc-300"
        >
          Open case
          <ArrowUpRight size={11} />
        </button>
      </div>

      <div className="grid lg:grid-cols-[220px_1fr]">
        <div className="border-b border-white/[0.07] p-5 lg:border-b-0 lg:border-r">
          <div className="text-[8px] uppercase tracking-[0.18em] text-zinc-600">
            Investigation Priority
          </div>

          <div className="mt-2 font-mono text-5xl font-semibold tracking-tight text-zinc-100">
            {priority.toFixed(1)}
          </div>

          <div className="mt-2 text-[9px] text-zinc-700">
            Relative investigative ranking
          </div>

          <div className="mt-7 space-y-3">
            <MiniStat
              label="Evidence diversity"
              value={`${Math.round(investigation.evidence_diversity * 100)}%`}
            />
            <MiniStat
              label="Persistence"
              value={`${Math.round(investigation.persistence_strength * 100)}%`}
            />
            <MiniStat
              label="Context factor"
              value={`${Math.round(investigation.context_factor * 100)}%`}
            />
          </div>
        </div>

        <div className="p-5">
          <div className="mb-4 text-[8px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
            Why this case?
          </div>

          <div className="mb-5">
            <div className="font-mono text-[11px] text-zinc-300">
              {vendor?.legal_name ?? investigation.vendor_id}
            </div>

            <div className="mt-1 text-[9px] text-zinc-700">
              {vendor?.headquarters_region ?? "Region unavailable"}
              {vendor?.vendor_type ? ` · ${vendor.vendor_type}` : ""}
            </div>
          </div>

          <div className="grid gap-2 md:grid-cols-3">
            {evidenceFamilies.map((family) => (
              <EvidenceFamily
                key={family}
                family={family}
                strongest={family === strongestFamily}
              />
            ))}
          </div>

          {evidence.length > 0 && (
            <div className="mt-5 border-t border-white/[0.06] pt-4">
              <div className="mb-3 flex items-center gap-2 text-[8px] uppercase tracking-[0.16em] text-zinc-600">
                <Clock3 size={11} />
                Recent evidence
              </div>

              <div className="space-y-2">
                {evidence.slice(0, 3).map((record, index) => (
                  <div
                    key={`${record.source ?? "evidence"}-${index}`}
                    className="rounded border border-white/[0.05] bg-white/[0.015] p-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[8px] uppercase tracking-wide text-zinc-500">
                        {String(
                          record.evidence_family ?? record.family ?? "evidence",
                        ).replaceAll("_", " ")}
                      </span>

                      <span className="font-mono text-[9px] text-zinc-600">
                        {typeof record.adjusted_strength === "number"
                          ? record.adjusted_strength.toFixed(2)
                          : typeof record.strength === "number"
                            ? record.strength.toFixed(2)
                            : "—"}
                      </span>
                    </div>

                    <div className="mt-1 text-[9px] leading-4 text-zinc-600">
                      {record.explanation ??
                        record.evidence ??
                        "Evidence record attached to investigation."}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-5 flex gap-2">
            <button
              type="button"
              onClick={onOpen}
              className="flex items-center gap-2 rounded bg-zinc-200 px-3 py-2 text-[8px] font-semibold uppercase tracking-wider text-black hover:bg-white"
            >
              <ShieldAlert size={11} />
              Investigate case
            </button>

            <button
              type="button"
              onClick={onOpen}
              className="flex items-center gap-2 rounded border border-white/10 px-3 py-2 text-[8px] uppercase tracking-wider text-zinc-500 hover:bg-white/[0.04]"
            >
              <GitBranch size={11} />
              View network
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function EvidenceFamily({
  family,
  strongest,
}: {
  family: string;
  strongest: boolean;
}) {
  return (
    <div
      className={`rounded border p-3 ${
        strongest
          ? "border-white/15 bg-white/[0.035]"
          : "border-white/[0.06] bg-white/[0.015]"
      }`}
    >
      <div className="mb-2 h-0.5 w-5 bg-zinc-500" />

      <div className="text-[9px] font-medium uppercase tracking-wide text-zinc-300">
        {family.replaceAll("_", " ")}
      </div>

      <div className="mt-1 text-[8px] leading-4 text-zinc-700">
        {strongest
          ? "Strongest contributing evidence family."
          : "Independent evidence contributing to case convergence."}
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/[0.05] pb-2">
      <span className="text-[8px] uppercase tracking-wide text-zinc-700">
        {label}
      </span>
      <span className="font-mono text-[9px] text-zinc-400">{value}</span>
    </div>
  );
}

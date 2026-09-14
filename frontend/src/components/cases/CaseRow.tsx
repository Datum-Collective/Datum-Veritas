import type { InvestigationCase } from "../../types/api";

interface CaseRowProps {
  item: InvestigationCase;
  selected?: boolean;
  onClick: () => void;
}

function familiesOf(item: InvestigationCase): string[] {
  if (Array.isArray(item.evidence_families)) {
    return item.evidence_families;
  }

  return item.evidence_families
    .split(",")
    .map((family: string) => family.trim())
    .filter(Boolean);
}

export function CaseRow({
  item,
  selected = false,
  onClick,
}: CaseRowProps) {
  const priority = item.investigation_priority;
  const families = familiesOf(item);
  const width = Math.min((priority / 40) * 100, 100);

  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "w-full border-b border-white/[0.05] px-4 py-3 text-left transition",
        selected
          ? "bg-white/[0.045]"
          : "hover:bg-white/[0.025]",
      ].join(" ")}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="font-mono text-[10px] text-zinc-300">
            {item.vendor_id}
          </div>

          <div className="mt-1 flex flex-wrap gap-1.5">
            {families.slice(0, 3).map((family: string) => (
              <span
                key={family}
                className="rounded border border-white/[0.06] px-1.5 py-0.5 text-[7px] uppercase tracking-wide text-zinc-600"
              >
                {family.replaceAll("_", " ")}
              </span>
            ))}
          </div>
        </div>

        <div className="shrink-0 text-right">
          <div className="font-mono text-[12px] text-zinc-200">
            {priority.toFixed(1)}
          </div>

          <div className="mt-1 h-px w-16 bg-zinc-900">
            <div
              className="h-px bg-zinc-500"
              style={{ width: `${width}%` }}
            />
          </div>
        </div>
      </div>
    </button>
  );
}

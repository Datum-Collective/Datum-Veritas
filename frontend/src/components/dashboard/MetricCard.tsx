interface MetricCardProps {
  label: string;
  value: number | string;
  detail?: string;
}

export function MetricCard({ label, value, detail }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-white/[0.07] bg-[#0c0d10] p-4">
      <div className="text-[8px] font-medium uppercase tracking-[0.18em] text-zinc-600">
        {label}
      </div>

      <div className="mt-2 font-mono text-2xl font-medium tracking-tight text-zinc-100">
        {value}
      </div>

      {detail && (
        <div className="mt-1 text-[9px] text-zinc-700">
          {detail}
        </div>
      )}
    </div>
  );
}

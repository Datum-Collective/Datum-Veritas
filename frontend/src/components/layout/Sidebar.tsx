import {
  Activity,
  BarChart3,
  Database,
  GitBranch,
  LayoutDashboard,
  Search,
  Shield,
  Users,
} from "lucide-react";

interface SidebarProps {
  active: string;
  onNavigate: (value: string) => void;
}

type NavItem = {
  id: string;
  label: string;
  icon: typeof LayoutDashboard;
};

const items: NavItem[] = [
  { id: "command", label: "Command Center", icon: LayoutDashboard },
  { id: "investigate", label: "Investigate", icon: Search },
  { id: "vendors", label: "Vendors", icon: Users },
  { id: "graph", label: "Evidence Graph", icon: GitBranch },
];

const systemItems: NavItem[] = [
  { id: "data", label: "Data Sources", icon: Database },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "health", label: "System Health", icon: Activity },
];

export function Sidebar({ active, onNavigate }: SidebarProps) {
  return (
    <aside className="hidden w-60 shrink-0 border-r border-white/[0.07] bg-[#0a0b0e] lg:flex lg:flex-col">
      <div className="flex h-16 items-center gap-3 border-b border-white/[0.07] px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded border border-white/10 bg-white/[0.035]">
          <Shield size={16} strokeWidth={1.7} />
        </div>

        <div>
          <div className="text-[13px] font-semibold tracking-[0.2em]">
            VERITAS
          </div>
          <div className="text-[8px] uppercase tracking-[0.2em] text-zinc-600">
            Procurement Intelligence
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3">
        <NavGroup title="Workspace" items={items} active={active} onNavigate={onNavigate} />
        <NavGroup title="System" items={systemItems} active={active} onNavigate={onNavigate} />
      </nav>

      <div className="border-t border-white/[0.07] p-4">
        <div className="flex items-center gap-2 text-[9px] uppercase tracking-[0.14em] text-zinc-600">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Evidence engine operational
        </div>
      </div>
    </aside>
  );
}

function NavGroup({
  title,
  items,
  active,
  onNavigate,
}: {
  title: string;
  items: NavItem[];
  active: string;
  onNavigate: (value: string) => void;
}) {
  return (
    <div className="mb-7">
      <div className="mb-2 px-3 text-[8px] font-semibold uppercase tracking-[0.2em] text-zinc-700">
        {title}
      </div>

      {items.map(({ id, label, icon: Icon }) => {
        const selected = active === id;

        return (
          <button
            key={id}
            type="button"
            onClick={() => onNavigate(id)}
            className={`mb-0.5 flex w-full items-center gap-3 rounded px-3 py-2.5 text-left text-[11px] transition ${
              selected
                ? "bg-white/[0.06] text-zinc-200"
                : "text-zinc-600 hover:bg-white/[0.025] hover:text-zinc-400"
            }`}
          >
            <Icon size={14} strokeWidth={1.7} />
            {label}
          </button>
        );
      })}
    </div>
  );
}

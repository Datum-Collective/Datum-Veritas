export type Page =
  | "command"
  | "investigations"
  | "graph";

interface SidebarProps {
  active: Page;
  onNavigate: (page: Page) => void;
}

const items: Array<{
  id: Page;
  label: string;
  code: string;
}> = [
  {
    id: "command",
    label: "Command Center",
    code: "01",
  },
  {
    id: "investigations",
    label: "Investigations",
    code: "02",
  },
  {
    id: "graph",
    label: "Evidence Graph",
    code: "03",
  },
];

export function Sidebar({
  active,
  onNavigate,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-label">
        INVESTIGATION
      </div>

      <nav className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`sidebar-item ${
              active === item.id
                ? "sidebar-item-active"
                : ""
            }`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="sidebar-code">
              {item.code}
            </span>

            <span>
              {item.label}
            </span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        VERITAS / EVIDENCE ENGINE
      </div>
    </aside>
  );
}

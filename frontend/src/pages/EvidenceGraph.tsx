import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";

interface GraphNode {
  id: string;
  label?: string;
  type?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  edge_type?: string;
  type?: string;
  observations?: number;
}

interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface EvidenceGraphProps {
  initialVendor?: string | null;
}

function vendorId(id: string) {
  return id.replace(/^vendor:/i, "");
}

function canonical(id: string) {
  return id.toLowerCase().startsWith("vendor:")
    ? id.toLowerCase()
    : `vendor:${id.toLowerCase()}`;
}

function isVendor(id: string) {
  return id.toLowerCase().startsWith("vendor:");
}

export function EvidenceGraph({
  initialVendor,
}: EvidenceGraphProps) {
  const [vendor, setVendor] = useState(initialVendor ?? "V0071");
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!vendor.trim()) return;

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const result = (await api.graph(vendor.trim().toUpperCase())) as GraphResponse;

        if (!cancelled) {
          setGraph(result);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load evidence graph.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [vendor]);

  /*
   * Critical distinction:
   *
   * The backend graph contains vendor→vendor CO_BID edges and
   * vendor→tender BID_ON edges.
   *
   * An investigation graph should not mix those into one topology.
   * We therefore use ONLY CO_BID relationships here.
   */
  const relationships = useMemo(() => {
    if (!graph) return [];

    const focus = canonical(vendor);

    return graph.edges
      .filter((edge) => {
        const type = (edge.edge_type ?? edge.type ?? "").toUpperCase();

        if (type !== "CO_BID") return false;

        const source = canonical(edge.source);
        const target = canonical(edge.target);

        if (!isVendor(source) || !isVendor(target)) return false;

        return source === focus || target === focus;
      })
      .sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0))
      .slice(0, 14);
  }, [graph, vendor]);

  const neighbors = useMemo(() => {
    const focus = canonical(vendor);

    const ids = new Set<string>();

    for (const edge of relationships) {
      const source = canonical(edge.source);
      const target = canonical(edge.target);

      ids.add(source === focus ? target : source);
    }

    return [...ids];
  }, [relationships, vendor]);

  const layout = useMemo(() => {
    const width = 900;
    const height = 600;

    const cx = width / 2;
    const cy = height / 2;

    const radius = Math.min(225, 125 + neighbors.length * 7);

    const nodes = neighbors.map((id, index) => {
      const angle =
        -Math.PI / 2 +
        (index / Math.max(neighbors.length, 1)) * Math.PI * 2;

      return {
        id,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
      };
    });

    return {
      width,
      height,
      cx,
      cy,
      nodes,
    };
  }, [neighbors]);

  const positions = new Map(
    layout.nodes.map((node) => [node.id, node]),
  );

  return (
    <section className="graph-page">
      <div className="page-kicker">
        VERITAS / EVIDENCE GRAPH
      </div>

      <div className="graph-heading">
        <div>
          <h1 className="page-title">
            Vendor Relationship Graph
          </h1>

          <p className="page-description">
            Co-bidding relationships surrounding the selected vendor.
            Tender participation is kept out of the relationship topology.
          </p>
        </div>

        <label className="graph-focus">
          <span className="graph-focus-label">
            FOCUS VENDOR
          </span>

          <input
            value={vendor}
            onChange={(event) =>
              setVendor(event.target.value.toUpperCase())
            }
            spellCheck={false}
          />
        </label>
      </div>

      {loading && (
        <div className="graph-status">
          LOADING RELATIONSHIP EVIDENCE...
        </div>
      )}

      {error && (
        <div className="graph-error">
          {error}
        </div>
      )}

      {!loading && !error && graph && (
        <>
          <div className="graph-meta">
            <span>
              FOCUS / {vendor}
            </span>

            <span>
              {relationships.length} DIRECT RELATIONSHIPS
            </span>

            <span>
              CO-BIDDING ONLY
            </span>
          </div>

          <div className="graph-workspace">
            <div className="graph-frame">
              <svg
                viewBox={`0 0 ${layout.width} ${layout.height}`}
                role="img"
                aria-label={`Vendor relationship graph for ${vendor}`}
              >
                <defs>
                  <pattern
                    id="graph-grid"
                    width="45"
                    height="45"
                    patternUnits="userSpaceOnUse"
                  >
                    <path
                      d="M 45 0 L 0 0 0 45"
                      fill="none"
                      stroke="#101010"
                      strokeWidth="1"
                    />
                  </pattern>
                </defs>

                <rect
                  width={layout.width}
                  height={layout.height}
                  fill="url(#graph-grid)"
                />

                {relationships.map((edge, index) => {
                  const source = canonical(edge.source);
                  const target = canonical(edge.target);

                  const neighbor =
                    source === canonical(vendor)
                      ? target
                      : source;

                  const node = positions.get(neighbor);

                  if (!node) return null;

                  const weight = Math.max(
                    0,
                    Math.min(edge.weight ?? 0, 1),
                  );

                  return (
                    <line
                      key={`${edge.source}-${edge.target}-${index}`}
                      x1={layout.cx}
                      y1={layout.cy}
                      x2={node.x}
                      y2={node.y}
                      stroke="#999"
                      strokeOpacity={0.25 + weight * 0.65}
                      strokeWidth={1 + weight * 3}
                    />
                  );
                })}

                {layout.nodes.map((node) => (
                  <g
                    key={node.id}
                    className="graph-node-clickable"
                    onClick={() =>
                      setVendor(vendorId(node.id))
                    }
                  >
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r="20"
                      fill="#080808"
                      stroke="#777"
                      strokeWidth="1"
                    />

                    <text
                      x={node.x}
                      y={node.y + 4}
                      textAnchor="middle"
                      className="graph-node-label"
                    >
                      {vendorId(node.id)}
                    </text>
                  </g>
                ))}

                <circle
                  cx={layout.cx}
                  cy={layout.cy}
                  r="37"
                  fill="#ededed"
                  stroke="#fff"
                  strokeWidth="2"
                />

                <text
                  x={layout.cx}
                  y={layout.cy - 3}
                  textAnchor="middle"
                  fill="#050505"
                  fontFamily="inherit"
                  fontSize="12"
                  fontWeight="600"
                >
                  {vendor}
                </text>

                <text
                  x={layout.cx}
                  y={layout.cy + 12}
                  textAnchor="middle"
                  fill="#555"
                  fontFamily="inherit"
                  fontSize="7"
                  letterSpacing="1"
                >
                  FOCUS
                </text>
              </svg>
            </div>

            <aside className="graph-sidebar">
              <div className="graph-sidebar-header">
                <div className="graph-sidebar-title">
                  Relationship Evidence
                </div>

                <div className="graph-sidebar-subtitle">
                  Strongest co-bidding relationships
                </div>
              </div>

              <div className="relationship-list">
                {relationships.map((edge, index) => {
                  const source = canonical(edge.source);
                  const target = canonical(edge.target);
                  const focus = canonical(vendor);

                  const other =
                    source === focus
                      ? target
                      : source;

                  return (
                    <button
                      type="button"
                      className="relationship-row"
                      key={`${edge.source}-${edge.target}-${index}`}
                      onClick={() =>
                        setVendor(vendorId(other))
                      }
                    >
                      <span className="relationship-rank">
                        {String(index + 1).padStart(2, "0")}
                      </span>

                      <span>
                        <span className="relationship-vendor">
                          {vendorId(other)}
                        </span>

                        <span className="relationship-detail">
                          CO-BID
                          {edge.observations
                            ? ` · ${edge.observations} observations`
                            : ""}
                        </span>
                      </span>

                      <span className="relationship-weight">
                        {(edge.weight ?? 0).toFixed(3)}
                      </span>
                    </button>
                  );
                })}

                {relationships.length === 0 && (
                  <div className="graph-status">
                    No direct co-bidding relationships found.
                  </div>
                )}
              </div>
            </aside>
          </div>
        </>
      )}
    </section>
  );
}

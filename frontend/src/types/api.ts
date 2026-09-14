export type EvidenceFamily =
  | "cooccurrence"
  | "economic"
  | "temporal"
  | "geographic"
  | "winner_rotation"
  | "winner_concentration"
  | string;

export interface InvestigationCase {
  vendor_id: string;
  investigation_priority: number;
  evidence_family_count: number;
  evidence_families: string | EvidenceFamily[];
  weighted_evidence_strength: number;
  strongest_family_strength: number;
  convergence_strength: number;
  evidence_diversity: number;
  persistence_strength: number;
  context_factor: number;
  minimum_family_context: number;
  core_evidence_families: string | null;
  supporting_evidence_families: string | null;
  core_evidence_strength: number;
  supporting_evidence_strength: number;
  evidence_count: number;
  risk_components: string;
  explanations: string;
}

export interface Vendor {
  vendor_id: string;
  legal_name?: string;
  name?: string;
  vendor_type?: string;
  region?: string;
  headquarters_region?: string;
  primary_category?: string;
  secondary_category?: string;
  specialization?: number;
  registration_date?: string;
}

export interface CaseDetail {
  case: InvestigationCase;
  vendor: Vendor | null;
}

export interface EvidenceRecord {
  evidence_family?: string;
  family?: string;
  source?: string;
  strength?: number;
  adjusted_strength?: number;
  evidence?: string;
  explanation?: string;
  vendor_id?: string;
  tender_id?: string;
  [key: string]: unknown;
}

export interface GraphNode {
  id: string;
  label?: string;
  type?: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  source: string;
  target: string;
  edge_type?: string;
  weight?: number;
  observations?: number;
  behavioral_strength?: number;
  economic_strength?: number;
  metadata?: string;
  [key: string]: unknown;
}

export interface CaseGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  [key: string]: unknown;
}

export interface Summary {
  total_cases?: number;
  investigation_cases?: number;
  multi_evidence_cases?: number;
  total_tenders?: number;
  procurement_tenders?: number;
  total_evidence_records?: number;
  evidence_records?: number;

  // Current backend summary fields
  vendors?: number;
  tenders?: number;
  bids?: number;
  multi_family_cases?: number;
  priority_min?: number;
  priority_median?: number;
  priority_p90?: number;
  priority_max?: number;
  evidence_family_counts?: Record<string, number>;
  evidence_families?: Record<string, number>;
}

import type {
  CaseDetail,
  CaseGraph,
  EvidenceRecord,
  InvestigationCase,
  Summary,
  Vendor,
} from "../types/api";

const API = "http://127.0.0.1:8000/api";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);

  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }

  return (await response.json()) as T;
}

export const api = {
  summary: () => request<Summary>("/summary"),

  cases: (limit = 50) =>
    request<{
      total: number;
      offset: number;
      limit: number;
      cases: InvestigationCase[];
    }>(`/cases?limit=${limit}`),

  case: (vendorId: string) =>
    request<CaseDetail>(
      `/cases/${encodeURIComponent(vendorId)}`,
    ),

  evidence: (vendorId: string) =>
    request<{ evidence: EvidenceRecord[] }>(
      `/cases/${encodeURIComponent(vendorId)}/evidence`,
    ),

  graph: (vendorId: string) =>
    request<CaseGraph>(
      `/cases/${encodeURIComponent(vendorId)}/graph`,
    ),

  vendor: (vendorId: string) =>
    request<{
      vendor: Vendor;
      case: InvestigationCase | null;
    }>(
      `/vendors/${encodeURIComponent(vendorId)}`,
    ),
};

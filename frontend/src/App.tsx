import { useState } from "react";
import { Sidebar, type Page } from "./components/Sidebar";
import { Brand } from "./components/Brand";
import { CommandCenter } from "./pages/CommandCenter";
import { EvidenceGraph } from "./pages/EvidenceGraph";
import { Investigations } from "./pages/Investigations";

function App() {
  const [page, setPage] =
    useState<Page>("command");

  const [selectedVendor, setSelectedVendor] =
    useState<string | null>(null);

  function openInvestigation(vendorId: string) {
    if (!vendorId) return;

    setSelectedVendor(vendorId);
    setPage("investigations");
  }

  function navigate(next: Page) {
    setPage(next);

    if (
      next !== "investigations" &&
      next !== "graph"
    ) {
      setSelectedVendor(null);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        active={page}
        onNavigate={navigate}
      />

      <div className="app-main">
        <header className="brand-header">
          <Brand />
        </header>

        <main className="main">
          {page === "command" && (
            <CommandCenter
              onInvestigate={openInvestigation}
            />
          )}

          {page === "investigations" && (
            <Investigations
              selectedVendor={selectedVendor}
              onSelect={(vendorId) =>
                setSelectedVendor(
                  vendorId || null,
                )
              }
            />
          )}

          {page === "graph" && (
            <EvidenceGraph
              initialVendor={selectedVendor}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;

import { DEFAULT_CONFIG } from "@doppelbot/config";

function App() {
  return (
    <main>
      <h1>doppelbot</h1>
      <p>Train against a slightly stronger version of yourself.</p>
      <dl>
        <dt>Cross-origin isolated</dt>
        <dd>{window.crossOriginIsolated ? "yes" : "no (multithreaded WASM unavailable)"}</dd>
        <dt>Model</dt>
        <dd>{DEFAULT_CONFIG.model.id}</dd>
      </dl>
    </main>
  );
}

export default App;

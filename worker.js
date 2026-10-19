// Runs Python (Pyodide + scikit-learn) off the main thread so the page stays responsive.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let py;
const ready = (async () => {
  postMessage({ type: "status", text: "Loading Python…" });
  py = await loadPyodide();
  postMessage({ type: "status", text: "Loading pandas & scikit-learn…" });
  await py.loadPackage(["numpy", "pandas", "scikit-learn"]);
  const code = await (await fetch("ml.py", { cache: "no-cache" })).text();
  py.runPython(code);
  postMessage({ type: "ready" });
})().catch((err) => postMessage({ type: "fatal", text: String(err && err.message || err) }));

onmessage = async (e) => {
  const { id, fn, args } = e.data;
  try {
    await ready;
    const f = py.globals.get(fn);
    const out = f(...args);
    f.destroy();
    postMessage({ id, ok: true, result: JSON.parse(out) });
  } catch (err) {
    // Python tracebacks are long; the last line holds the actual message.
    const lines = String(err && err.message || err).trim().split("\n");
    postMessage({ id, ok: false, error: lines[lines.length - 1].replace(/^\w+Error: /, "") });
  }
};

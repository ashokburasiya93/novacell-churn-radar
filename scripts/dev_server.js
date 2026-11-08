// Local server: static files from public/ plus the api/ handlers. Run: node scripts/dev_server.js
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
const TYPES = { ".html": "text/html", ".json": "application/json", ".js": "text/javascript" };

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname.startsWith("/api/")) {
    const mod = await import(join(ROOT, "api", url.pathname.slice(5) + ".js"));
    let body = "";
    for await (const chunk of req) body += chunk;
    const r = { statusCode: 200, headers: {} };
    const wrap = {
      status(c) { r.statusCode = c; return wrap; },
      setHeader(k, v) { r.headers[k] = v; },
      json(o) { res.writeHead(r.statusCode, { "Content-Type": "application/json", ...r.headers }); res.end(JSON.stringify(o)); },
    };
    return mod.default({ method: req.method, query: Object.fromEntries(url.searchParams), body: body ? JSON.parse(body) : {} }, wrap);
  }
  const path = join(ROOT, "public", url.pathname === "/" ? "index.html" : url.pathname);
  try {
    res.writeHead(200, { "Content-Type": TYPES[extname(path)] || "application/octet-stream" });
    res.end(await readFile(path));
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(process.env.PORT || 3000, () => console.log("http://localhost:" + (process.env.PORT || 3000)));

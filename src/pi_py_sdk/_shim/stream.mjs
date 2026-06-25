// Model-streaming shim: bridge @earendil-works/pi-ai's `streamSimple` over JSONL.
//
// This is the *only* place pi-py reaches the raw model layer. The full-agent client
// (PiAgent) spawns `pi --mode rpc`; this shim instead spawns just pi-ai, so a Python
// agent loop can own the turn structure and tools while delegating the LLM call (and
// pi's 30+ providers, auth, transports, local models) to pi-ai.
//
// Protocol (one JSON object per line):
//   stdin  <- {type:"stream", id, provider, model, context, options?}
//             {type:"stream", id, model:{...full Model...}, context, options?}
//             {type:"abort", id}
//             {type:"list_models", id, provider?}
//             {type:"list_providers", id}
//             {type:"ping", id}
//   stdout -> {type:"stream_event", id, event}   (one per pi-ai AssistantMessageEvent)
//             {type:"stream_error", id, error}    (shim-level failure: bad model, thrown error)
//             {type:"response", id, command, success, data?, error?}  (list_*/abort/ping)
//
// pi-ai already terminates each stream with a "done" or "error" AssistantMessageEvent
// carrying the final message, so the Python side needs no delta-accumulation logic.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

// --- args ------------------------------------------------------------------
const argv = process.argv.slice(2);
let piAiDir = null;
let authPath = path.join(os.homedir(), ".pi", "agent", "auth.json");
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--pi-ai-dir") piAiDir = argv[++i];
  else if (argv[i] === "--auth-path") authPath = argv[++i];
}
if (!piAiDir) {
  process.stderr.write("stream.mjs: missing --pi-ai-dir\n");
  process.exit(2);
}

// --- import pi-ai via its package.json exports (no bare-specifier resolution,
//     which fails because pi-ai declares an import-only `exports` map) ---------
function loadPiAi(dir) {
  const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
  // The global API this shim uses (registerBuiltInApiProviders / getProviders /
  // getModel / getModels / streamSimple / getEnvApiKey) moved off the main entry
  // into the "./compat" entrypoint in pi-ai 0.80. Prefer "./compat" when present;
  // fall back to "." for older builds that still expose those names on the main entry.
  const entryRel =
    pkg.exports?.["./compat"]?.import ?? pkg.exports?.["."]?.import ?? "./dist/index.js";
  const oauthRel = pkg.exports?.["./oauth"]?.import ?? "./dist/oauth.js";
  const toUrl = (rel) => pathToFileURL(path.join(dir, rel)).href;
  return Promise.all([import(toUrl(entryRel)), import(toUrl(oauthRel))]);
}

const [ai, oauth] = await loadPiAi(piAiDir);
ai.registerBuiltInApiProviders();

// --- output helper ---------------------------------------------------------
const out = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");
const ok = (id, command, data) => out({ type: "response", id, command, success: true, data });
const fail = (id, command, error) => out({ type: "response", id, command, success: false, error });

// --- credentials -----------------------------------------------------------
// Resolution order: caller-supplied apiKey > provider env var (pi-ai handles it) >
// the coding agent's OAuth login in ~/.pi/agent/auth.json (refreshed on expiry).
function loadAuth() {
  try {
    return JSON.parse(fs.readFileSync(authPath, "utf8"));
  } catch {
    return {};
  }
}

function saveAuth(auth) {
  try {
    fs.writeFileSync(authPath, JSON.stringify(auth, null, 2));
  } catch {
    // best-effort: a read-only auth store just means we refresh again next time
  }
}

async function resolveApiKey(provider, options) {
  if (options.apiKey) return options.apiKey; // explicit caller key wins
  try {
    if (ai.getEnvApiKey(provider)) return undefined; // let pi-ai read the env var
  } catch {
    // unknown provider for env lookup — fall through to OAuth
  }
  const cred = loadAuth()[provider];
  if (!cred || typeof cred !== "object" || !cred.access || !cred.refresh) return undefined;
  const provObj = oauth.getOAuthProvider(provider);
  if (!provObj) return cred.access;
  let current = cred;
  if (typeof current.expires === "number" && current.expires < Date.now()) {
    try {
      const refreshed = await provObj.refreshToken(current);
      current = { ...current, ...refreshed };
      const auth = loadAuth();
      auth[provider] = current;
      saveAuth(auth);
    } catch {
      // refresh failed — try the (likely expired) token so the model surfaces the auth error
    }
  }
  try {
    return provObj.getApiKey(current);
  } catch {
    return current.access;
  }
}

// --- streaming -------------------------------------------------------------
const controllers = new Map(); // id -> AbortController
const active = new Set(); // in-flight stream promises, drained on shutdown

function resolveModel(req) {
  if (req.model && typeof req.model === "object") return req.model; // full Model spec (e.g. local model)
  const model = ai.getModel(req.provider, req.model);
  if (!model) throw new Error(`${req.provider}/${req.model} not found`);
  return model;
}

async function handleStream(req) {
  const { id } = req;
  let model;
  try {
    model = resolveModel(req);
  } catch (err) {
    out({ type: "stream_error", id, error: `Unknown model: ${err?.message ?? err}` });
    return;
  }
  const provider = model?.provider ?? req.provider;
  const options = { ...(req.options ?? {}) };
  try {
    const key = await resolveApiKey(provider, options);
    if (key) options.apiKey = key;
  } catch {
    // proceed; the model will report any auth failure as an `error` event
  }

  const controller = new AbortController();
  controllers.set(id, controller);
  options.signal = controller.signal;
  const context = req.context ?? { messages: [] };

  try {
    const events = ai.streamSimple(model, context, options);
    for await (const event of events) {
      out({ type: "stream_event", id, event });
    }
  } catch (err) {
    out({ type: "stream_error", id, error: String(err?.message ?? err) });
  } finally {
    controllers.delete(id);
  }
}

function slimModel(m) {
  return {
    id: m.id,
    name: m.name,
    provider: m.provider,
    api: m.api,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    reasoning: m.reasoning,
    input: m.input,
    cost: m.cost,
  };
}

function handleListModels(req) {
  try {
    const models = [];
    const providers = req.provider ? [req.provider] : ai.getProviders();
    for (const p of providers) {
      try {
        for (const m of ai.getModels(p)) models.push(slimModel(m));
      } catch {
        // skip providers that fail to enumerate (e.g. missing config)
      }
    }
    ok(req.id, "list_models", { models });
  } catch (err) {
    fail(req.id, "list_models", String(err?.message ?? err));
  }
}

function handleListProviders(req) {
  try {
    ok(req.id, "list_providers", { providers: ai.getProviders() });
  } catch (err) {
    fail(req.id, "list_providers", String(err?.message ?? err));
  }
}

// --- dispatch --------------------------------------------------------------
const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let req;
  try {
    req = JSON.parse(trimmed);
  } catch {
    return; // ignore non-JSON lines
  }
  switch (req.type) {
    case "stream": {
      // concurrent: do not await — streams interleave by id. Track so a stdin
      // close (graceful shutdown) drains in-flight streams before exiting.
      const p = handleStream(req).finally(() => active.delete(p));
      active.add(p);
      break;
    }
    case "abort": {
      controllers.get(req.id)?.abort();
      ok(req.id, "abort", { aborted: true });
      break;
    }
    case "list_models":
      handleListModels(req);
      break;
    case "list_providers":
      handleListProviders(req);
      break;
    case "ping":
      ok(req.id, "ping", { ok: true });
      break;
    default:
      fail(req.id, req.type ?? "unknown", `Unknown request type: ${req.type}`);
  }
});
rl.on("close", async () => {
  await Promise.allSettled([...active]); // let in-flight streams finish
  process.exit(0);
});

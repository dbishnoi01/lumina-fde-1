# DESIGN.md — LUMINA

## Components

Five pieces, in three runtimes.

1. **Web UI** — the provided React 18 + Vite SPA, deployed to **Vercel**. It only ever
  holds a public gateway URL (`VITE_API_URL`); no key or provider name is baked in.
2. **Gateway** — an Express service that is the *only* thing the browser reaches. Binds
  `0.0.0.0:$PORT`. Holds no provider keys.
3. **Agent service** — an Express service that runs the agent loop, both gears, all six
  tools, memory, RAG retrieval, and deep search. Binds `127.0.0.1:8000` (loopback).
   Holds every provider key.
4. **Jobs worker** — a background polling loop (same container, no HTTP port) that turns
  an uploaded document into indexed, searchable chunks. Runs off the request path.
5. **MongoDB Atlas (M0)** — one cluster that is the whole persistence layer: authoritative
  state (`threads`, `messages`, `memories`, `spaces`, `documents`, `chunks`, `jobs`,
   `requests`, `runs`), the disposable `searchCache` (TTL), the two vector indexes and one
   text index, and GridFS for raw uploads.

The gateway, agent, and worker run as **three processes inside one Render free-tier
container**, started by a small supervisor (`render-start.mjs`). Render exposes only
`$PORT`, so only the gateway is publicly routable. The non-service state that still makes
decisions: `jobs` (the worker's queue), `searchCache` (whether a search is free), and
`runs/*.json` (what the grading gates read).

## Responsibilities

The interesting rules are the exclusions.

- **Gateway — the only browser-facing edge, and a mirror not an interpreter.** It is the
only component allowed to accept a cross-origin request, enforce `X-User-Id` (401
without it), validate bodies against the `@lumina/contract` zod schemas (400), rate-limit
per user (429), assign/propagate `X-Request-Id`, and stream SSE back unbuffered. It
forwards the agent's status and body **verbatim** — a 429/404/413/502 from the agent
survives intact. It may **not** hold a provider key, call an LLM, or decide anything
about the answer. If the agent is unreachable it returns 502, never a plausible 2xx.
- **Agent — the only holder of keys and the only enforcer of spend.** It is the only
component that reads `ANTHROPIC_API_KEY`/`GROQ_API_KEY`/`GEMINI_API_KEY`/`TAVILY_API_KEY`,
the only one that talks to Groq / Gemini / Tavily, and the **only** enforcer of the loop
caps and the `DEEP_DAILY_CAP` (deep 5/user/day → 429). It decides retrieval routing
(`auto|web|docs`), guarantees grounding (every `[n]` resolves to something fetched in
*this* request), and reports termination honestly (`done|cap|error`). A quick search may
never reach `plan_research` — the tool list is filtered by depth *before* the first model
call, not by a prompt asking nicely.
- **Worker — the only writer of chunk embeddings.** It is the only component allowed to
parse a PDF and write to `chunks`, and the only one that flips a document to `indexed` —
and only after a read-your-write probe proves the chunk is searchable.
- **Atlas — the only durable store.** Nothing is "remembered" that a query against Atlas
would not return.



## Communication

- **Browser → Gateway: HTTPS + SSE.** Regular JSON for most routes; for
`POST /threads/:id/ask` the gateway streams `text/event-stream` with `X-Accel-Buffering: no` and a flush per frame so time-to-first-token survives the hop. If the gateway is
down the browser shows a network error; there is nothing to fake.
- **Gateway → Agent: HTTP + SSE over loopback (**`127.0.0.1:8000`**).** Same contract; the
gateway pipes the agent's SSE bytes straight through. If the agent is down or throws, the
gateway surfaces 502 and the in-flight `ask` stream ends as an `error` event — never a
truncated answer dressed as success.
- **Agent → Worker: a Mongo collection, not a call.** The agent inserts a `pending` row
into `jobs` and returns `202` immediately; the worker polls, claims atomically
(`findOneAndUpdate` on `status:'pending'`), and does the work. Decoupling by a collection
means an upload can never block an answer stream, and a worker crash just leaves a
claimed row for the stale-claim sweeper to reset — the request already succeeded.
- **Everything → Atlas: the MongoDB driver**, one pooled client per process. If Atlas is
unreachable, `/health` reports `db: down` and the bench refuses to measure.



## State

- **Authoritative (in Atlas):** `threads`, `messages` (thread history), `memories`
(durable facts + their embedding), `spaces`, `documents` (ingest status), `chunks` (text
  - locator + embedding), `jobs` (the queue), `requests` and `runs` (observability). GridFS
  holds the raw upload. `userId` (from `X-User-Id`) owns every row.
- **Cache (disposable):** `searchCache` — keyed by `sha256(normalized query + provider)`,
expired by a TTL index, fronted by a per-process in-memory LRU. Deleting all of it costs
nothing but a few repeated Tavily calls; `searchCached` in the `done` event is only true
when *every* search in the request was a hit.
- **Consistency for "written but not yet searchable":** Atlas Search indexes are eventually
consistent, so an upserted chunk is not immediately returned by `$vectorSearch`. The
worker therefore does not trust the write — it runs a read-your-write probe (query the
vector index for a chunk it just wrote) and only flips the document to `indexed` once the
probe returns it. That is the whole reason `indexed` is a distinct status from `embedding`.



## Trade-offs

- **One Render container instead of two isolated services (Fly private networking).** The
PRD's stated reason for two services is independent deploy and failure. I kept the
*logical* split (gateway/agent/worker as separate processes, keys only in the agent, agent
on loopback) but gave up physical isolation to stay on a genuinely free tier. What I lose:
a crash in one process restarts all three, and I can't scale them independently. Given a
two-week single-user assignment, that trade is worth $0/month; for production it would not
be. **This is the decision I'm least sure about** — Render's cold-start on idle is a real
risk to the TTFT p95 gate, and I'm mitigating it only by warming the container before the
eval run rather than paying for always-on.
- **Gemini 3.5 flash lite instead of Claude Sonnet 5.** I gave up some answer quality and the
"deep is clearly better" human gate gets harder, in exchange for a free tier and much
faster tokens (which *helps* the TTFT and full-answer SLAs). The provider is swappable
behind one interface, so if quality costs me the human gate I can point `LLM_PROVIDER` at
Anthropic without touching the loop.
- **Gemini** `gemini-embedding-001` **at 1536-d instead of OpenAI** `text-embedding-3-small`**.**
Same dimension so the Atlas index is unchanged; I gave up the marginally better-documented
OpenAI model for a free tier, and took on Gemini's rate limits (which force me to batch and
pace ingestion).
- **Atlas Vector Search instead of a dedicated vector store (Qdrant/pgvector).** One document
per citation — the chunk text, its page locator, and its embedding live together, so a
citation is one read and `spaceId` is a plain filter inside `$vectorSearch`. The cost is
the M0 three-index ceiling: exactly 2 vector + 1 text, none to spare, so hybrid retrieval
has to fit in that budget.
- **MAX_TOOL_CALLS=4** A quick answer is light enough that a ~0.1-vCPU box (free Render) survives the bench's concurrency-4 load; the cost is more wall-clock cap terminations.

# P3 — GenAI Assistant frontend (Next.js)

Chat UI for the HR Policy RAG assistant, replacing the earlier Streamlit frontend
(`front/streamlit.py`, still present but no longer the primary path). Calls the
existing FastAPI backend (`api/`) directly — no changes needed there.

## Local development

```bash
npm install
cp .env.local.example .env.local
# edit .env.local: point NEXT_PUBLIC_API_BASE_URL at the deployed API, or at
# http://localhost:8000 if you're also running the backend locally via
# `uv run uvicorn api.server:app --reload` from the project root
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## What it does

- New chat / thread sidebar, backed by `GET /threads` and `GET /threads/{id}/messages`
- `thread_id` is kept in the URL so a page reload resumes the same conversation
- Sends questions via `POST /chat`, renders replies as markdown
- Distinguishes connection errors, timeouts, and API errors in the chat itself, matching the previous Streamlit app's behavior

See `src/lib/api.ts` and `src/lib/types.ts` for the exact API contract this depends on.

## Deployment

Deployed as its **own, separate Vercel project** from the FastAPI backend (same
relationship the Streamlit app had to it):

1. Create a new Vercel project pointed at this same git repository.
2. In the project's Settings → General, set **Root Directory** to `web`.
3. Set the `NEXT_PUBLIC_API_BASE_URL` environment variable to the deployed API's
   URL. Because it's `NEXT_PUBLIC_`-prefixed, it's inlined at build time — changing
   it later requires a redeploy, not just an env var update.

No `vercel.json` is needed here; Vercel's Next.js framework preset handles the
build automatically once Root Directory is set.

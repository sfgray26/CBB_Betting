# 🗃️ ARCHIVED — OpenClaw & CBB Edge Cheatsheet

> **Status:** Archived (Mar 2026)  
> **Reason:** Consolidated into `QUICKREF.md`  
> **Note:** OpenClaw CLI commands have been deprecated in favor of direct integration

---

# 🦞 OpenClaw & CBB Edge Cheatsheet

Quick reference for OpenClaw CLI and common project operations.

---

## 🦞 OpenClaw CLI Commands

### Service Management
- `openclaw gateway start` — Start the background gateway service.
- `openclaw gateway stop` — Stop the background gateway service.
- `openclaw gateway restart` — Restart the background gateway service.
- `openclaw gateway status` — Check the status of the gateway service.
- `openclaw gateway run` — Run the gateway in the **foreground** (useful for debugging).
- `openclaw gateway --force start` — Force start (kills any existing process on the port).

### Health & Monitoring
- `openclaw health` — Check health of the gateway and connected channels (Discord, etc.).
- `openclaw status` — Show channel health and recent sessions.
- `openclaw doctor` — Run health checks and suggest quick fixes.
- `openclaw logs` — Tail the gateway logs.

### Communication & Messages
- `openclaw message send --channel discord --target "CHANNEL_ID" --message "Hello"` — Send a Discord message.
- `openclaw tui` — Open the Terminal UI to interact with OpenClaw.
- `openclaw dashboard` — Open the web-based Control UI.

### Configuration & Discovery
- `openclaw configure` — Run the interactive setup wizard.
- `openclaw models list` — List available models (Ollama, etc.).
- `openclaw gateway discover` — Discover other gateways on the network.

---

## 🏀 CBB Edge Project Commands

### Backend & UI
- `uvicorn backend.main:app --reload` — Start the API backend.
- `streamlit run dashboard/app.py` — Start the Streamlit dashboard.
- `curl http://localhost:8000/health` — Check backend health.

### Database (PostgreSQL)
- `python scripts/init_db.py --seed` — Initialize/Seed the database.
- `psql $DATABASE_URL` — Connect to the database.

### Analysis & Maintenance
- `curl -X POST http://localhost:8000/admin/run-analysis -H "X-API-Key: $API_KEY"` — Trigger nightly analysis.
- `pytest tests/` — Run the full test suite.

---

## 🚨 Troubleshooting
- **Port 18789 already in use?** Run `openclaw gateway stop` or manually kill the process using the port.
- **Discord not connecting?** Check your `DISCORD_BOT_TOKEN` in `.env` and run `openclaw health`.
- **Database connection error?** Ensure PostgreSQL is running and `DATABASE_URL` is correct.

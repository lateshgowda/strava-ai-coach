# Strava AI Coach

A fully local, AI-powered personal endurance coaching platform.  
Connects to Strava, syncs all activity types, and applies sports-science analytics to give you actionable coaching — not just charts.

> **Core principle:** All training calculations are deterministic Python analytics.  
> The AI only interprets and explains — it never invents numbers.

---

## Features at a Glance

| Feature | What it does |
|---------|-------------|
| **Readiness Score** | Daily 0–100 score + recommendation (Rest → Hard Workout OK) |
| **Injury Risk** | Low/Medium/High with specific contributing factors |
| **Adaptive Weekly Plan** | Dynamically generated based on fatigue, ACWR, goal, and experience |
| **Race Predictions** | 5K / 10K / HM / Marathon projections with confidence score |
| **AI Conversational Coach** | Memory-persistent chat that knows your full training context |
| **Workout Generator** | Structured sessions (easy, intervals, tempo, long run) with target paces |
| **Athlete Profile** | Personalises everything — goal, experience, max HR, target race date |
| **Training Intelligence** | Aerobic fitness trend, monotony, strain, pace/HR efficiency, long-run fade |
| **Fatigue & Recovery** | Fatigue gauge, recovery gauge, ACWR gauge, load stats, load trend |
| **All Activity Types** | Runs, rides, strength, yoga, walks — all factored into load metrics |

---

## Architecture

```
Strava API
    │
    │ OAuth2 + REST
    ▼
FastAPI Backend  ──── SQLite DB
(localhost:8000)      (./data/strava.db)
    │
    │ ←→ OpenRouter AI (qwen/qwen3-14b)
    ▼
Streamlit Frontend
(localhost:8501)
```

Both services run in Docker. The frontend communicates with the backend over the internal Docker network.

---

## Project Structure

```
strava-ai-coach/
├── backend/
│   ├── main.py                        # FastAPI app + all endpoints
│   ├── models/
│   │   ├── activity.py                # Activity + StravaToken ORM models
│   │   └── profile.py                 # AthleteProfile + ChatMemory ORM models
│   ├── db/
│   │   ├── database.py                # SQLAlchemy engine + init_db
│   │   └── crud.py                    # All DB read/write operations
│   ├── strava/
│   │   ├── auth.py                    # OAuth2 login + token refresh
│   │   ├── client.py                  # Strava REST API client
│   │   └── sync.py                    # Incremental + full activity sync
│   ├── analytics/
│   │   ├── engine.py                  # Core metrics (pace, HR, ACWR, fatigue, PRs)
│   │   ├── training_state.py          # Advanced state (monotony, efficiency, fade)
│   │   ├── readiness.py               # Daily readiness engine
│   │   ├── injury_risk.py             # Injury risk assessment
│   │   ├── recommendations.py         # Adaptive weekly plan generator
│   │   ├── race_predictor.py          # Riegel formula + confidence scores
│   │   └── workout_generator.py       # Structured workout builder
│   ├── services/
│   │   └── activity_service.py        # Orchestrates analytics → dashboard payload
│   └── ai/
│       └── coach.py                   # OpenRouter prompts (insight, summary, chat)
├── frontend/
│   └── streamlit_app.py               # All 10 dashboard tabs
├── data/
│   └── strava.db                      # SQLite database (created on first run)
├── Dockerfile                         # Backend image
├── Dockerfile.frontend                # Frontend image
├── docker-compose.yml
├── requirements.txt
└── .env                               # Your credentials (not in git)
```

---

## Dashboard Tabs

| # | Tab | What's inside |
|---|-----|---------------|
| 1 | **Latest Run** | Distance, pace, time, HR, cadence, elevation. Km splits chart fetched on-demand from Strava and cached. |
| 2 | **Weekly** | 12-week mileage bar chart. Tabular view with distance, run count, avg pace, avg HR per week. |
| 3 | **Monthly Trends** | 6-month volume. Pace trend, HR trend, cadence trend charts. 4-week summary strip. |
| 4 | **AI Coach** | Persistent-memory chat. Knows your metrics, profile, readiness, and race predictions. Context-aware suggested prompts. |
| 5 | **Fatigue & Recovery** | Readiness recommendation card. Three gauges: Fatigue / Recovery / ACWR. Load stats row (streak, this week vs last week). 8-week load trend chart. |
| 6 | **Long Runs** | Long run progression by month. Personal records (5K/10K/HM via Riegel). HM projection. |
| 7 | **All Activities** | Filter by activity type. Summary cards. Volume chart. Activity mix pie. Full activities table. |
| 8 | **Training Intelligence** | Training state metrics · HR/pace efficiency trend charts · Long run fade analysis · Cadence analysis · Weekly consistency heatmap · Race predictions with confidence · Adaptive weekly plan · Injury risk detail |
| 9 | **Workouts** | Select type + available time → generate structured session with warmup, blocks, target paces, cooldown, and coaching note. |
| 10 | **Profile** | Athlete profile form. Feeds into AI coaching, pace targets, adaptive plans, and race predictions. Sleep/HRV fields for future wearable integration. |

**Top banner** (visible on every tab once data is loaded):
- Today's Readiness score
- Injury Risk level  
- Half Marathon projection
- Fitness trajectory direction

---

## Analytics Reference

### Core Metrics
| Metric | How it's computed |
|--------|-------------------|
| Fatigue score (0–100) | 7-day ATL ÷ 28-day CTL, scaled 0–100 |
| Recovery score (0–100) | Inverse fatigue + rest-day bonus |
| ACWR | 7-day acute load ÷ 28-day chronic average |
| Training streak | Consecutive days with any activity |
| Personal records | Riegel formula from best eligible run performances |

### Advanced Training State
| Metric | Sports science basis |
|--------|---------------------|
| Monotony score | Banister (1975): mean ÷ std of daily load over 28 days. Target < 1.5 |
| Strain score | Weekly load × monotony — high strain = accumulated fatigue risk |
| Consistency score | % of last 8 weeks containing at least 1 run |
| Recovery debt | (ATL − CTL) / CTL — positive = training above chronic baseline |
| Aerobic fitness score | Speed-per-HR efficiency over last 8 weeks, normalised 0–100 |
| HR efficiency trend | HR/km regression over last 20 runs (declining = improving) |
| Pace efficiency trend | Pace/HR-unit regression (declining = more speed per heartbeat) |
| Long run pace fade | First-third vs last-third split time comparison |
| Long run HR decoupling | HR drift between first and second half of long runs |
| Overreaching flag | ACWR > 1.3 AND high strain |
| Detraining flag | Load dropped > 25% over 3 weeks vs prior 3 weeks |

### Activity-Type Load Formulas
| Activity | Load |
|----------|------|
| Run | `distance_km × (1 + elevation_m / 1000)` |
| Ride | `distance_km × 0.5 × (1 + elevation_m / 2000)` |
| Walk | `distance_km × 0.3` |
| Hike | `distance_km × 0.6 × (1 + elevation_m / 1000)` |
| WeightTraining | `duration_min × 0.22` |
| Yoga | `duration_min × 0.06` |
| Swim | `duration_min × 0.20` |
| CrossFit | `duration_min × 0.25` |
| Everything else | `duration_min × 0.10` |

### Readiness Tiers
| Tier | Score | Meaning |
|------|-------|---------|
| ✅ Hard Workout OK | ≥ 68 | Prime condition for intervals or long run |
| 🟢 Moderate OK | 52–67 | Steady aerobic runs — avoid max intensity |
| 🟡 Easy Only | 35–51 | Fully conversational effort only |
| 🟠 Recovery Day | 20–34 | Walking, yoga, or light stretching only |
| 🛑 Full Rest | < 20 | Rest is the training today |

### AI Coach
Model: `qwen/qwen3-14b` via OpenRouter (free tier). Override with `OPENROUTER_MODEL` in `.env`.

Every response is grounded in: fatigue + recovery + ACWR + readiness + race predictions + training state + athlete profile. The coach references your actual numbers, explains the physiology, and gives specific recommendations.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/auth/strava/login` | Redirect to Strava OAuth consent |
| `GET` | `/auth/strava/callback` | OAuth callback — saves token |
| `POST` | `/auth/strava/disconnect` | Remove stored OAuth token |
| `GET` | `/auth/status` | Connection status + athlete name |
| `POST` | `/sync/activities` | Sync from Strava (`?full_sync=true` for full history) |
| `GET` | `/dashboard` | Full analytics payload for frontend |
| `GET` | `/ai/insight` | One-shot coaching insight |
| `GET` | `/ai/weekly` | Weekly training summary |
| `POST` | `/ai/chat` | Conversational coach (persistent memory) |
| `GET` | `/ai/chat/memory` | View stored chat history |
| `DELETE` | `/ai/chat/memory` | Clear all chat history |
| `GET` | `/profile` | Get athlete profile |
| `POST` | `/profile` | Save / update athlete profile |
| `POST` | `/workout/generate` | Generate a structured workout |
| `GET` | `/activities` | List activities (`?limit=100&activity_type=Run`) |
| `GET` | `/activities/{id}/splits` | Km splits — fetches on-demand, caches to DB |

---

## Setup

### Prerequisites
- Docker and Docker Compose
- A Strava account
- A Strava API application (free, 2 minutes to create)
- An OpenRouter API key (free tier available)

---

### Step 1 — Create a Strava API Application

1. Go to [https://www.strava.com/settings/api](https://www.strava.com/settings/api)
2. Fill in:
   - **Application Name**: Strava AI Coach
   - **Category**: Data Importer
   - **Website**: `http://localhost:8000`
   - **Authorization Callback Domain**: `localhost`
3. Copy your **Client ID** and **Client Secret**

---

### Step 2 — Get an OpenRouter API Key

1. Sign up at [https://openrouter.ai](https://openrouter.ai)
2. Go to **API Keys** → create a key

The default model `qwen/qwen3-14b` is free. You can switch to any OpenRouter model (e.g. `anthropic/claude-3-haiku`, `openai/gpt-4o-mini`) via `OPENROUTER_MODEL`.

---

### Step 3 — Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Strava
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REDIRECT_URI=http://localhost:8000/auth/strava/callback

# AI
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
OPENROUTER_MODEL=qwen/qwen3-14b

# Database (leave as-is for Docker)
DATABASE_URL=sqlite:////data/strava.db

# Frontend → Backend (inside Docker network — leave as-is)
BACKEND_URL=http://backend:8000
```

---

### Step 4 — Build and Start

```bash
docker compose up --build
```

Both services are ready when you see:
```
backend   | INFO: Application startup complete.
frontend  | You can now view your Streamlit app in your browser.
```

Open **http://localhost:8501**

---

### Step 5 — Connect Strava

Visit: **http://localhost:8000/auth/strava/login**

Authorise the app on Strava. You'll be redirected back to the dashboard.

---

### Step 6 — Sync Your Activities

In the sidebar:
- Click **Sync Activities**
- Check **Full re-sync** on first setup to download your full history
- Subsequent syncs are incremental (new activities only)

---

### Step 7 — Set Up Your Profile

Go to the **Profile** tab and fill in:
- Age, weight, max HR, resting HR
- Primary goal (HM / 10K / Marathon / Base Building)
- Target race date
- Running experience level
- Preferred weekly km and long run day

This immediately improves AI coaching, race predictions, pace targets, and the adaptive weekly plan.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STRAVA_CLIENT_ID` | Yes | — | Strava app Client ID |
| `STRAVA_CLIENT_SECRET` | Yes | — | Strava app Client Secret |
| `STRAVA_REDIRECT_URI` | No | `http://localhost:8000/auth/strava/callback` | OAuth callback URL |
| `OPENROUTER_API_KEY` | Yes (for AI) | — | OpenRouter API key |
| `OPENROUTER_MODEL` | No | `qwen/qwen3-14b` | Any OpenRouter model |
| `DATABASE_URL` | No | `sqlite:////data/strava.db` | SQLAlchemy database URL |
| `BACKEND_URL` | No | `http://backend:8000` | Backend URL used by the frontend |

---

## Running Without Docker

```bash
pip install -r requirements.txt
mkdir -p data

export DATABASE_URL=sqlite:///./data/strava.db
export STRAVA_CLIENT_ID=...
export STRAVA_CLIENT_SECRET=...
export OPENROUTER_API_KEY=...

# Terminal 1 — backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
export BACKEND_URL=http://localhost:8000
streamlit run frontend/streamlit_app.py --server.port 8501
```

---

## Sync Behaviour

| Mode | What happens |
|------|-------------|
| **Incremental** (default) | Fetches only activities newer than the most recently stored one |
| **Full re-sync** (checkbox) | Deletes all stored activities, then re-fetches full Strava history |

Splits are **not** fetched during sync (avoids Strava rate limits). They are fetched on-demand when you view the Latest Run tab, then cached in the database.

Rate limits: Strava allows 100 requests per 15 minutes. The sync uses 1 request per 200 activities — so a full history of 2000 activities uses ~10 requests.

---

## Troubleshooting

**`client_id: invalid` on login** — Docker has stale env vars after `.env` was edited. Fix:
```bash
docker compose up -d --force-recreate
```
`docker compose restart` does NOT reload `.env`.

**No split data for a run** — Normal. Open the Latest Run tab and wait for the spinner. Splits are fetched on first view and cached.

**Strava 429 rate limit** — You've hit 100 requests per 15 min. Wait 15 minutes and try again.

**AI not responding** — Verify `OPENROUTER_API_KEY` is active at [openrouter.ai/keys](https://openrouter.ai/keys). Then force-recreate to reload env vars.

**No fatigue/recovery data** — These require at least 2 weeks of activities. Sync your full history with Full re-sync.

**Reset everything and start fresh:**
```bash
rm data/strava.db
docker compose up -d --force-recreate
```

# Trading Decision Dashboard

RO
## Descriere
Sistem de suport decizional pentru trading crypto (BTC/altcoins). Agrega indicatori tehnici, context Smart Money, sentiment si un tag astro experimental. NU este bot de trading si NU executa ordine.

## Caracteristici principale
- Date live Binance/MEXC (REST + WS optional)
- Indicatori modulari (1 fisier / indicator)
- Scor agregat BUY / SELL / NO_TRADE
- Market regime zilnic (AGGRESSIVE / MODERATE / CHILL)
- Smart Money (Order Blocks + FVG + Sweeps + BOS/CHoCH)
- Sentiment: futures (funding/OI/long-short) + spot (buy/sell ratio proxy)
- UI Streamlit summary-grid: 2 paritati simultan x 4 ferestre timeframe, cu Start/Stop si mod Short/Long
- CSV logging la minut, rotatie zilnica, plus stari/conditii per indicator (`sig_*`, `conf_*`, `reason_*`)
- ML advisory real (logistic regression) + calibrare praguri + backtesting walk-forward
- Generator prompt automat pentru Claude/Codex (fisier text in `prompts/<SYMBOL>/`)
- Astro calendar experimental (Moon phases + aspects)

## Cerinte
- Python 3.12+ (obligatoriu)
- Windows / Mac / Linux

## Instalare pas cu pas (Windows)
1) Creeaza venv
```
python -m venv .venv
.venv\Scripts\activate
```
2) Instaleaza dependinte
```
python -m pip install -r requirements.txt
```
3) Configureaza .env
```
copy .env.example .env
```
4) (Optional) Ephemeris pentru astro
```
python scripts\download_ephemeris.py
```
5) Ruleaza UI
```
streamlit run src\ui\dashboard.py
```
Din folderul baza (`Trading Decision Dashboard/`), foloseste:
```
python -m streamlit run main\src\ui\dashboard.py
```

## Instalare pas cu pas (Linux/Mac)
1) Creeaza venv
```
python3 -m venv .venv
source .venv/bin/activate
```
2) Instaleaza dependinte
```
python -m pip install -r requirements.txt
```
3) Configureaza .env
```
cp .env.example .env
```
4) (Optional) Ephemeris pentru astro
```
python scripts/download_ephemeris.py
```
5) Ruleaza UI
```
streamlit run src/ui/dashboard.py
```

## Rulare CLI (optional)
```
python -m app start
python -m app stop
python -m app status
python -m app validate-config
```
Comenzile se ruleaza din folderul `main/`.
`start` ruleaza continuu; `stop` se da dintr-un al doilea terminal (nu doar `Ctrl+C`).

## ML Training + Backtest
Ruleaza din `main/`:
```
python scripts/train_model.py --symbol BTCUSDT
python scripts/backtest.py --symbol BTCUSDT
python scripts/backtest.py --symbol BTCUSDT --walk-forward --window-size 800 --step-size 100
```

## Prompt pentru Claude/Codex
- Config: `prompt_generator` in `config/default.yaml`
- Prompt generat automat in loop (prima generatie imediat, apoi la `interval_minutes`)
- Locatie implicita: `prompts/<SYMBOL>/latest_prompt_claude.txt`, `prompts/<SYMBOL>/latest_prompt_codex.txt` (+ `latest_prompt.txt`)
- Arhiva: `prompts/<SYMBOL>/archive/prompt_<TARGET>_<SYMBOL>_YYYYMMDD_HHMM.txt`
- Raspunsul AI (daca este parsat manual) se salveaza in `prompts/<SYMBOL>/latest_analysis.json`

## Configurare
- Fisier principal: `config/default.yaml`
- Profile: `config/aggressive.yaml`, `config/conservative.yaml`
- Multi-symbol: `app.symbols` in YAML sau `APP_SYMBOLS=BTCUSDT,ETHUSDT`
- Provider date: `data.provider` = auto/binance/mexc (auto selecteaza disponibil)

Sectiuni cheie:
- `indicator_weights`
- `indicator_params`
- `thresholds`
- `sentiment` (spot_trade_depth pentru imbalance avansat)
- `csv`, `time_sync`, `alerts`, `astro`, `ml`

## Note importante
- Sentiment futures functioneaza doar cu `market_type: futures`
- Pentru spot folosim proxy buy/sell ratio din 24h ticker + optional recent trades
- MEXC WS actualizeaza candle-ul curent in timp real (fara flag explicit de inchidere)
- In UI, butoanele `START/STOP` lanseaza/opresc engine-ul CLI (`python -m app start/stop`)
- In UI, modul `Short/Long Trade` seteaza timeframe-urile de analiza:
  - Short: `1m, 5m, 15m, 1h, 4h` (summary: `1m, 15m, 1h, 4h`)
  - Long: `1h, 4h, 1d, 1w` (summary: `1h, 4h, 1d, 1w`)
- UI afiseaza pentru fiecare paritate 4 carduri de summary pe timeframe-urile modului selectat
- CSV continua sa foloseasca timestamp ancorat pe `1m`
- Timestamp CSV este sincronizat NTP (cu fallback la ceasul local), iar runner-ul este aliniat la ceas (`time_sync.align_runner_to_clock`)
- CSV include `captured_at` (timp real de colectare) si `capture_lag_sec` pentru audit
- Pentru sincronizare stricta la minut, `time_sync.run_second_offset` poate fi negativ (ex: `-2.0`) ca prefetch-ul sa incheie aproape de fix
- Astro este context experimental (implicit NEUTRAL; optional bias directional din config)
- Darvas foloseste confirmare explicita (`confirmation_bars`) + suport volum/volatilitate
- Astro este experimental (nu predictor validat)

## Teste
```
pytest
```

## Troubleshooting
- Daca nu apar date: verifica conexiunea si timeframes
- Daca astro da eroare: ruleaza download ephemeris

EN
## Description
Decision-support system for crypto trading (BTC/altcoins). Aggregates technical indicators, Smart Money context, sentiment, and an experimental astro tag. It is NOT a trading bot and does NOT place orders.

## Key features
- Live Binance/MEXC market data (REST + optional WS)
- Modular indicators (1 file / indicator)
- Aggregated BUY / SELL / NO_TRADE scoring
- Daily market regime (AGGRESSIVE / MODERATE / CHILL)
- Smart Money (Order Blocks + FVG + Sweeps + BOS/CHoCH)
- Sentiment: futures (funding/OI/long-short) + spot (buy/sell proxy)
- Streamlit summary-grid UI: 2 simultaneous pairs x 4 timeframe windows, with Start/Stop and Short/Long mode
- Minute CSV logging with daily rotation, plus per-indicator state/condition (`sig_*`, `conf_*`, `reason_*`)
- Real ML advisory (logistic regression) + threshold calibration + walk-forward backtesting
- Automatic Claude/Codex prompt generator (`prompts/<SYMBOL>/`)
- Experimental astro calendar (Moon phases + aspects)

## Requirements
- Python 3.12+ (required)
- Windows / Mac / Linux

## Step-by-step setup (Windows)
1) Create venv
```
python -m venv .venv
.venv\Scripts\activate
```
2) Install deps
```
python -m pip install -r requirements.txt
```
3) Configure .env
```
copy .env.example .env
```
4) (Optional) Ephemeris for astro
```
python scripts\download_ephemeris.py
```
5) Run UI
```
streamlit run src\ui\dashboard.py
```
From the workspace root (`Trading Decision Dashboard/`), use:
```
python -m streamlit run main\src\ui\dashboard.py
```

## Step-by-step setup (Linux/Mac)
1) Create venv
```
python3 -m venv .venv
source .venv/bin/activate
```
2) Install deps
```
python -m pip install -r requirements.txt
```
3) Configure .env
```
cp .env.example .env
```
4) (Optional) Ephemeris for astro
```
python scripts/download_ephemeris.py
```
5) Run UI
```
streamlit run src/ui/dashboard.py
```

## CLI (optional)
```
python -m app start
python -m app stop
python -m app status
python -m app validate-config
```
Run CLI commands from the `main/` folder.
`start` runs continuously; use `stop` from a second terminal (not only `Ctrl+C`).

## ML Training + Backtest
Run from `main/`:
```
python scripts/train_model.py --symbol BTCUSDT
python scripts/backtest.py --symbol BTCUSDT
python scripts/backtest.py --symbol BTCUSDT --walk-forward --window-size 800 --step-size 100
```

## Claude/Codex Prompt
- Config block: `prompt_generator` in `config/default.yaml`
- Prompt is auto-generated in the runner loop (first generation immediately, then every `interval_minutes`)
- Default paths: `prompts/<SYMBOL>/latest_prompt_claude.txt`, `prompts/<SYMBOL>/latest_prompt_codex.txt` (+ `latest_prompt.txt`)
- Archive path: `prompts/<SYMBOL>/archive/prompt_<TARGET>_<SYMBOL>_YYYYMMDD_HHMM.txt`
- Parsed AI output (manual paste flow) is saved to `prompts/<SYMBOL>/latest_analysis.json`

## Configuration
- Main file: `config/default.yaml`
- Profiles: `config/aggressive.yaml`, `config/conservative.yaml`
- Multi-symbol: `app.symbols` in YAML or `APP_SYMBOLS=BTCUSDT,ETHUSDT`
- Data provider: `data.provider` = auto/binance/mexc (auto selects available)

Key sections:
- `indicator_weights`
- `indicator_params`
- `thresholds`
- `sentiment` (spot_trade_depth for deeper imbalance)
- `csv`, `time_sync`, `alerts`, `astro`, `ml`

## Notes
- Futures sentiment works only with `market_type: futures`
- Spot uses buy/sell proxy from 24h ticker + optional recent trades
- MEXC WS updates the current candle in real time (no explicit close flag)
- In UI, `START/STOP` controls launch/stop the CLI engine (`python -m app start/stop`)
- In UI, `Short/Long Trade` sets analysis timeframes:
  - Short: `1m, 5m, 15m, 1h, 4h` (summary: `1m, 15m, 1h, 4h`)
  - Long: `1h, 4h, 1d, 1w` (summary: `1h, 4h, 1d, 1w`)
- UI shows 4 summary cards per pair, on the selected mode timeframes
- CSV remains anchored on `1m` timestamp
- CSV timestamp is NTP-synced (fallback to local device clock), and the runner is clock-aligned (`time_sync.align_runner_to_clock`)
- CSV includes `captured_at` (real capture time) and `capture_lag_sec` for auditing
- For strict minute sync, `time_sync.run_second_offset` can be negative (e.g. `-2.0`) so prefetch finishes near minute close
- Astro is experimental context (default NEUTRAL; optional directional bias in config)
- Darvas uses explicit breakout confirmation (`confirmation_bars`) with volume/volatility support
- Astro is experimental (not a validated predictor)

## Tests
```
pytest
```

## Troubleshooting
- No data in UI: check connectivity and timeframes
- Astro errors: run ephemeris download

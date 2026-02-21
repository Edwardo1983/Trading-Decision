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
- UI Streamlit cu Start/Stop, Signal Board (live + CSV), tabel clasic, sumar, evenimente
- CSV logging la minut, rotatie zilnica, plus stari/conditii per indicator (`sig_*`, `conf_*`, `reason_*`)
- ML placeholder (lookback 21 candles)
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

## Prompt pentru Claude/Codex
- Config: `prompt_generator` in `config/default.yaml`
- Prompt generat automat in loop (prima generatie imediat, apoi la `interval_minutes`)
- Locatie implicita: `prompts/<SYMBOL>/latest_prompt.txt`
- Arhiva: `prompts/<SYMBOL>/archive/prompt_<SYMBOL>_YYYYMMDD_HHMM.txt`
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
- `csv`, `alerts`, `astro`, `ml`

## Note importante
- Sentiment futures functioneaza doar cu `market_type: futures`
- Pentru spot folosim proxy buy/sell ratio din 24h ticker + optional recent trades
- MEXC WS actualizeaza candle-ul curent in timp real (fara flag explicit de inchidere)
- In UI, modul `Signal Board (CSV Logs)` afiseaza ultimul rand din CSV pe simbol
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
- Streamlit UI with Start/Stop, Signal Board (live + CSV), classic table, summary, events
- Minute CSV logging with daily rotation, plus per-indicator state/condition (`sig_*`, `conf_*`, `reason_*`)
- ML placeholder (21-candle lookback)
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

## Claude/Codex Prompt
- Config block: `prompt_generator` in `config/default.yaml`
- Prompt is auto-generated in the runner loop (first generation immediately, then every `interval_minutes`)
- Default path: `prompts/<SYMBOL>/latest_prompt.txt`
- Archive path: `prompts/<SYMBOL>/archive/prompt_<SYMBOL>_YYYYMMDD_HHMM.txt`
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
- `csv`, `alerts`, `astro`, `ml`

## Notes
- Futures sentiment works only with `market_type: futures`
- Spot uses buy/sell proxy from 24h ticker + optional recent trades
- MEXC WS updates the current candle in real time (no explicit close flag)
- In UI, `Signal Board (CSV Logs)` displays the latest CSV row per symbol
- Astro is experimental (not a validated predictor)

## Tests
```
pytest
```

## Troubleshooting
- No data in UI: check connectivity and timeframes
- Astro errors: run ephemeris download

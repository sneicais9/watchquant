"""
WatchQuant MVP — Foundation
============================
Questo file crea tutto il necessario per partire:
1. Database SQLite con schema completo
2. Configurazione del progetto
3. Catalogo referenze orologi economici pre-popolato

COME USARE:
    pip install -r requirements.txt
    python foundation.py

Creerà:
    - watchquant.db (database SQLite)
    - config.json (configurazione editabile)
    - cartella logs/
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# 1. CONFIGURAZIONE PROGETTO
# ============================================================

DEFAULT_CONFIG = {
    "project_name": "WatchQuant MVP",
    "version": "0.1.0",

    # Database
    "database": {
        "path": "watchquant.db"
    },

    # Scraping settings
    "scraping": {
        "ebay": {
            "app_id": "",       # <-- Inserire quando pronte
            "cert_id": "",      # <-- Inserire quando pronte
            "calls_per_day": 5000,
            "category_id": "31387",  # Orologi da polso
            "marketplace": "EBAY_IT"
        },
        "chrono24": {
            "enabled": False,    # Abilitare quando proxy pronto
            "proxy": "",
            "max_pages": 3,
            "delay_min_sec": 3,
            "delay_max_sec": 7
        },
        "rate_limits": {
            "requests_per_minute": 10,
            "pause_on_error_sec": 60
        }
    },

    # Alert (Telegram)
    "alerts": {
        "telegram_token": "",   # <-- Inserire quando pronto
        "telegram_chat_id": "", # <-- Inserire quando pronto
        "email_enabled": False
    },

    # Trading parameters
    "strategy": {
        "underval_alert_threshold": 0.15,     # 15% = ALERT
        "underval_strong_buy_threshold": 0.25, # 25% = STRONG BUY
        "underval_suspicious_threshold": 0.40,  # 40% = possibile truffa
        "max_position_pct": 0.20,              # Max 20% capitale per pezzo
        "min_position_eur": 50,                # Min €50 (orologi economici)
        "cash_reserve_pct": 0.30,              # 30% cash sempre
        "stop_loss_pct": -0.15,                # -15% stop loss
        "take_profit_pct": 0.25,               # +25% take profit
        "max_single_brand_pct": 0.40,          # Max 40% su un brand
        "platform_fee_avg": 0.10,              # 10% commissioni media
        "shipping_cost_eur": 15,               # Spedizione media
        "slippage_pct": 0.03                   # 3% slippage
    },

    # Valuta
    "currency": {
        "base": "EUR",
        "converter_api": "https://api.frankfurter.app"
    }
}


def create_config(path="config.json"):
    """Crea il file di configurazione se non esiste."""
    if os.path.exists(path):
        print(f"  [OK] Config già esistente: {path}")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Config creato: {path}")


def load_config(path="config.json"):
    """Carica la configurazione."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. SCHEMA DATABASE
# ============================================================

SCHEMA_SQL = """
-- ============================================================
-- TABELLA: watches (anagrafica orologi)
-- Ogni referenza unica ha una riga qui
-- ============================================================
CREATE TABLE IF NOT EXISTS watches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brand           TEXT NOT NULL,           -- es. 'Seiko', 'Tissot'
    model           TEXT NOT NULL,           -- es. 'Presage', 'PRX'
    reference       TEXT NOT NULL UNIQUE,    -- es. 'SRPD37K1', 'T1374071104100'
    model_family    TEXT,                    -- es. 'Presage', 'PRX'
    year_production INTEGER,                -- anno produzione (se noto)
    case_material   TEXT DEFAULT 'steel',   -- steel, gold, titanium...
    case_size_mm    REAL,                   -- diametro cassa
    movement_type   TEXT,                   -- automatic, quartz, manual
    has_complication INTEGER DEFAULT 0,     -- 1 se cronografo/GMT/etc
    retail_price_eur REAL,                  -- prezzo listino (se noto)
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- TABELLA: listings (ogni annuncio trovato online)
-- Il cuore dei dati: ogni scrape produce righe qui
-- ============================================================
CREATE TABLE IF NOT EXISTS listings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id        INTEGER NOT NULL REFERENCES watches(id),
    source          TEXT NOT NULL,           -- 'ebay', 'chrono24', 'vinted', 'subito'
    external_id     TEXT,                    -- ID originale sulla piattaforma
    title           TEXT,                    -- titolo dell'annuncio
    price           REAL NOT NULL,           -- prezzo in EUR (normalizzato)
    currency_original TEXT DEFAULT 'EUR',
    condition       TEXT,                    -- 'new', 'like_new', 'good', 'fair', 'poor'
    condition_score INTEGER,                -- 4=new, 3=like_new, 2=good, 1=fair, 0=poor
    has_box         INTEGER DEFAULT 0,
    has_papers      INTEGER DEFAULT 0,
    has_warranty    INTEGER DEFAULT 0,
    completeness_score INTEGER DEFAULT 0,   -- box(1) + papers(1) + warranty(1)
    dial_variant    TEXT,                    -- colore/tipo quadrante
    seller_location TEXT,
    url             TEXT,
    image_urls      TEXT,                    -- JSON array di URL immagini
    scraped_at      TEXT DEFAULT (datetime('now')),
    status          TEXT DEFAULT 'active',  -- 'active', 'sold', 'expired'
    
    -- Campi calcolati dal modello (aggiornati dopo scoring)
    fair_value      REAL,                   -- stima fair value dal modello
    underval_score  REAL,                   -- (fair_value - price) / fair_value
    confidence      REAL,                   -- confidenza del modello (0-1)
    signal          TEXT,                   -- 'STRONG_BUY', 'BUY', 'WATCH', etc.
    scored_at       TEXT,                   -- quando è stato valutato
    
    UNIQUE(source, external_id)             -- evita duplicati per piattaforma
);

-- ============================================================
-- TABELLA: price_history (serie storiche aggregate)
-- Ogni giorno, per ogni referenza, salva statistiche di mercato
-- ============================================================
CREATE TABLE IF NOT EXISTS price_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id        INTEGER NOT NULL REFERENCES watches(id),
    date            TEXT NOT NULL,           -- 'YYYY-MM-DD'
    source          TEXT,                    -- NULL = aggregato multi-fonte
    avg_price       REAL,
    median_price    REAL,
    min_price       REAL,
    max_price       REAL,
    num_listings    INTEGER,
    UNIQUE(watch_id, date, source)
);

-- ============================================================
-- TABELLA: sentiment_data (analisi sentiment da Reddit, etc.)
-- ============================================================
CREATE TABLE IF NOT EXISTS sentiment_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id        INTEGER REFERENCES watches(id),  -- NULL = sentiment per brand
    brand           TEXT,
    source          TEXT NOT NULL,           -- 'reddit', 'google_trends', 'instagram'
    sentiment_score REAL,                   -- da -1.0 a +1.0
    mention_count   INTEGER DEFAULT 0,
    hype_index      REAL,                   -- score composito
    measured_at     TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- TABELLA: portfolio (i TUOI orologi — comprati e venduti)
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id        INTEGER NOT NULL REFERENCES watches(id),
    buy_price       REAL NOT NULL,
    buy_date        TEXT NOT NULL,
    buy_source      TEXT,                   -- dove hai comprato
    sell_price      REAL,                   -- NULL se ancora in portafoglio
    sell_date       TEXT,
    sell_source     TEXT,
    costs           TEXT,                   -- JSON: {"shipping":15, "auth":0, "commission":30}
    total_cost      REAL,                   -- buy_price + tutti i costi
    net_profit      REAL,                   -- sell_price - total_cost (calcolato)
    roi_pct         REAL,                   -- net_profit / total_cost * 100
    status          TEXT DEFAULT 'holding', -- 'holding', 'listed', 'sold'
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- TABELLA: scrape_log (log delle sessioni di scraping)
-- Per monitorare salute e performance degli scraper
-- ============================================================
CREATE TABLE IF NOT EXISTS scrape_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    started_at      TEXT DEFAULT (datetime('now')),
    finished_at     TEXT,
    status          TEXT DEFAULT 'running', -- 'running', 'success', 'error'
    listings_found  INTEGER DEFAULT 0,
    listings_new    INTEGER DEFAULT 0,
    error_message   TEXT
);

-- ============================================================
-- INDICI per performance
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_listings_watch_id ON listings(watch_id);
CREATE INDEX IF NOT EXISTS idx_listings_source ON listings(source);
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
CREATE INDEX IF NOT EXISTS idx_listings_underval ON listings(underval_score);
CREATE INDEX IF NOT EXISTS idx_listings_signal ON listings(signal);
CREATE INDEX IF NOT EXISTS idx_listings_scraped ON listings(scraped_at);
CREATE INDEX IF NOT EXISTS idx_price_history_watch ON price_history(watch_id, date);
CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio(status);
CREATE INDEX IF NOT EXISTS idx_watches_reference ON watches(reference);
CREATE INDEX IF NOT EXISTS idx_watches_brand ON watches(brand);
"""


def create_database(db_path="watchquant.db"):
    """Crea il database SQLite con tutto lo schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"  [OK] Database creato: {db_path}")


# ============================================================
# 3. CATALOGO REFERENZE — OROLOGI ECONOMICI
# ============================================================
# Questi sono i modelli che tracceremo per primi.
# Fascia: €80 — €800 sul mercato secondario.
# Selezionati per: volume di scambi, riconoscibilità, margine potenziale.

WATCH_CATALOG = [
    # --- SEIKO ---
    {"brand": "Seiko", "model": "Presage Cocktail Time", "reference": "SRPD37K1",
     "model_family": "Presage", "case_material": "steel", "case_size_mm": 40.5,
     "movement_type": "automatic", "retail_price_eur": 420,
     "notes": "Molto richiesto, quadrante blu 'Manhattan'"},
    {"brand": "Seiko", "model": "Presage Cocktail Time", "reference": "SSA346J1",
     "model_family": "Presage", "case_material": "steel", "case_size_mm": 40.5,
     "movement_type": "automatic", "retail_price_eur": 450,
     "notes": "Versione power reserve, buon mercato secondario"},
    {"brand": "Seiko", "model": "Prospex Turtle", "reference": "SRPE93K1",
     "model_family": "Prospex", "case_material": "steel", "case_size_mm": 45.0,
     "movement_type": "automatic", "retail_price_eur": 490,
     "notes": "Save the Ocean edition, molto collezionato"},
    {"brand": "Seiko", "model": "Prospex King Turtle", "reference": "SRPE07K1",
     "model_family": "Prospex", "case_material": "steel", "case_size_mm": 45.0,
     "movement_type": "automatic", "retail_price_eur": 520,
     "notes": "Versione premium del Turtle"},
    {"brand": "Seiko", "model": "5 Sports", "reference": "SRPD55K1",
     "model_family": "Seiko 5", "case_material": "steel", "case_size_mm": 42.5,
     "movement_type": "automatic", "retail_price_eur": 280,
     "notes": "Entry level, altissimo volume secondario"},
    {"brand": "Seiko", "model": "SKX007", "reference": "SKX007K2",
     "model_family": "SKX", "case_material": "steel", "case_size_mm": 42.5,
     "movement_type": "automatic", "retail_price_eur": 250,
     "notes": "ICONA. Fuori produzione, prezzi in crescita"},
    {"brand": "Seiko", "model": "SKX009", "reference": "SKX009K2",
     "model_family": "SKX", "case_material": "steel", "case_size_mm": 42.5,
     "movement_type": "automatic", "retail_price_eur": 250,
     "notes": "Pepsi bezel, stesso status dell'SKX007"},

    # --- ORIENT ---
    {"brand": "Orient", "model": "Bambino V2", "reference": "FAC00005W0",
     "model_family": "Bambino", "case_material": "steel", "case_size_mm": 40.5,
     "movement_type": "automatic", "retail_price_eur": 180,
     "notes": "Best value dress watch, molto scambiato"},
    {"brand": "Orient", "model": "Kamasu", "reference": "RA-AA0003R19B",
     "model_family": "Kamasu", "case_material": "steel", "case_size_mm": 41.8,
     "movement_type": "automatic", "retail_price_eur": 280,
     "notes": "Diver eccellente per il prezzo"},
    {"brand": "Orient", "model": "Ray II", "reference": "FAA02005D9",
     "model_family": "Ray", "case_material": "steel", "case_size_mm": 41.5,
     "movement_type": "automatic", "retail_price_eur": 200,
     "notes": "Diver entry level molto popolare"},

    # --- CASIO G-SHOCK ---
    {"brand": "Casio", "model": "G-Shock Casioak", "reference": "GA-2100-1A1ER",
     "model_family": "Casioak", "case_material": "resin", "case_size_mm": 45.4,
     "movement_type": "quartz", "retail_price_eur": 99,
     "notes": "Hype altissimo, edizioni limitate valgono multipli"},
    {"brand": "Casio", "model": "G-Shock Casioak Metal", "reference": "GM-2100-1AER",
     "model_family": "Casioak", "case_material": "steel", "case_size_mm": 44.4,
     "movement_type": "quartz", "retail_price_eur": 199,
     "notes": "Versione metal, premium secondario"},
    {"brand": "Casio", "model": "G-Shock Full Metal", "reference": "GMW-B5000D-1ER",
     "model_family": "G-Shock 5000", "case_material": "steel", "case_size_mm": 43.2,
     "movement_type": "quartz", "retail_price_eur": 499,
     "notes": "Iconico, full metal, buon resale"},
    {"brand": "Casio", "model": "G-Shock Square", "reference": "DW-5600E-1VER",
     "model_family": "G-Shock 5600", "case_material": "resin", "case_size_mm": 42.8,
     "movement_type": "quartz", "retail_price_eur": 69,
     "notes": "Classico senza tempo, edizioni speciali molto ricercate"},

    # --- TISSOT ---
    {"brand": "Tissot", "model": "PRX Powermatic 80", "reference": "T1374071104100",
     "model_family": "PRX", "case_material": "steel", "case_size_mm": 40.0,
     "movement_type": "automatic", "retail_price_eur": 650,
     "notes": "MOLTO hype, stile Royal Oak affordable"},
    {"brand": "Tissot", "model": "PRX Quartz", "reference": "T1374101104100",
     "model_family": "PRX", "case_material": "steel", "case_size_mm": 40.0,
     "movement_type": "quartz", "retail_price_eur": 350,
     "notes": "Versione entry, ottimo volume"},
    {"brand": "Tissot", "model": "Gentleman Powermatic 80", "reference": "T1274071104100",
     "model_family": "Gentleman", "case_material": "steel", "case_size_mm": 40.0,
     "movement_type": "automatic", "retail_price_eur": 595,
     "notes": "Dress watch versatile, buon secondario"},

    # --- HAMILTON ---
    {"brand": "Hamilton", "model": "Khaki Field Mechanical", "reference": "H69439931",
     "model_family": "Khaki Field", "case_material": "steel", "case_size_mm": 38.0,
     "movement_type": "manual", "retail_price_eur": 450,
     "notes": "Field watch iconico, 80h power reserve"},
    {"brand": "Hamilton", "model": "Khaki Field Auto", "reference": "H70455133",
     "model_family": "Khaki Field", "case_material": "steel", "case_size_mm": 38.0,
     "movement_type": "automatic", "retail_price_eur": 545,
     "notes": "Versione automatica, molto scambiato"},
    {"brand": "Hamilton", "model": "Khaki Aviation Pilot Day Date", "reference": "H64615135",
     "model_family": "Khaki Aviation", "case_material": "steel", "case_size_mm": 42.0,
     "movement_type": "automatic", "retail_price_eur": 695,
     "notes": "Interstellar watch, fan base forte"},

    # --- SWATCH / MOONSWATCH ---
    {"brand": "Swatch", "model": "MoonSwatch Mission to the Moon", "reference": "SO33M100",
     "model_family": "MoonSwatch", "case_material": "bioceramic", "case_size_mm": 42.0,
     "movement_type": "quartz", "retail_price_eur": 260,
     "notes": "Secondario volatile, margini interessanti"},
    {"brand": "Swatch", "model": "MoonSwatch Mission to Mars", "reference": "SO33R100",
     "model_family": "MoonSwatch", "case_material": "bioceramic", "case_size_mm": 42.0,
     "movement_type": "quartz", "retail_price_eur": 260,
     "notes": "Rosso molto richiesto"},
    {"brand": "Swatch", "model": "MoonSwatch Mission to Neptune", "reference": "SO33N100",
     "model_family": "MoonSwatch", "case_material": "bioceramic", "case_size_mm": 42.0,
     "movement_type": "quartz", "retail_price_eur": 260,
     "notes": "Blu navy, tra i più ricercati"},

    # --- CITIZEN ---
    {"brand": "Citizen", "model": "Promaster Diver", "reference": "BN0150-28E",
     "model_family": "Promaster", "case_material": "steel", "case_size_mm": 44.0,
     "movement_type": "quartz", "retail_price_eur": 250,
     "notes": "Eco-Drive, diver robusto e popolare"},
    {"brand": "Citizen", "model": "Promaster NY0040", "reference": "NY0040-09EE",
     "model_family": "Promaster", "case_material": "steel", "case_size_mm": 42.0,
     "movement_type": "automatic", "retail_price_eur": 280,
     "notes": "Fuori produzione, prezzi in salita"},

    # --- VOSTOK ---
    {"brand": "Vostok", "model": "Amphibia", "reference": "420059",
     "model_family": "Amphibia", "case_material": "steel", "case_size_mm": 40.0,
     "movement_type": "automatic", "retail_price_eur": 80,
     "notes": "Cult following, bassissimo prezzo, edizioni limitate ricercate"},

    # --- TIMEX ---
    {"brand": "Timex", "model": "Marlin Automatic", "reference": "TW2T22700",
     "model_family": "Marlin", "case_material": "steel", "case_size_mm": 40.0,
     "movement_type": "automatic", "retail_price_eur": 250,
     "notes": "Reissue vintage, molto popolare nel segmento entry"},
]


def populate_catalog(db_path="watchquant.db"):
    """Inserisce il catalogo referenze nel database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    inserted = 0
    skipped = 0
    for w in WATCH_CATALOG:
        try:
            cursor.execute("""
                INSERT INTO watches 
                    (brand, model, reference, model_family, case_material, 
                     case_size_mm, movement_type, retail_price_eur, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                w["brand"], w["model"], w["reference"],
                w.get("model_family"), w.get("case_material"),
                w.get("case_size_mm"), w.get("movement_type"),
                w.get("retail_price_eur"), w.get("notes")
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1  # Referenza già presente

    conn.commit()
    conn.close()
    print(f"  [OK] Catalogo: {inserted} referenze inserite, {skipped} già presenti")


# ============================================================
# 4. REQUIREMENTS.TXT (contenuto)
# ============================================================

REQUIREMENTS = """# WatchQuant MVP — Dependencies
# Installa con: pip install -r requirements.txt

# --- Core ---
requests>=2.31.0          # HTTP requests per API
beautifulsoup4>=4.12.0    # Parsing HTML

# --- Database ---
# SQLite è incluso in Python, nessun pacchetto extra

# --- Data ---
pandas>=2.1.0             # Manipolazione dati
numpy>=1.25.0             # Calcoli numerici

# --- ML (Fase 1: modello semplice) ---
scikit-learn>=1.3.0       # Ridge/Lasso regression + pipeline

# --- Scheduling ---
apscheduler>=3.10.0       # Job scheduling

# --- Alert ---
python-telegram-bot>=20.0 # Telegram bot per alert

# --- Dashboard ---
streamlit>=1.28.0         # Dashboard web
plotly>=5.17.0            # Grafici interattivi

# --- Utility ---
python-dotenv>=1.0.0      # Variabili d'ambiente
tqdm>=4.66.0              # Progress bar
"""


def create_requirements(path="requirements.txt"):
    """Crea il file requirements.txt."""
    if os.path.exists(path):
        print(f"  [OK] requirements.txt già esistente")
        return
    with open(path, "w") as f:
        f.write(REQUIREMENTS.strip())
    print(f"  [OK] requirements.txt creato")


# ============================================================
# 5. STRUTTURA CARTELLE
# ============================================================

DIRECTORIES = [
    "scrapers",         # Scraper per ogni piattaforma
    "models",           # Modelli ML e pipeline
    "strategy",         # Segnali, risk management, backtesting
    "alerts",           # Telegram bot, email
    "dashboard",        # Streamlit app
    "data",             # Dati grezzi, export CSV
    "logs",             # Log file
    "tests",            # Test
]


def create_directories():
    """Crea la struttura di cartelle del progetto."""
    for d in DIRECTORIES:
        Path(d).mkdir(exist_ok=True)
        # Crea __init__.py per i package Python
        if d not in ("data", "logs", "tests"):
            init_file = Path(d) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
    print(f"  [OK] Cartelle create: {', '.join(DIRECTORIES)}")


# ============================================================
# 6. UTILITY — Funzioni helper usate ovunque
# ============================================================

def get_db_connection(db_path=None):
    """Restituisce una connessione SQLite con row_factory."""
    if db_path is None:
        config = load_config()
        db_path = config["database"]["path"]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Accesso per nome colonna
    conn.execute("PRAGMA journal_mode=WAL")  # Performance migliori
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_all_references(db_path="watchquant.db"):
    """Restituisce tutte le referenze dal catalogo, utile per gli scraper."""
    conn = get_db_connection(db_path)
    rows = conn.execute(
        "SELECT id, brand, model, reference, model_family FROM watches ORDER BY brand, model"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_stats(db_path="watchquant.db"):
    """Stampa statistiche del database."""
    conn = get_db_connection(db_path)
    tables = {
        "watches": "Referenze nel catalogo",
        "listings": "Annunci trovati",
        "price_history": "Record storici prezzi",
        "sentiment_data": "Record sentiment",
        "portfolio": "Pezzi in portafoglio",
        "scrape_log": "Sessioni di scraping",
    }
    print("\n  📊 STATO DATABASE")
    print("  " + "─" * 40)
    for table, desc in tables.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {desc:.<35} {count:>5}")
    conn.close()


# ============================================================
# MAIN — Esegui setup completo
# ============================================================

def main():
    print("\n" + "=" * 50)
    print("  🕐 WatchQuant MVP — Setup Iniziale")
    print("=" * 50 + "\n")

    print("  [1/5] Creazione cartelle progetto...")
    create_directories()

    print("\n  [2/5] Creazione config.json...")
    create_config()

    print("\n  [3/5] Creazione database SQLite...")
    create_database()

    print("\n  [4/5] Inserimento catalogo referenze...")
    populate_catalog()

    print("\n  [5/5] Creazione requirements.txt...")
    create_requirements()

    # Mostra riepilogo
    db_stats()

    print("\n" + "=" * 50)
    print("  ✅ SETUP COMPLETATO!")
    print("=" * 50)
    print("""
  PROSSIMI PASSI:
  
  1. Installa le dipendenze:
     pip install -r requirements.txt

  2. Verifica che tutto funzioni:
     python foundation.py
     (rieseguirlo è sicuro, non sovrascrive nulla)

  3. Quando hai le API key eBay, inseriscile in config.json
     alla voce scraping → ebay → app_id / cert_id

  4. Il prossimo blocco sarà: SCRAPER eBay + Chrono24
""")


if __name__ == "__main__":
    main()

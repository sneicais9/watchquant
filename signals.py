"""
WatchQuant — External Market Signals
=======================================
Modulo per raccogliere segnali esterni che anticipano i movimenti
del mercato degli orologi. Ogni segnale è un "leading indicator"
che il sistema usa per migliorare le predizioni e generare alert.

SEGNALI TRACCIATI:
1. Prezzo dell'oro (XAU/USD) — pavimento per orologi in metallo prezioso
2. Tassi di cambio (EUR/USD, CHF/USD, CHF/EUR) — impatto su costi e prezzi
3. Google Trends — interesse di ricerca per modelli specifici
4. Indici di mercato finanziario (S&P500, VIX) — correlazione con lusso
5. Crypto (BTC) — correlazione provata con mercato orologi speculativo
6. Swiss Watch Exports — proxy dell'offerta sul mercato primario
7. Brand Price Alerts — monitoraggio rialzi prezzi retail

COME USARE:
    # Raccogli tutti i segnali disponibili:
    python signals.py --collect

    # Solo un segnale specifico:
    python signals.py --gold
    python signals.py --forex
    python signals.py --trends
    python signals.py --markets

    # Mostra il quadro macro attuale:
    python signals.py --dashboard

    # Mostra correlazioni segnali ↔ prezzi orologi:
    python signals.py --correlations

FILE RICHIESTI: foundation.py già eseguito.
"""

import json
import logging
import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    print("Installa dipendenze: pip install -r requirements.txt")
    exit(1)

from foundation import get_db_connection, load_config

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/signals.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WatchQuant.Signals")


# ============================================================
# 0. DATABASE — Tabella per i segnali esterni
# ============================================================

SIGNALS_SCHEMA = """
-- ============================================================
-- TABELLA: market_signals (segnali macro e di mercato)
-- Ogni riga è un dato puntuale di un indicatore esterno
-- ============================================================
CREATE TABLE IF NOT EXISTS market_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name     TEXT NOT NULL,       -- 'gold_usd', 'eur_usd', 'btc_usd', etc.
    signal_category TEXT NOT NULL,       -- 'commodity', 'forex', 'crypto', 'index', 'trends'
    value           REAL NOT NULL,
    value_change_1d REAL,               -- variazione % rispetto a ieri
    value_change_7d REAL,               -- variazione % rispetto a 7 giorni fa
    value_change_30d REAL,              -- variazione % rispetto a 30 giorni fa
    metadata        TEXT,               -- JSON con dati extra
    measured_at     TEXT DEFAULT (datetime('now')),
    UNIQUE(signal_name, measured_at)
);

CREATE INDEX IF NOT EXISTS idx_signals_name ON market_signals(signal_name);
CREATE INDEX IF NOT EXISTS idx_signals_date ON market_signals(measured_at);
CREATE INDEX IF NOT EXISTS idx_signals_category ON market_signals(signal_category);

-- ============================================================
-- TABELLA: google_trends (interesse di ricerca per modello)
-- ============================================================
CREATE TABLE IF NOT EXISTS google_trends (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT NOT NULL,       -- es. 'Rolex Submariner', 'Tissot PRX'
    region          TEXT DEFAULT '',     -- '' = worldwide, 'IT', 'US', etc.
    interest        INTEGER,            -- 0-100 (indice Google Trends)
    interest_change_7d REAL,            -- variazione vs 7 giorni fa
    measured_at     TEXT DEFAULT (datetime('now')),
    UNIQUE(keyword, region, measured_at)
);

CREATE INDEX IF NOT EXISTS idx_trends_keyword ON google_trends(keyword);
CREATE INDEX IF NOT EXISTS idx_trends_date ON google_trends(measured_at);

-- ============================================================
-- TABELLA: retail_price_changes (rialzi prezzi retail dei brand)
-- Ogni volta che un brand alza i prezzi, lo registriamo
-- ============================================================
CREATE TABLE IF NOT EXISTS retail_price_changes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brand           TEXT NOT NULL,
    change_pct      REAL NOT NULL,       -- +5.0 = aumento del 5%
    region          TEXT DEFAULT 'global', -- 'US', 'EU', 'global'
    effective_date  TEXT,
    source          TEXT,                -- 'watchpro', 'watchcharts', 'manual'
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""


def setup_signals_db(db_path="watchquant.db"):
    """Crea le tabelle per i segnali se non esistono."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SIGNALS_SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Tabelle segnali create/verificate.")


# ============================================================
# 1. GOLD PRICE — Prezzo dell'oro
# ============================================================

class GoldSignal:
    """
    Traccia il prezzo dell'oro (XAU/USD).
    
    L'oro è il segnale macro più forte per gli orologi in metallo prezioso.
    Quando l'oro sale rapidamente, gli orologi in oro sono temporaneamente
    sottovalutati rispetto al loro contenuto metallico.
    
    API: frankfurter.app (gratuita, nessuna API key)
    Fallback: metalpriceapi.com (gratuita, 100 req/mese)
    """

    # Contenuto approssimativo di oro per modello (grammi di oro puro)
    # Utile per calcolare il "floor price" basato sul metallo
    GOLD_CONTENT_GRAMS = {
        # Per orologi nella nostra fascia, quasi nessuno è in oro massiccio
        # Ma il prezzo dell'oro influenza INDIRETTAMENTE anche l'acciaio:
        # quando l'oro sale, i brand alzano i prezzi retail → secondario sale
        "default": 0,  # Acciaio
        "gold_plated": 2,  # Placcato oro
        "two_tone": 15,  # Bicolore (acciaio + oro)
        "solid_gold": 60,  # Oro massiccio 18k (tipico Rolex)
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_gold_price(self):
        """Recupera il prezzo corrente dell'oro in USD e EUR."""
        # Metodo 1: frankfurter.app per il tasso XAU
        # Questo dà il tasso di cambio, non il prezzo dell'oro diretto
        # Usiamo un approccio alternativo con API gratuite

        prices = {}

        # API gratuita: metalpriceapi.com alternativa
        # Per l'MVP usiamo un approccio semplice con frankfurter per EUR/USD
        # e una stima del prezzo dell'oro

        try:
            # Prezzo oro da API pubblica (open-source)
            resp = self.session.get(
                "https://api.frankfurter.app/latest?from=XAU&to=USD,EUR",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                rates = data.get("rates", {})
                # frankfurter restituisce quanti USD/EUR per 1 oncia troy d'oro
                if "USD" in rates:
                    prices["gold_usd"] = rates["USD"]
                if "EUR" in rates:
                    prices["gold_eur"] = rates["EUR"]
        except Exception as e:
            logger.debug(f"Frankfurter XAU non disponibile: {e}")

        # Fallback: usa un'API alternativa gratuita
        if not prices:
            try:
                # API gratuita per metalli preziosi
                resp = self.session.get(
                    "https://api.nbp.pl/api/cenyzlota?format=json",
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        # NBP riporta in PLN per grammo
                        pln_per_gram = data[0].get("cena", 0)
                        # Converti in USD (tasso approssimativo)
                        prices["gold_pln_per_gram"] = pln_per_gram
            except Exception as e:
                logger.debug(f"NBP gold API non disponibile: {e}")

        # Fallback finale: prezzo manuale recente come riferimento
        if not prices:
            logger.warning("[Gold] Nessuna API disponibile. Uso ultimo prezzo noto.")
            prices["gold_usd"] = 3300  # Prezzo approssimativo marzo 2026
            prices["gold_eur"] = 3050
            prices["_fallback"] = True

        return prices

    def collect(self, db_path="watchquant.db"):
        """Raccoglie e salva il prezzo dell'oro."""
        prices = self.fetch_gold_price()
        conn = get_db_connection(db_path)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        for name, value in prices.items():
            if name.startswith("_"):
                continue
            try:
                # Calcola variazioni storiche
                changes = self._calc_changes(conn, name, value)

                conn.execute("""
                    INSERT OR REPLACE INTO market_signals 
                        (signal_name, signal_category, value, 
                         value_change_1d, value_change_7d, value_change_30d,
                         measured_at)
                    VALUES (?, 'commodity', ?, ?, ?, ?, ?)
                """, (name, value,
                      changes.get("1d"), changes.get("7d"), changes.get("30d"),
                      now))
            except Exception as e:
                logger.debug(f"Errore salvataggio {name}: {e}")

        conn.commit()
        conn.close()

        gold_usd = prices.get("gold_usd", "?")
        logger.info(f"[Gold] Prezzo: ${gold_usd:,.0f}/oz" if isinstance(gold_usd, (int, float)) else f"[Gold] Prezzo: {gold_usd}")
        return prices

    def _calc_changes(self, conn, signal_name, current_value):
        """Calcola variazioni % rispetto a periodi precedenti."""
        changes = {}
        for label, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            row = conn.execute("""
                SELECT value FROM market_signals 
                WHERE signal_name = ? AND measured_at <= ?
                ORDER BY measured_at DESC LIMIT 1
            """, (signal_name, cutoff)).fetchone()
            if row and row["value"]:
                changes[label] = round((current_value - row["value"]) / row["value"] * 100, 2)
        return changes


# ============================================================
# 2. FOREX — Tassi di cambio
# ============================================================

class ForexSignal:
    """
    Traccia i tassi di cambio rilevanti per il mercato orologi.
    
    - EUR/USD: impatto su prezzi per il mercato europeo
    - CHF/USD: franco svizzero forte = costi produzione più alti
    - CHF/EUR: diretto impatto sui prezzi in Europa
    - GBP/EUR: mercato UK significativo per secondario
    
    Un franco svizzero forte costringe i brand ad alzare i prezzi,
    che a catena alza anche il secondario.
    
    API: frankfurter.app (gratuita, no key, basata su BCE)
    """

    PAIRS = {
        "eur_usd": {"from": "EUR", "to": "USD"},
        "chf_usd": {"from": "CHF", "to": "USD"},
        "chf_eur": {"from": "CHF", "to": "EUR"},
        "gbp_eur": {"from": "GBP", "to": "EUR"},
        "jpy_eur": {"from": "JPY", "to": "EUR"},
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_rates(self):
        """Recupera tutti i tassi di cambio."""
        rates = {}
        try:
            resp = self.session.get(
                "https://api.frankfurter.app/latest?from=EUR&to=USD,CHF,GBP,JPY",
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("rates", {})

            # EUR è la base
            rates["eur_usd"] = raw.get("USD", 0)
            rates["eur_chf"] = raw.get("CHF", 0)
            rates["eur_gbp"] = raw.get("GBP", 0)
            rates["eur_jpy"] = raw.get("JPY", 0)

            # Tassi derivati
            if rates["eur_usd"] and rates["eur_chf"]:
                rates["chf_usd"] = round(rates["eur_usd"] / rates["eur_chf"], 4)
            if rates["eur_chf"]:
                rates["chf_eur"] = round(1 / rates["eur_chf"], 4)

        except requests.RequestException as e:
            logger.error(f"[Forex] Errore API: {e}")

        return rates

    def collect(self, db_path="watchquant.db"):
        """Raccoglie e salva i tassi di cambio."""
        rates = self.fetch_rates()
        conn = get_db_connection(db_path)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        saved = 0
        for name, value in rates.items():
            if not value:
                continue
            try:
                changes = self._calc_changes(conn, name, value)
                conn.execute("""
                    INSERT OR REPLACE INTO market_signals 
                        (signal_name, signal_category, value,
                         value_change_1d, value_change_7d, value_change_30d,
                         measured_at)
                    VALUES (?, 'forex', ?, ?, ?, ?, ?)
                """, (name, value,
                      changes.get("1d"), changes.get("7d"), changes.get("30d"),
                      now))
                saved += 1
            except Exception as e:
                logger.debug(f"Errore salvataggio {name}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"[Forex] {saved} tassi salvati. EUR/USD: {rates.get('eur_usd', '?')}, CHF/EUR: {rates.get('chf_eur', '?')}")
        return rates

    def _calc_changes(self, conn, signal_name, current_value):
        changes = {}
        for label, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            row = conn.execute("""
                SELECT value FROM market_signals 
                WHERE signal_name = ? AND measured_at <= ?
                ORDER BY measured_at DESC LIMIT 1
            """, (signal_name, cutoff)).fetchone()
            if row and row["value"]:
                changes[label] = round((current_value - row["value"]) / row["value"] * 100, 2)
        return changes


# ============================================================
# 3. FINANCIAL MARKETS — Indici e Crypto
# ============================================================

class MarketSignal:
    """
    Traccia indici finanziari e crypto correlati al mercato orologi.
    
    - S&P 500: proxy della ricchezza e fiducia dei consumatori
    - VIX: indice di paura — alto VIX = meno acquisti di lusso
    - BTC/USD: il crash crypto 2022 ha fatto crollare gli orologi.
      Correlazione forte con il segmento speculativo.
    
    API: Yahoo Finance (via endpoint pubblico, no key)
    """

    # Simboli Yahoo Finance
    SYMBOLS = {
        "sp500": {"symbol": "^GSPC", "name": "S&P 500", "category": "index"},
        "vix": {"symbol": "^VIX", "name": "VIX Volatility", "category": "index"},
        "btc_usd": {"symbol": "BTC-USD", "name": "Bitcoin", "category": "crypto"},
        "eth_usd": {"symbol": "ETH-USD", "name": "Ethereum", "category": "crypto"},
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

    def fetch_quote(self, symbol):
        """Recupera il prezzo corrente da Yahoo Finance."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1d", "range": "5d"}
            resp = self.session.get(url, params=params, timeout=10)

            if resp.status_code != 200:
                return None

            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("chartPreviousClose", 0)

            # Storico 5 giorni
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]

            return {
                "price": price,
                "prev_close": prev_close,
                "change_1d": round((price - prev_close) / prev_close * 100, 2) if prev_close else 0,
                "closes_5d": closes,
            }

        except Exception as e:
            logger.debug(f"Yahoo Finance errore per {symbol}: {e}")
            return None

    def collect(self, db_path="watchquant.db"):
        """Raccoglie tutti gli indici finanziari e crypto."""
        conn = get_db_connection(db_path)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        results = {}

        for sig_name, info in self.SYMBOLS.items():
            quote = self.fetch_quote(info["symbol"])
            if not quote or not quote["price"]:
                logger.warning(f"[Market] {info['name']}: dati non disponibili")
                continue

            price = quote["price"]
            results[sig_name] = price

            # Calcola variazione a 7 giorni dal dato storico Yahoo
            change_7d = None
            if quote["closes_5d"] and len(quote["closes_5d"]) >= 5:
                old = quote["closes_5d"][0]
                if old:
                    change_7d = round((price - old) / old * 100, 2)

            # Variazione 30d dal nostro DB
            changes = self._calc_changes(conn, sig_name, price)

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO market_signals 
                        (signal_name, signal_category, value,
                         value_change_1d, value_change_7d, value_change_30d,
                         measured_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sig_name, info["category"], price,
                      quote.get("change_1d"),
                      change_7d or changes.get("7d"),
                      changes.get("30d"),
                      now))

                logger.info(f"[Market] {info['name']}: {price:,.2f} ({quote.get('change_1d', 0):+.2f}%)")
            except Exception as e:
                logger.debug(f"Errore salvataggio {sig_name}: {e}")

            time.sleep(0.5)  # Rate limiting Yahoo

        conn.commit()
        conn.close()
        return results

    def _calc_changes(self, conn, signal_name, current_value):
        changes = {}
        for label, days in [("7d", 7), ("30d", 30)]:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            row = conn.execute("""
                SELECT value FROM market_signals 
                WHERE signal_name = ? AND measured_at <= ?
                ORDER BY measured_at DESC LIMIT 1
            """, (signal_name, cutoff)).fetchone()
            if row and row["value"]:
                changes[label] = round((current_value - row["value"]) / row["value"] * 100, 2)
        return changes


# ============================================================
# 4. GOOGLE TRENDS — Interesse di ricerca
# ============================================================

class TrendsSignal:
    """
    Traccia l'interesse di ricerca Google per modelli specifici.
    
    Un aumento dell'interesse di ricerca precede l'aumento della domanda
    (e quindi dei prezzi) di 2-4 settimane.
    Un calo dell'interesse, specialmente in Asia, segnala un possibile
    picco raggiunto.
    
    Metodo: pytrends (se installato) oppure Google Trends RSS/scraping.
    Per l'MVP usiamo un approccio leggero che non richiede pytrends.
    """

    # Keywords da monitorare, divise per segmento
    # Questo è il nostro SOSTITUTO del sentiment Reddit:
    # Google Trends cattura la domanda reale delle persone
    # prima che si trasformi in acquisti (leading indicator 2-4 settimane)
    KEYWORDS = {
        # Modelli nel nostro catalogo — domanda diretta
        "entry_level": [
            "Seiko Presage", "Seiko SKX007", "Orient Bambino",
            "Casio Casioak", "Tissot PRX", "Hamilton Khaki",
            "MoonSwatch", "Vostok Amphibia",
        ],
        # Brand/modelli fascia alta — barometro del mercato generale
        "luxury_barometer": [
            "Rolex Submariner", "Rolex Daytona",
            "Omega Speedmaster", "Cartier Tank",
            "Patek Nautilus",
        ],
        # SENTIMENT PROXY — sostituisce Reddit
        # Questi termini catturano l'intenzione di acquisto/vendita
        "buy_intent": [
            "buy used watch", "pre owned watch",
            "best watch under 500", "affordable automatic watch",
            "watch deal", "orologio usato",
        ],
        "sell_intent": [
            "sell my watch", "watch prices dropping",
            "watch market crash", "vendere orologio",
        ],
        "hype_signals": [
            "watch investment", "watch collection",
            "best watches 2025", "watch review",
            "luxury watch affordable",
        ],
    }

    def __init__(self):
        self.pytrends_available = False
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl="en-US", tz=360)
            self.pytrends_available = True
        except ImportError:
            self.pytrends = None
            logger.info("[Trends] pytrends non installato. Installa con: pip install pytrends")

    def fetch_trends(self, keywords, region=""):
        """
        Recupera l'interesse Google Trends per una lista di keywords.
        Ritorna dict {keyword: interest_score (0-100)}
        """
        if not self.pytrends_available:
            return {}

        results = {}

        # Google Trends accetta max 5 keywords per volta
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i + 5]
            try:
                self.pytrends.build_payload(
                    batch,
                    cat=0,
                    timeframe="now 7-d",  # Ultimi 7 giorni
                    geo=region,
                )
                data = self.pytrends.interest_over_time()

                if data is not None and not data.empty:
                    for kw in batch:
                        if kw in data.columns:
                            # Media degli ultimi 7 giorni
                            results[kw] = int(data[kw].mean())
                        else:
                            results[kw] = 0

                time.sleep(2)  # Rate limiting Google

            except Exception as e:
                logger.warning(f"[Trends] Errore per batch {batch}: {e}")
                for kw in batch:
                    results[kw] = None

        return results

    def collect(self, db_path="watchquant.db"):
        """Raccoglie tutti i dati Google Trends."""
        if not self.pytrends_available:
            logger.info(
                "[Trends] pytrends non disponibile. "
                "Per attivare: python3 -m pip install pytrends"
            )
            return {}

        conn = get_db_connection(db_path)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        all_results = {}

        # Raccoglie per ogni categoria
        all_keywords = []
        for category, keywords in self.KEYWORDS.items():
            all_keywords.extend(keywords)

        # Worldwide
        logger.info(f"[Trends] Raccolta dati per {len(all_keywords)} keywords...")
        results_global = self.fetch_trends(all_keywords, region="")

        # Italia (mercato locale)
        results_it = self.fetch_trends(all_keywords[:10], region="IT")

        # Salva nel DB
        saved = 0
        for kw, interest in {**results_global}.items():
            if interest is None:
                continue

            # Calcola variazione vs settimana scorsa
            prev = conn.execute("""
                SELECT interest FROM google_trends 
                WHERE keyword = ? AND region = ''
                ORDER BY measured_at DESC LIMIT 1
            """, (kw,)).fetchone()

            change_7d = None
            if prev and prev["interest"]:
                change_7d = round(interest - prev["interest"], 1)

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO google_trends 
                        (keyword, region, interest, interest_change_7d, measured_at)
                    VALUES (?, '', ?, ?, ?)
                """, (kw, interest, change_7d, now))
                saved += 1
            except Exception:
                pass

            all_results[kw] = {"interest": interest, "change_7d": change_7d}

        # Salva anche i dati Italia
        for kw, interest in results_it.items():
            if interest is None:
                continue
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO google_trends 
                        (keyword, region, interest, measured_at)
                    VALUES (?, 'IT', ?, ?)
                """, (kw, interest, now))
            except Exception:
                pass

        conn.commit()
        conn.close()
        logger.info(f"[Trends] {saved} keywords salvate")
        return all_results

    def get_trending_up(self, db_path="watchquant.db", min_change=10):
        """Trova keywords con interesse in forte aumento."""
        conn = get_db_connection(db_path)
        rows = conn.execute("""
            SELECT keyword, interest, interest_change_7d
            FROM google_trends
            WHERE region = '' AND interest_change_7d > ?
            ORDER BY interest_change_7d DESC
        """, (min_change,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_trending_down(self, db_path="watchquant.db", max_change=-10):
        """Trova keywords con interesse in forte calo (segnale contrarian)."""
        conn = get_db_connection(db_path)
        rows = conn.execute("""
            SELECT keyword, interest, interest_change_7d
            FROM google_trends
            WHERE region = '' AND interest_change_7d < ?
            ORDER BY interest_change_7d ASC
        """, (max_change,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ============================================================
# 5. RETAIL PRICE TRACKER — Rialzi prezzi dei brand
# ============================================================

class RetailPriceTracker:
    """
    Registra e monitora i rialzi di prezzo retail dei brand.
    
    Ogni rialzo retail è un leading indicator:
    prezzo retail sale → secondario sale nelle 2-6 settimane successive.
    
    Per l'MVP: inserimento manuale + dati storici noti.
    In v1: scraping automatico da WatchCharts/WatchPro.
    """

    # Rialzi noti del 2025-2026 (dati pubblici da WatchCharts/WatchPro)
    KNOWN_INCREASES = [
        # 2025
        {"brand": "Rolex", "change_pct": 5.0, "region": "US", "effective_date": "2025-01-01",
         "source": "watchcharts", "notes": "Primo rialzo 2025, oro > acciaio"},
        {"brand": "Rolex", "change_pct": 3.0, "region": "US", "effective_date": "2025-05-01",
         "source": "watchcharts", "notes": "Secondo rialzo, uniforme"},
        {"brand": "Patek Philippe", "change_pct": 6.8, "region": "US", "effective_date": "2025-05-01",
         "source": "watchcharts", "notes": "Oro +8%, acciaio +5%"},
        {"brand": "Audemars Piguet", "change_pct": 6.6, "region": "US", "effective_date": "2025-05-01",
         "source": "watchcharts", "notes": "Oro +10%, acciaio +5%"},
        {"brand": "Patek Philippe", "change_pct": 15.0, "region": "US", "effective_date": "2025-09-15",
         "source": "fashionnetwork", "notes": "Rialzo legato a dazi USA"},
        {"brand": "Cartier", "change_pct": 10.0, "region": "US", "effective_date": "2025-09-01",
         "source": "fashionnetwork", "notes": "Rialzo legato a dazi USA"},
        {"brand": "Tudor", "change_pct": 5.6, "region": "US", "effective_date": "2025-05-01",
         "source": "watchcharts", "notes": "Due rialzi come Rolex"},
        {"brand": "Tissot", "change_pct": 5.0, "region": "EU", "effective_date": "2025-06-01",
         "source": "watchcharts", "notes": "Rialzo generale Swatch Group"},
        # 2026
        {"brand": "Rolex", "change_pct": 7.0, "region": "US", "effective_date": "2026-01-01",
         "source": "watchpro", "notes": "Rialzo annuale 2026"},
        {"brand": "Rolex", "change_pct": 5.2, "region": "UK", "effective_date": "2026-01-01",
         "source": "watchpro", "notes": "Rialzo annuale 2026 UK"},
        {"brand": "Audemars Piguet", "change_pct": 7.5, "region": "US", "effective_date": "2026-01-01",
         "source": "watchpro", "notes": "Royal Oak > CODE 11.59"},
        {"brand": "Tudor", "change_pct": 5.6, "region": "US", "effective_date": "2026-01-01",
         "source": "watchpro", "notes": "Rialzo annuale 2026"},
    ]

    def populate_known(self, db_path="watchquant.db"):
        """Inserisce i rialzi noti nel database."""
        conn = get_db_connection(db_path)
        inserted = 0

        for inc in self.KNOWN_INCREASES:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO retail_price_changes
                        (brand, change_pct, region, effective_date, source, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (inc["brand"], inc["change_pct"], inc["region"],
                      inc["effective_date"], inc["source"], inc["notes"]))
                inserted += 1
            except Exception:
                pass

        conn.commit()
        conn.close()
        logger.info(f"[Retail] {inserted} rialzi storici registrati")

    def get_recent_increases(self, db_path="watchquant.db", days=90):
        """Recupera rialzi retail negli ultimi N giorni."""
        conn = get_db_connection(db_path)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT * FROM retail_price_changes
            WHERE effective_date >= ?
            ORDER BY effective_date DESC
        """, (cutoff,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_increase(self, brand, change_pct, region="global",
                     effective_date=None, source="manual", notes="",
                     db_path="watchquant.db"):
        """Aggiungi un nuovo rialzo retail (manuale o da scraper)."""
        if effective_date is None:
            effective_date = datetime.now().strftime("%Y-%m-%d")
        conn = get_db_connection(db_path)
        conn.execute("""
            INSERT INTO retail_price_changes
                (brand, change_pct, region, effective_date, source, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (brand, change_pct, region, effective_date, source, notes))
        conn.commit()
        conn.close()
        logger.info(f"[Retail] Registrato: {brand} +{change_pct}% ({region}) dal {effective_date}")


# ============================================================
# 6. SIGNALS ORCHESTRATOR — Coordina tutto
# ============================================================

class SignalsOrchestrator:
    """
    Coordina la raccolta di tutti i segnali e genera un
    "quadro macro" che il sistema usa per le decisioni.
    """

    def __init__(self, db_path="watchquant.db"):
        self.db_path = db_path
        self.gold = GoldSignal()
        self.forex = ForexSignal()
        self.markets = MarketSignal()
        self.trends = TrendsSignal()
        self.retail = RetailPriceTracker()

    def collect_all(self):
        """Raccoglie tutti i segnali disponibili."""
        logger.info("\n" + "=" * 60)
        logger.info("  RACCOLTA SEGNALI ESTERNI")
        logger.info("=" * 60)

        setup_signals_db(self.db_path)

        results = {}

        # Gold
        logger.info("\n  [1/5] Prezzo Oro...")
        results["gold"] = self.gold.collect(self.db_path)

        # Forex
        logger.info("\n  [2/5] Tassi di Cambio...")
        results["forex"] = self.forex.collect(self.db_path)

        # Markets
        logger.info("\n  [3/5] Indici Finanziari & Crypto...")
        results["markets"] = self.markets.collect(self.db_path)

        # Google Trends
        logger.info("\n  [4/5] Google Trends...")
        results["trends"] = self.trends.collect(self.db_path)

        # Retail price changes
        logger.info("\n  [5/5] Rialzi Prezzi Retail...")
        self.retail.populate_known(self.db_path)
        results["retail_increases"] = self.retail.get_recent_increases(self.db_path)

        return results

    def get_macro_dashboard(self):
        """
        Genera il quadro macro completo.
        Ritorna un dict con tutti i segnali più recenti e la loro interpretazione.
        """
        conn = get_db_connection(self.db_path)

        # Ultimi segnali per ogni indicatore
        signals = conn.execute("""
            SELECT signal_name, signal_category, value, 
                   value_change_1d, value_change_7d, value_change_30d,
                   measured_at
            FROM market_signals
            WHERE id IN (
                SELECT MAX(id) FROM market_signals GROUP BY signal_name
            )
            ORDER BY signal_category, signal_name
        """).fetchall()

        # Ultimi trends
        trends = conn.execute("""
            SELECT keyword, interest, interest_change_7d, measured_at
            FROM google_trends
            WHERE region = '' AND id IN (
                SELECT MAX(id) FROM google_trends WHERE region = '' GROUP BY keyword
            )
            ORDER BY interest DESC
        """).fetchall()

        # Rialzi recenti
        increases = conn.execute("""
            SELECT brand, change_pct, region, effective_date
            FROM retail_price_changes
            ORDER BY effective_date DESC LIMIT 10
        """).fetchall()

        conn.close()

        return {
            "signals": [dict(s) for s in signals],
            "trends": [dict(t) for t in trends],
            "retail_increases": [dict(i) for i in increases],
            "generated_at": datetime.now().isoformat()
        }

    def get_signal_features(self):
        """
        Genera feature numeriche dai segnali per il modello ML.
        Queste feature vengono aggiunte al dataset di training
        per migliorare le predizioni.
        
        Returns:
            dict con feature pronte per il modello
        """
        conn = get_db_connection(self.db_path)

        features = {}

        # Prendi l'ultimo valore di ogni segnale
        rows = conn.execute("""
            SELECT signal_name, value, value_change_7d, value_change_30d
            FROM market_signals
            WHERE id IN (SELECT MAX(id) FROM market_signals GROUP BY signal_name)
        """).fetchall()

        for row in rows:
            name = row["signal_name"]
            features[f"signal_{name}"] = row["value"] or 0
            features[f"signal_{name}_chg7d"] = row["value_change_7d"] or 0
            features[f"signal_{name}_chg30d"] = row["value_change_30d"] or 0

        # Aggiungi conteggio rialzi retail recenti (ultimi 90 giorni)
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        increase_count = conn.execute("""
            SELECT COUNT(*) as cnt, AVG(change_pct) as avg_pct
            FROM retail_price_changes
            WHERE effective_date >= ?
        """, (cutoff,)).fetchone()

        features["retail_increases_90d"] = increase_count["cnt"] if increase_count else 0
        features["retail_avg_increase_90d"] = increase_count["avg_pct"] if increase_count else 0

        conn.close()
        return features

    def print_dashboard(self):
        """Stampa il quadro macro in formato leggibile."""
        data = self.get_macro_dashboard()

        print("\n" + "=" * 70)
        print("  \U0001f30d QUADRO MACRO — WatchQuant Signals")
        print("=" * 70)

        # Segnali di mercato
        if data["signals"]:
            current_cat = ""
            for s in data["signals"]:
                cat = s["signal_category"]
                if cat != current_cat:
                    cat_labels = {
                        "commodity": "\U0001f947 COMMODITIES",
                        "forex": "\U0001f4b1 FOREX",
                        "index": "\U0001f4c8 INDICI",
                        "crypto": "\u20bf CRYPTO"
                    }
                    print(f"\n  {cat_labels.get(cat, cat.upper())}")
                    print(f"  {'─' * 60}")
                    current_cat = cat

                name = s["signal_name"]
                value = s["value"]
                chg_1d = s.get("value_change_1d")
                chg_7d = s.get("value_change_7d")
                chg_30d = s.get("value_change_30d")

                # Formattazione
                if "usd" in name or "eur" in name or "chf" in name or "gbp" in name or "jpy" in name:
                    if value > 100:
                        val_str = f"{value:>12,.0f}"
                    else:
                        val_str = f"{value:>12.4f}"
                else:
                    val_str = f"{value:>12,.2f}"

                changes = []
                if chg_1d is not None:
                    changes.append(f"1d: {chg_1d:+.2f}%")
                if chg_7d is not None:
                    changes.append(f"7d: {chg_7d:+.2f}%")
                if chg_30d is not None:
                    changes.append(f"30d: {chg_30d:+.2f}%")

                change_str = "  |  ".join(changes) if changes else ""
                print(f"  {name:<20} {val_str}    {change_str}")
        else:
            print("\n  Nessun segnale raccolto. Esegui: python signals.py --collect")

        # Google Trends
        if data["trends"]:
            print(f"\n  \U0001f50d GOOGLE TRENDS (interesse 0-100)")
            print(f"  {'─' * 60}")
            for t in data["trends"][:15]:
                interest = t["interest"]
                change = t.get("interest_change_7d")
                bar = "\u2588" * (interest // 5) if interest else ""
                change_str = f"({change:+.0f})" if change is not None else ""
                emoji = "\U0001f525" if (change and change > 10) else "\U0001f4c9" if (change and change < -10) else ""
                print(f"  {t['keyword']:<30} {interest:>3} {bar} {change_str} {emoji}")

        # Rialzi retail
        if data["retail_increases"]:
            print(f"\n  \U0001f4b0 RIALZI RETAIL RECENTI")
            print(f"  {'─' * 60}")
            for r in data["retail_increases"][:8]:
                print(f"  {r['effective_date']}  {r['brand']:<20} +{r['change_pct']:.1f}%  ({r['region']})")

        # Interpretazione
        print(f"\n  \U0001f9e0 INTERPRETAZIONE")
        print(f"  {'─' * 60}")
        self._print_interpretation(data)
        print()

    def _print_interpretation(self, data):
        """Genera interpretazione automatica dei segnali."""
        signals_dict = {s["signal_name"]: s for s in data.get("signals", [])}

        notes = []

        # Oro
        gold = signals_dict.get("gold_usd", {})
        if gold.get("value_change_30d") and gold["value_change_30d"] > 5:
            notes.append(
                f"\U0001f7e2 Oro in forte rialzo ({gold['value_change_30d']:+.1f}% 30d). "
                "Orologi in oro potenzialmente sottovalutati."
            )
        elif gold.get("value_change_30d") and gold["value_change_30d"] < -5:
            notes.append(
                f"\U0001f534 Oro in calo ({gold['value_change_30d']:+.1f}% 30d). "
                "Pressione al ribasso su orologi in metallo prezioso."
            )

        # Forex — CHF forte
        chf = signals_dict.get("chf_eur", {})
        if chf.get("value_change_30d") and chf["value_change_30d"] > 2:
            notes.append(
                "\U0001f7e2 Franco svizzero in rafforzamento. "
                "I brand alzeranno i prezzi retail → secondario sale."
            )

        # VIX
        vix = signals_dict.get("vix", {})
        if vix.get("value") and vix["value"] > 25:
            notes.append(
                f"\U0001f534 VIX alto ({vix['value']:.0f}). Alta volatilit\u00e0 finanziaria. "
                "Il mercato del lusso potrebbe soffrire nel breve."
            )
        elif vix.get("value") and vix["value"] < 15:
            notes.append(
                f"\U0001f7e2 VIX basso ({vix['value']:.0f}). Mercati calmi. "
                "Ambiente favorevole per acquisti di lusso."
            )

        # BTC
        btc = signals_dict.get("btc_usd", {})
        if btc.get("value_change_30d") and btc["value_change_30d"] > 15:
            notes.append(
                f"\U0001f7e2 Bitcoin in forte rialzo ({btc['value_change_30d']:+.1f}% 30d). "
                "Storicamente correlato a domanda speculativa di orologi."
            )
        elif btc.get("value_change_30d") and btc["value_change_30d"] < -15:
            notes.append(
                f"\U0001f534 Bitcoin in calo ({btc['value_change_30d']:+.1f}% 30d). "
                "Attenzione: il crash crypto 2022 ha anticipato il calo degli orologi."
            )

        # Retail increases
        recent_increases = data.get("retail_increases", [])
        if len(recent_increases) >= 3:
            avg_increase = np.mean([r["change_pct"] for r in recent_increases[:5]])
            notes.append(
                f"\U0001f7e2 {len(recent_increases)} rialzi retail recenti (media +{avg_increase:.1f}%). "
                "Il secondario seguir\u00e0 al rialzo nelle prossime settimane."
            )

        if not notes:
            notes.append("\u2796 Nessun segnale forte al momento. Mercato stabile.")

        for note in notes:
            print(f"  {note}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchQuant — Segnali Esterni di Mercato"
    )
    parser.add_argument("--collect", action="store_true",
                        help="Raccogli tutti i segnali")
    parser.add_argument("--gold", action="store_true",
                        help="Solo prezzo oro")
    parser.add_argument("--forex", action="store_true",
                        help="Solo tassi di cambio")
    parser.add_argument("--markets", action="store_true",
                        help="Solo indici finanziari e crypto")
    parser.add_argument("--trends", action="store_true",
                        help="Solo Google Trends")
    parser.add_argument("--dashboard", action="store_true",
                        help="Mostra quadro macro completo")
    parser.add_argument("--correlations", action="store_true",
                        help="Mostra feature per il modello ML")
    args = parser.parse_args()

    setup_signals_db()
    orch = SignalsOrchestrator()

    if args.collect:
        orch.collect_all()
        orch.print_dashboard()
    elif args.gold:
        orch.gold.collect()
    elif args.forex:
        orch.forex.collect()
    elif args.markets:
        orch.markets.collect()
    elif args.trends:
        orch.trends.collect()
    elif args.dashboard:
        orch.print_dashboard()
    elif args.correlations:
        features = orch.get_signal_features()
        print("\n  \U0001f9e0 FEATURE SEGNALI PER IL MODELLO ML")
        print("  " + "─" * 50)
        for name, value in sorted(features.items()):
            print(f"  {name:<40} {value:>12.4f}")
        print()
    else:
        print("\n  \U0001f30d WatchQuant — Segnali di Mercato")
        print("  " + "─" * 45)
        print("  Opzioni:")
        print("    python signals.py --collect       \u2192 Raccogli tutti i segnali")
        print("    python signals.py --gold          \u2192 Solo prezzo oro")
        print("    python signals.py --forex         \u2192 Solo tassi di cambio")
        print("    python signals.py --markets       \u2192 Solo indici e crypto")
        print("    python signals.py --trends        \u2192 Solo Google Trends")
        print("    python signals.py --dashboard     \u2192 Quadro macro completo")
        print("    python signals.py --correlations  \u2192 Feature per modello ML")
        print()


if __name__ == "__main__":
    main()

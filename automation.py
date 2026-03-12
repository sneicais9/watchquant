"""
WatchQuant MVP — Alert Engine & Scheduler
============================================
Il pilota automatico del sistema:
1. AlertEngine — notifiche Telegram quando trova opportunità
2. PriceAggregator — calcola e salva statistiche giornaliere dei prezzi
3. WatchQuantScheduler — orchestra tutto: scraping → scoring → alert
4. HealthMonitor — controlla che tutto funzioni e avvisa se ci sono problemi

COME USARE:
    # Test alert Telegram (verifica che funzioni):
    python automation.py --test-telegram

    # Esegui un ciclo completo manuale:
    python automation.py --run-once

    # Avvia il sistema automatico in background:
    python automation.py --start

    # Mostra lo schedule programmato:
    python automation.py --show-schedule

SETUP TELEGRAM:
    1. Apri Telegram, cerca @BotFather
    2. Manda /newbot e segui le istruzioni
    3. Copia il token (es. 123456:ABC-DEF...)
    4. Cerca il tuo bot e mandagli /start
    5. Per trovare il chat_id, visita:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       (cerca "chat":{"id":XXXXXXX})
    6. Inserisci token e chat_id in config.json → alerts

FILE RICHIESTI: foundation.py, scrapers.py, models.py già funzionanti.
"""

import json
import logging
import argparse
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    print("Installa dipendenze: pip install -r requirements.txt")
    exit(1)

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

from foundation import get_db_connection, load_config
from scrapers import ScraperOrchestrator, Normalizer
from models import PricingModel, UndervalDetector, FeatureBuilder, ModelTrainer

# Prova a importare signals (opzionale)
try:
    from signals import SignalsOrchestrator, setup_signals_db
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False

# Prova a importare marketplace scrapers (opzionale)
try:
    from marketplace_scrapers import MarketplaceOrchestrator
    HAS_MARKETPLACE = True
except ImportError:
    HAS_MARKETPLACE = False

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/automation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WatchQuant.Automation")


# ============================================================
# 1. ALERT ENGINE — Notifiche Telegram
# ============================================================

class AlertEngine:
    """
    Gestisce le notifiche via Telegram.
    
    Invia alert per:
    - Opportunità STRONG_BUY e BUY trovate
    - Report giornaliero del portafoglio
    - Errori di sistema (scraper bloccati, modello degradato)
    - Variazioni significative di prezzo su pezzi in portafoglio
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, config_path="config.json"):
        config = load_config(config_path)
        alerts_cfg = config.get("alerts", {})
        self.token = alerts_cfg.get("telegram_token", "")
        self.chat_id = alerts_cfg.get("telegram_chat_id", "")
        self.enabled = bool(self.token and self.chat_id)

        # Tracking per evitare alert duplicati
        self._sent_alerts = set()
        self._daily_count = 0
        self._max_daily = 50  # Limite giornaliero per non spammare

    def is_configured(self):
        return self.enabled

    def _call_api(self, method, data):
        """Chiama l'API Telegram."""
        if not self.enabled:
            return None
        try:
            url = self.TELEGRAM_API.format(token=self.token, method=method)
            resp = requests.post(url, json=data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[Telegram] Errore API: {e}")
            return None

    def send_message(self, text, parse_mode="Markdown", disable_preview=True):
        """Invia un messaggio Telegram."""
        if not self.enabled:
            logger.info(f"[Telegram] (non configurato) Messaggio: {text[:80]}...")
            return False

        if self._daily_count >= self._max_daily:
            logger.warning("[Telegram] Limite giornaliero raggiunto.")
            return False

        result = self._call_api("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview
        })

        if result and result.get("ok"):
            self._daily_count += 1
            return True
        return False

    # --- Alert specifici ---

    def send_opportunity(self, listing):
        """Invia alert per un'opportunità di acquisto."""
        # Evita duplicati (stesso listing nella stessa sessione)
        alert_key = f"{listing.get('source')}_{listing.get('external_id', listing.get('listing_id'))}"
        if alert_key in self._sent_alerts:
            return
        self._sent_alerts.add(alert_key)

        signal = listing.get("signal", "BUY")
        emoji = {"STRONG_BUY": "\U0001f525", "BUY": "\u2b50"}.get(signal, "\U0001f440")

        price = listing.get("price", 0)
        fair_value = listing.get("fair_value", 0)
        discount_pct = listing.get("underval_score", 0) * 100
        discount_eur = fair_value - price if fair_value else 0
        confidence = listing.get("confidence", 0)

        # Completezza
        comp = int(listing.get("completeness_score", 0))
        box_icon = "\u2705" if listing.get("has_box") else "\u274c"
        papers_icon = "\u2705" if listing.get("has_papers") else "\u274c"

        msg = (
            f"{emoji} *{signal}*\n\n"
            f"*{listing.get('brand', '?')} {listing.get('model', '?')}*\n"
            f"Ref: `{listing.get('reference', '?')}`\n\n"
            f"\U0001f4b0 Prezzo: \u20ac{price:,.0f}\n"
            f"\U0001f3af Fair Value: \u20ac{fair_value:,.0f}\n"
            f"\U0001f4c9 *Sconto: {discount_pct:+.1f}% (\u20ac{discount_eur:,.0f})*\n\n"
            f"\U0001f4ca Confidenza: {confidence:.0%}\n"
            f"\U0001f4e6 Box: {box_icon}  Papers: {papers_icon}\n"
            f"\U0001f50d Condizione: {listing.get('condition', '?')}\n"
            f"\U0001f310 Fonte: {listing.get('source', '?')}\n"
        )

        url = listing.get("url", "")
        if url and "example.com" not in url:
            msg += f"\n[\U0001f517 Vedi annuncio]({url})"

        self.send_message(msg)
        logger.info(f"[Alert] {signal}: {listing.get('brand')} {listing.get('reference')} @ \u20ac{price:,.0f}")

    def send_daily_report(self, stats):
        """Invia il report giornaliero."""
        msg = (
            "\U0001f4cb *REPORT GIORNALIERO — WatchQuant*\n"
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_\n\n"
            f"\U0001f50d *Scraping*\n"
            f"  Listing trovati: {stats.get('listings_found', 0)}\n"
            f"  Nuovi listing: {stats.get('listings_new', 0)}\n\n"
            f"\U0001f3af *Opportunit\u00e0*\n"
            f"  \U0001f525 STRONG BUY: {stats.get('strong_buy', 0)}\n"
            f"  \u2b50 BUY: {stats.get('buy', 0)}\n"
            f"  \U0001f440 WATCH: {stats.get('watch', 0)}\n\n"
        )

        if stats.get("portfolio_nav"):
            msg += (
                f"\U0001f4bc *Portafoglio*\n"
                f"  NAV: \u20ac{stats['portfolio_nav']:,.0f}\n"
                f"  Pezzi: {stats.get('portfolio_count', 0)}\n"
                f"  P&L non realizzato: \u20ac{stats.get('unrealized_pnl', 0):+,.0f}\n"
            )

        self.send_message(msg)

    def send_error(self, component, error_msg):
        """Invia alert di errore di sistema."""
        msg = (
            "\u26a0\ufe0f *ERRORE SISTEMA*\n\n"
            f"Componente: `{component}`\n"
            f"Errore: {error_msg[:300]}\n"
            f"Ora: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_message(msg)

    def send_price_alert(self, watch_info, old_price, new_price, change_pct):
        """Invia alert per variazione significativa di prezzo."""
        direction = "\U0001f4c8" if change_pct > 0 else "\U0001f4c9"
        msg = (
            f"{direction} *VARIAZIONE PREZZO*\n\n"
            f"*{watch_info.get('brand', '?')} {watch_info.get('model', '?')}*\n"
            f"Ref: `{watch_info.get('reference', '?')}`\n\n"
            f"Prezzo medio: \u20ac{old_price:,.0f} \u2192 \u20ac{new_price:,.0f}\n"
            f"Variazione: *{change_pct:+.1f}%*"
        )
        self.send_message(msg)

    def reset_daily_counter(self):
        """Reset del contatore giornaliero e dei duplicati."""
        self._daily_count = 0
        self._sent_alerts.clear()


# ============================================================
# 2. PRICE AGGREGATOR — Statistiche giornaliere
# ============================================================

class PriceAggregator:
    """
    Calcola e salva le statistiche aggregate dei prezzi
    per ogni referenza, ogni giorno.
    Popola la tabella price_history.
    """

    def __init__(self, db_path="watchquant.db"):
        self.db_path = db_path

    def aggregate_today(self):
        """Calcola le statistiche di oggi per tutte le referenze."""
        conn = get_db_connection(self.db_path)
        today = datetime.now().strftime("%Y-%m-%d")

        # Statistiche per referenza + source
        query = """
            SELECT 
                l.watch_id,
                l.source,
                AVG(l.price) as avg_price,
                -- mediana approssimata (la vera mediana serve subquery)
                AVG(l.price) as median_price,
                MIN(l.price) as min_price,
                MAX(l.price) as max_price,
                COUNT(*) as num_listings
            FROM listings l
            WHERE l.status = 'active' AND l.price > 0
            GROUP BY l.watch_id, l.source
        """
        rows = conn.execute(query).fetchall()

        inserted = 0
        for row in rows:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO price_history 
                        (watch_id, date, source, avg_price, median_price,
                         min_price, max_price, num_listings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["watch_id"], today, row["source"],
                    round(row["avg_price"], 2),
                    round(row["median_price"], 2),
                    round(row["min_price"], 2),
                    round(row["max_price"], 2),
                    row["num_listings"]
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Errore aggregazione: {e}")

        # Statistiche aggregate (tutte le fonti)
        query_all = """
            SELECT 
                l.watch_id,
                AVG(l.price) as avg_price,
                AVG(l.price) as median_price,
                MIN(l.price) as min_price,
                MAX(l.price) as max_price,
                COUNT(*) as num_listings
            FROM listings l
            WHERE l.status = 'active' AND l.price > 0
            GROUP BY l.watch_id
        """
        rows_all = conn.execute(query_all).fetchall()

        for row in rows_all:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO price_history 
                        (watch_id, date, source, avg_price, median_price,
                         min_price, max_price, num_listings)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
                """, (
                    row["watch_id"], today,
                    round(row["avg_price"], 2),
                    round(row["median_price"], 2),
                    round(row["min_price"], 2),
                    round(row["max_price"], 2),
                    row["num_listings"]
                ))
                inserted += 1
            except Exception:
                pass

        conn.commit()
        conn.close()
        logger.info(f"[Aggregator] Statistiche giornaliere salvate: {inserted} record")
        return inserted

    def detect_price_changes(self, threshold_pct=10):
        """
        Rileva variazioni significative di prezzo rispetto a ieri.
        Restituisce lista di referenze con variazione > threshold.
        """
        conn = get_db_connection(self.db_path)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        query = """
            SELECT 
                t.watch_id,
                w.brand, w.model, w.reference,
                y.avg_price as yesterday_price,
                t.avg_price as today_price,
                ((t.avg_price - y.avg_price) / y.avg_price * 100) as change_pct
            FROM price_history t
            JOIN price_history y ON t.watch_id = y.watch_id 
                AND y.date = ? AND y.source IS NULL
            JOIN watches w ON t.watch_id = w.id
            WHERE t.date = ? AND t.source IS NULL
              AND ABS((t.avg_price - y.avg_price) / y.avg_price * 100) > ?
            ORDER BY ABS(change_pct) DESC
        """
        changes = pd.read_sql_query(query, conn, params=(yesterday, today, threshold_pct))
        conn.close()
        return changes


# ============================================================
# 3. HEALTH MONITOR — Controlla la salute del sistema
# ============================================================

class HealthMonitor:
    """
    Monitora la salute di tutti i componenti:
    - Scraper funzionanti (nessun errore nelle ultime 24h)
    - Modello non troppo vecchio (< 7 giorni)
    - Database in crescita (nuovi listing nelle ultime 24h)
    """

    def __init__(self, db_path="watchquant.db"):
        self.db_path = db_path

    def check_all(self):
        """Esegue tutti i controlli. Ritorna lista di problemi."""
        issues = []
        issues.extend(self._check_scrapers())
        issues.extend(self._check_model())
        issues.extend(self._check_database())
        return issues

    def _check_scrapers(self):
        """Verifica che gli scraper funzionino."""
        issues = []
        conn = get_db_connection(self.db_path)
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

        # Cerca errori recenti
        errors = conn.execute("""
            SELECT source, COUNT(*) as error_count, MAX(error_message) as last_error
            FROM scrape_log 
            WHERE status = 'error' AND started_at > ?
            GROUP BY source
        """, (cutoff,)).fetchall()

        for err in errors:
            if err["error_count"] >= 3:
                issues.append({
                    "component": "scraper",
                    "severity": "high",
                    "message": f"Scraper {err['source']}: {err['error_count']} errori in 24h. Ultimo: {err['last_error'][:100]}"
                })

        # Verifica che ci siano scrape recenti
        recent = conn.execute("""
            SELECT COUNT(*) as count FROM scrape_log 
            WHERE status = 'success' AND started_at > ?
        """, (cutoff,)).fetchone()

        if recent["count"] == 0:
            issues.append({
                "component": "scraper",
                "severity": "medium",
                "message": "Nessun scrape completato con successo nelle ultime 24h"
            })

        conn.close()
        return issues

    def _check_model(self):
        """Verifica lo stato del modello ML."""
        issues = []
        meta_path = Path("models/model_metadata.json")

        if not meta_path.exists():
            issues.append({
                "component": "model",
                "severity": "high",
                "message": "Nessun modello addestrato. Esegui: python models.py --train"
            })
            return issues

        with open(meta_path) as f:
            meta = json.load(f)

        trained_at = datetime.fromisoformat(meta.get("trained_at", "2000-01-01"))
        age_days = (datetime.now() - trained_at).days

        if age_days > 7:
            issues.append({
                "component": "model",
                "severity": "medium",
                "message": f"Modello vecchio di {age_days} giorni. Consigliato retrain settimanale."
            })

        # Verifica qualità
        mape = meta.get("final_metrics", {}).get("mape", 100)
        if mape > 20:
            issues.append({
                "component": "model",
                "severity": "high",
                "message": f"Modello poco accurato (MAPE: {mape}%). Servono più dati o tuning."
            })

        return issues

    def _check_database(self):
        """Verifica crescita del database."""
        issues = []
        conn = get_db_connection(self.db_path)
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

        new_listings = conn.execute("""
            SELECT COUNT(*) as count FROM listings WHERE scraped_at > ?
        """, (cutoff,)).fetchone()["count"]

        total_listings = conn.execute(
            "SELECT COUNT(*) as count FROM listings"
        ).fetchone()["count"]

        conn.close()

        if total_listings == 0:
            issues.append({
                "component": "database",
                "severity": "high",
                "message": "Database vuoto. Esegui: python scrapers.py --demo"
            })
        elif new_listings == 0:
            issues.append({
                "component": "database",
                "severity": "low",
                "message": "Nessun nuovo listing nelle ultime 24h"
            })

        return issues

    def print_status(self):
        """Stampa lo stato di salute del sistema."""
        issues = self.check_all()

        print("\n" + "=" * 50)
        print("  \U0001f3e5 HEALTH CHECK — WatchQuant")
        print("=" * 50)

        if not issues:
            print("\n  \u2705 Tutti i sistemi funzionano correttamente!\n")
            return

        severity_icons = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e1"}

        for issue in issues:
            icon = severity_icons.get(issue["severity"], "\u2753")
            print(f"\n  {icon} [{issue['component'].upper()}] {issue['message']}")

        print()


# ============================================================
# 4. SCHEDULER — Il cuore dell'automazione
# ============================================================

class WatchQuantScheduler:
    """
    Orchestra l'esecuzione automatica di tutti i componenti.
    
    Schedule (MVP):
    - Ogni 6 ore:   scraping eBay + scoring
    - Ogni 12 ore:  scraping Chrono24
    - Ogni giorno:  aggregazione prezzi + report giornaliero
    - Ogni settimana: retrain modello + health check completo
    
    Flusso per ogni ciclo di scraping:
    1. Scrape nuovi listing
    2. Scoring con il modello
    3. Trova opportunità
    4. Invia alert per STRONG_BUY e BUY
    5. Log tutto
    """

    def __init__(self, config_path="config.json", db_path="watchquant.db"):
        self.config_path = config_path
        self.db_path = db_path
        self.config = load_config(config_path)

        self.alert_engine = AlertEngine(config_path)
        self.aggregator = PriceAggregator(db_path)
        self.health = HealthMonitor(db_path)

        if HAS_SCHEDULER:
            self.scheduler = BlockingScheduler()
        else:
            self.scheduler = None

    # --- Job individuali ---

    def job_scrape_and_score(self, sources=None, demo=False):
        """
        Job principale: scrape → score → alert.
        Questo è il ciclo che gira ogni 6 ore.
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"  JOB: Scrape & Score — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        stats = {"listings_found": 0, "listings_new": 0,
                 "strong_buy": 0, "buy": 0, "watch": 0}

        # Step 1: Scraping
        try:
            orchestrator = ScraperOrchestrator(self.config_path, self.db_path)
            results = orchestrator.run(sources=sources, demo=demo)

            if results:
                for source_data in results.values():
                    stats["listings_found"] += source_data.get("found", 0)
                    stats["listings_new"] += source_data.get("new", 0)

        except Exception as e:
            logger.error(f"Errore scraping: {e}")
            self.alert_engine.send_error("scraper", str(e))
            return stats

        # Step 1.5: Raccolta segnali macro (se disponibile)
        if HAS_SIGNALS:
            try:
                logger.info("  Raccolta segnali macro...")
                setup_signals_db(self.db_path)
                sig_orch = SignalsOrchestrator(self.db_path)
                sig_orch.gold.collect(self.db_path)
                sig_orch.forex.collect(self.db_path)
                sig_orch.markets.collect(self.db_path)
                logger.info("  Segnali macro aggiornati")
            except Exception as e:
                logger.warning(f"Segnali macro non disponibili: {e}")

        # Step 1.6: Marketplace (Vinted + Subito)
        if HAS_MARKETPLACE and not demo:
            try:
                logger.info("  Scraping marketplace (Vinted + Subito)...")
                mp_orch = MarketplaceOrchestrator(self.config_path, self.db_path)
                mp_results = mp_orch.run()
                for mp_data in mp_results.values():
                    stats["listings_found"] += mp_data.get("found", 0)
                    stats["listings_new"] += mp_data.get("new", 0)
            except Exception as e:
                logger.warning(f"Marketplace non disponibili: {e}")

        # Step 2: Scoring (solo se ci sono nuovi listing)
        if stats["listings_new"] > 0:
            try:
                detector = UndervalDetector(self.config_path, self.db_path)
                scored = detector.score_all_listings()

                if scored is not None:
                    stats["strong_buy"] = (scored["signal"] == "STRONG_BUY").sum()
                    stats["buy"] = (scored["signal"] == "BUY").sum()
                    stats["watch"] = (scored["signal"] == "WATCH").sum()

                    # Step 3: Alert per opportunità
                    opportunities = scored[
                        scored["signal"].isin(["STRONG_BUY", "BUY"])
                    ].head(10)  # Max 10 alert per ciclo

                    for _, opp in opportunities.iterrows():
                        self.alert_engine.send_opportunity(opp.to_dict())

            except Exception as e:
                logger.error(f"Errore scoring: {e}")
                self.alert_engine.send_error("scoring", str(e))
        else:
            logger.info("Nessun nuovo listing, skip scoring.")

        logger.info(
            f"  Ciclo completato — Trovati: {stats['listings_found']}, "
            f"Nuovi: {stats['listings_new']}, "
            f"STRONG_BUY: {stats['strong_buy']}, BUY: {stats['buy']}"
        )
        return stats

    def job_daily_aggregation(self):
        """
        Job giornaliero: aggrega prezzi, rileva variazioni, invia report.
        """
        logger.info("\n  JOB: Aggregazione Giornaliera")

        # Aggrega prezzi
        self.aggregator.aggregate_today()

        # Rileva variazioni
        changes = self.aggregator.detect_price_changes(threshold_pct=10)
        if not changes.empty:
            for _, ch in changes.iterrows():
                self.alert_engine.send_price_alert(
                    ch.to_dict(),
                    ch["yesterday_price"],
                    ch["today_price"],
                    ch["change_pct"]
                )

        # Report giornaliero
        conn = get_db_connection(self.db_path)
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

        listings_24h = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE scraped_at > ?", (cutoff,)
        ).fetchone()[0]

        scored_data = conn.execute("""
            SELECT signal, COUNT(*) as cnt FROM listings 
            WHERE signal IS NOT NULL GROUP BY signal
        """).fetchall()

        portfolio_data = conn.execute("""
            SELECT COUNT(*) as count, SUM(total_cost) as total
            FROM portfolio WHERE status = 'holding'
        """).fetchone()

        conn.close()

        stats = {
            "listings_found": listings_24h,
            "listings_new": listings_24h,
            "strong_buy": 0, "buy": 0, "watch": 0,
            "portfolio_count": portfolio_data["count"] if portfolio_data else 0,
            "portfolio_nav": portfolio_data["total"] if portfolio_data else 0,
        }
        for row in scored_data:
            signal_key = row["signal"].lower().replace(" ", "_")
            if signal_key in stats:
                stats[signal_key] = row["cnt"]

        self.alert_engine.send_daily_report(stats)
        self.alert_engine.reset_daily_counter()

    def job_weekly_retrain(self):
        """
        Job settimanale: retrain modello + health check.
        """
        logger.info("\n  JOB: Retrain Settimanale")

        trainer = ModelTrainer(self.config_path, self.db_path)
        success = trainer.full_pipeline()

        if not success:
            self.alert_engine.send_error("model_training", "Retrain settimanale fallito")

        # Health check
        issues = self.health.check_all()
        if issues:
            high_issues = [i for i in issues if i["severity"] == "high"]
            if high_issues:
                for issue in high_issues:
                    self.alert_engine.send_error(
                        issue["component"],
                        issue["message"]
                    )

    # --- Gestione scheduler ---

    def _job_collect_trends(self):
        """Job dedicato per Google Trends (lento, va eseguito separatamente)."""
        if not HAS_SIGNALS:
            return
        try:
            logger.info("  JOB: Raccolta Google Trends")
            setup_signals_db(self.db_path)
            sig_orch = SignalsOrchestrator(self.db_path)
            sig_orch.trends.collect(self.db_path)
        except Exception as e:
            logger.warning(f"Google Trends non disponibile: {e}")

    def start(self, demo=False):
        """
        Avvia lo scheduler automatico.
        Il sistema gira finché non viene fermato con Ctrl+C.
        """
        if not HAS_SCHEDULER:
            logger.error("APScheduler non installato. pip install apscheduler")
            return

        logger.info("\n" + "=" * 60)
        logger.info("  \U0001f680 WatchQuant Scheduler — AVVIO")
        logger.info("=" * 60)

        # Determina le sorgenti da usare
        sources = None
        if demo:
            job_fn_scrape = lambda: self.job_scrape_and_score(demo=True)
            logger.info("  Modalit\u00e0: DEMO (dati simulati)")
        else:
            job_fn_scrape = lambda: self.job_scrape_and_score()
            logger.info("  Modalit\u00e0: PRODUZIONE")

        # Configura i job
        # Scrape + Score: ogni 6 ore
        self.scheduler.add_job(
            job_fn_scrape,
            IntervalTrigger(hours=6),
            id="scrape_and_score",
            name="Scrape + Score",
            next_run_time=datetime.now() + timedelta(seconds=10),  # Prima esecuzione subito
            misfire_grace_time=600
        )

        # Aggregazione giornaliera: ogni giorno alle 23:00
        self.scheduler.add_job(
            self.job_daily_aggregation,
            CronTrigger(hour=23, minute=0),
            id="daily_aggregation",
            name="Aggregazione Giornaliera",
            misfire_grace_time=3600
        )

        # Google Trends: ogni giorno alle 08:00 (separato perché è lento)
        if HAS_SIGNALS:
            self.scheduler.add_job(
                self._job_collect_trends,
                CronTrigger(hour=8, minute=0),
                id="google_trends",
                name="Google Trends Collection",
                misfire_grace_time=3600
            )

        # Retrain settimanale: ogni domenica alle 03:00
        self.scheduler.add_job(
            self.job_weekly_retrain,
            CronTrigger(day_of_week="sun", hour=3, minute=0),
            id="weekly_retrain",
            name="Retrain Settimanale",
            misfire_grace_time=7200
        )

        # Health check giornaliero: ogni giorno alle 09:00
        self.scheduler.add_job(
            lambda: self.health.print_status(),
            CronTrigger(hour=9, minute=0),
            id="health_check",
            name="Health Check",
            misfire_grace_time=3600
        )

        # Reset alert counter: ogni giorno a mezzanotte
        self.scheduler.add_job(
            self.alert_engine.reset_daily_counter,
            CronTrigger(hour=0, minute=0),
            id="reset_alerts",
            name="Reset Alert Counter"
        )

        self.print_schedule()

        # Gestione Ctrl+C
        def shutdown(signum, frame):
            logger.info("\n  \u23f9 Shutdown in corso...")
            self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Notifica avvio
        self.alert_engine.send_message(
            "\u2705 *WatchQuant Scheduler avviato*\n"
            f"Modalit\u00e0: {'DEMO' if demo else 'PRODUZIONE'}\n"
            f"Ora: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )

        # Avvia (bloccante)
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("  Scheduler fermato.")

    def run_once(self, demo=False):
        """Esegue un singolo ciclo completo (senza scheduler)."""
        logger.info("\n  \u25b6\ufe0f Esecuzione singolo ciclo...")

        # Scrape + Score
        stats = self.job_scrape_and_score(demo=demo)

        # Aggregazione
        self.aggregator.aggregate_today()

        # Health check
        self.health.print_status()

        return stats

    def print_schedule(self):
        """Stampa lo schedule programmato."""
        print("\n" + "=" * 60)
        print("  \U0001f4c5 SCHEDULE PROGRAMMATO")
        print("=" * 60)

        schedule = [
            ("Scrape + Score + Signals", "Ogni 6 ore", "Scraping, segnali macro, scoring, alert"),
            ("Google Trends", "Ogni giorno 08:00", "Raccolta interesse di ricerca Google"),
            ("Aggregazione Prezzi", "Ogni giorno 23:00", "Statistiche giornaliere + report"),
            ("Retrain Modello", "Domenica 03:00", "Retrain ML + health check"),
            ("Health Check", "Ogni giorno 09:00", "Verifica salute sistema"),
            ("Reset Alert", "Ogni giorno 00:00", "Reset contatore alert"),
        ]

        for name, freq, desc in schedule:
            print(f"\n  \U0001f552 {name}")
            print(f"     Frequenza: {freq}")
            print(f"     Azione: {desc}")

        print(f"\n  Premi Ctrl+C per fermare.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchQuant — Automazione & Alert"
    )
    parser.add_argument(
        "--start", action="store_true",
        help="Avvia lo scheduler automatico (gira in background)"
    )
    parser.add_argument(
        "--start-demo", action="store_true",
        help="Avvia lo scheduler in modalità demo (dati simulati)"
    )
    parser.add_argument(
        "--run-once", action="store_true",
        help="Esegui un singolo ciclo completo"
    )
    parser.add_argument(
        "--run-once-demo", action="store_true",
        help="Esegui un singolo ciclo in modalità demo"
    )
    parser.add_argument(
        "--test-telegram", action="store_true",
        help="Invia un messaggio di test su Telegram"
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Esegui health check"
    )
    parser.add_argument(
        "--show-schedule", action="store_true",
        help="Mostra lo schedule programmato"
    )
    args = parser.parse_args()

    scheduler = WatchQuantScheduler()

    if args.start:
        scheduler.start(demo=False)
    elif args.start_demo:
        scheduler.start(demo=True)
    elif args.run_once:
        scheduler.run_once(demo=False)
    elif args.run_once_demo:
        scheduler.run_once(demo=True)
    elif args.test_telegram:
        alert = AlertEngine()
        if alert.is_configured():
            success = alert.send_message(
                "\u2705 *Test WatchQuant*\n\n"
                "Se leggi questo messaggio, il bot Telegram funziona!\n"
                f"Ora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            )
            if success:
                print("\n  \u2705 Messaggio inviato con successo!\n")
            else:
                print("\n  \u274c Invio fallito. Controlla token e chat_id.\n")
        else:
            print("\n  \u274c Telegram non configurato.")
            print("  Inserisci telegram_token e telegram_chat_id in config.json")
            print("  Vedi le istruzioni all'inizio di questo file.\n")
    elif args.health:
        scheduler.health.print_status()
    elif args.show_schedule:
        scheduler.print_schedule()
    else:
        print("\n  \U0001f916 WatchQuant — Automazione & Alert")
        print("  " + "\u2500" * 45)
        print("  Opzioni:")
        print("    python automation.py --run-once-demo  \u2192 Test: un ciclo con dati simulati")
        print("    python automation.py --run-once       \u2192 Un ciclo con dati reali")
        print("    python automation.py --start-demo     \u2192 Scheduler automatico (demo)")
        print("    python automation.py --start          \u2192 Scheduler automatico (produzione)")
        print("    python automation.py --test-telegram  \u2192 Test notifica Telegram")
        print("    python automation.py --health         \u2192 Health check sistema")
        print("    python automation.py --show-schedule  \u2192 Mostra schedule programmato")
        print()
        print("  Flusso completo consigliato:")
        print("    1. python foundation.py                \u2192 Setup iniziale")
        print("    2. python scrapers.py --demo           \u2192 Popola dati test")
        print("    3. python models.py --train            \u2192 Addestra modello")
        print("    4. streamlit run dashboard.py          \u2192 Apri dashboard")
        print("    5. python automation.py --start-demo   \u2192 Avvia automazione")
        print()


if __name__ == "__main__":
    main()

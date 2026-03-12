"""
WatchQuant MVP — Scraper Framework
====================================
Questo modulo gestisce tutto il data ingestion:
1. BaseScraper — classe base con retry, logging, rate limiting
2. EBayScraper — API ufficiale eBay Browse
3. Chrono24Scraper — scraping con requests + BeautifulSoup
4. Normalizer — pulizia, normalizzazione condizioni, dedup
5. ScraperOrchestrator — coordina tutto, salva nel DB

COME USARE:
    # Test senza API key (modalità demo con dati simulati):
    python scrapers.py --demo

    # Con API key eBay configurata in config.json:
    python scrapers.py --source ebay

    # Scrape completo (tutte le fonti attive):
    python scrapers.py --all

FILE RICHIESTO: foundation.py deve essere già stato eseguito.
"""

import sqlite3
import json
import os
import re
import time
import random
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import requests
except ImportError:
    print("Installa dipendenze: pip install -r requirements.txt")
    exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Importa utility dal foundation
from foundation import get_db_connection, load_config, get_all_references

# ============================================================
# LOGGING SETUP
# ============================================================

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/scrapers.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WatchQuant.Scrapers")


# ============================================================
# 1. NORMALIZER — Pulizia e standardizzazione dati
# ============================================================

class Normalizer:
    """
    Normalizza i dati grezzi degli scraper:
    - Condizioni → score standard
    - Valuta → EUR
    - Completezza → score
    - Deduplicazione
    """

    # Mappa condizioni: ogni piattaforma usa nomi diversi
    CONDITION_MAP = {
        # Standard
        "new": "new", "nuovo": "new", "neuf": "new",
        "brand new": "new", "sigillato": "new", "sealed": "new",
        "unworn": "new",
        # Like new
        "like new": "like_new", "come nuovo": "like_new",
        "like_new": "like_new", "mint": "like_new",
        "ottime condizioni": "like_new", "excellent": "like_new",
        "très bon état": "like_new", "ottimo": "like_new",
        "very good": "like_new",
        # Good
        "good": "good", "buono": "good", "buone condizioni": "good",
        "bon état": "good", "gut": "good", "used": "good",
        "pre-owned": "good", "usato": "good",
        # Fair
        "fair": "fair", "discreto": "fair", "acceptable": "fair",
        "accettabile": "fair", "segni di usura": "fair",
        # Poor
        "poor": "poor", "da revisionare": "poor",
        "for parts": "poor", "per ricambi": "poor",
    }

    CONDITION_SCORES = {
        "new": 4, "like_new": 3, "good": 2, "fair": 1, "poor": 0
    }

    # Parole chiave per rilevare box/papers dal titolo
    BOX_KEYWORDS = ["box", "scatola", "cofanetto", "full set", "fullset", "completo"]
    PAPERS_KEYWORDS = ["papers", "documenti", "garanzia", "card", "certificato", "warranty"]

    def normalize_condition(self, raw_condition):
        """Mappa una condizione grezza allo standard."""
        if not raw_condition:
            return "good", 2  # default conservativo

        clean = raw_condition.lower().strip()
        for key, value in self.CONDITION_MAP.items():
            if key in clean:
                return value, self.CONDITION_SCORES[value]

        return "good", 2  # default se non riconosciuto

    def detect_completeness(self, title="", description=""):
        """Rileva box/papers/warranty dal testo dell'annuncio."""
        text = f"{title} {description}".lower()
        has_box = any(kw in text for kw in self.BOX_KEYWORDS)
        has_papers = any(kw in text for kw in self.PAPERS_KEYWORDS)
        has_warranty = "warranty" in text or "garanzia attiva" in text
        score = int(has_box) + int(has_papers) + int(has_warranty)
        return has_box, has_papers, has_warranty, score

    def normalize_price(self, price_value, currency="EUR"):
        """
        Converte il prezzo in EUR.
        Per l'MVP usiamo tassi fissi approssimativi.
        In v1 useremo frankfurter.app per tassi giornalieri.
        """
        if not price_value or price_value <= 0:
            return None

        rates_to_eur = {
            "EUR": 1.0,
            "USD": 0.92,
            "GBP": 1.17,
            "CHF": 1.04,
            "JPY": 0.0061,
            "SEK": 0.087,
            "PLN": 0.23,
        }
        rate = rates_to_eur.get(currency.upper(), 1.0)
        return round(price_value * rate, 2)

    def parse_price_text(self, text):
        """Estrae un prezzo numerico da testo tipo '€ 1.234,56' o '1,234.56 EUR'."""
        if not text:
            return None, "EUR"

        text = text.strip()

        # Rileva valuta
        currency = "EUR"
        currency_symbols = {
            "€": "EUR", "EUR": "EUR", "eur": "EUR",
            "$": "USD", "USD": "USD",
            "£": "GBP", "GBP": "GBP",
            "CHF": "CHF", "¥": "JPY", "JPY": "JPY",
        }
        for symbol, curr in currency_symbols.items():
            if symbol in text:
                currency = curr
                break

        # Rimuovi tutto tranne numeri, punti, virgole
        clean = re.sub(r"[^\d.,]", "", text)
        if not clean:
            return None, currency

        # Gestisci formati europei (1.234,56) e americani (1,234.56)
        if "," in clean and "." in clean:
            if clean.rindex(",") > clean.rindex("."):
                # Formato europeo: 1.234,56
                clean = clean.replace(".", "").replace(",", ".")
            else:
                # Formato americano: 1,234.56
                clean = clean.replace(",", "")
        elif "," in clean:
            # Solo virgola: potrebbe essere decimale (3,50) o migliaia (1,234)
            parts = clean.split(",")
            if len(parts[-1]) == 2:
                clean = clean.replace(",", ".")  # decimale
            else:
                clean = clean.replace(",", "")  # migliaia

        try:
            return float(clean), currency
        except ValueError:
            return None, currency

    def generate_dedup_hash(self, reference, condition, seller_location, price):
        """
        Genera un hash per identificare potenziali duplicati cross-piattaforma.
        Stesso orologio su eBay e Subito → stesso hash.
        """
        key = f"{reference}|{condition}|{seller_location}|{round(price, -1)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def normalize_listing(self, raw):
        """
        Normalizza un listing grezzo in formato standard per il DB.
        Input: dizionario grezzo dallo scraper
        Output: dizionario pronto per INSERT
        """
        # Prezzo
        if isinstance(raw.get("price"), str):
            price, currency = self.parse_price_text(raw["price"])
        else:
            price = raw.get("price")
            currency = raw.get("currency", "EUR")

        price_eur = self.normalize_price(price, currency)
        if not price_eur or price_eur < 10:
            return None  # Scarta listing senza prezzo valido

        # Condizione
        condition, condition_score = self.normalize_condition(
            raw.get("condition", "")
        )

        # Completezza
        title = raw.get("title", "")
        has_box, has_papers, has_warranty, completeness = self.detect_completeness(
            title, raw.get("description", "")
        )
        # Sovrascrivi se lo scraper ha dati espliciti
        if raw.get("has_box") is not None:
            has_box = bool(raw["has_box"])
        if raw.get("has_papers") is not None:
            has_papers = bool(raw["has_papers"])
        completeness = int(has_box) + int(has_papers) + int(has_warranty)

        return {
            "source": raw.get("source", "unknown"),
            "external_id": raw.get("external_id", ""),
            "title": title,
            "price": price_eur,
            "currency_original": currency,
            "condition": condition,
            "condition_score": condition_score,
            "has_box": int(has_box),
            "has_papers": int(has_papers),
            "has_warranty": int(has_warranty),
            "completeness_score": completeness,
            "dial_variant": raw.get("dial_variant"),
            "seller_location": raw.get("seller_location"),
            "url": raw.get("url"),
            "image_urls": json.dumps(raw.get("image_urls", [])),
            "status": "active",
        }


# ============================================================
# 2. BASE SCRAPER — Classe base con funzionalità comuni
# ============================================================

class BaseScraper(ABC):
    """Classe base per tutti gli scraper."""

    def __init__(self, config, db_path="watchquant.db"):
        self.config = config
        self.db_path = db_path
        self.normalizer = Normalizer()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        })

    @property
    @abstractmethod
    def source_name(self):
        """Nome della fonte (es. 'ebay', 'chrono24')."""
        pass

    @abstractmethod
    def search_reference(self, brand, model, reference, **kwargs):
        """Cerca listing per una referenza specifica. Ritorna lista di dict grezzi."""
        pass

    def rate_limit_pause(self):
        """Pausa tra le richieste per rispettare rate limits."""
        delay = self.config.get("scraping", {}).get(
            "rate_limits", {}
        ).get("requests_per_minute", 10)
        pause = 60.0 / delay + random.uniform(0.5, 2.0)
        time.sleep(pause)

    def save_listings(self, watch_id, raw_listings):
        """Normalizza e salva i listing nel database."""
        conn = get_db_connection(self.db_path)
        saved = 0
        skipped = 0

        for raw in raw_listings:
            normalized = self.normalizer.normalize_listing(raw)
            if not normalized:
                skipped += 1
                continue

            try:
                conn.execute("""
                    INSERT INTO listings 
                        (watch_id, source, external_id, title, price, 
                         currency_original, condition, condition_score,
                         has_box, has_papers, has_warranty, completeness_score,
                         dial_variant, seller_location, url, image_urls, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    watch_id, normalized["source"], normalized["external_id"],
                    normalized["title"], normalized["price"],
                    normalized["currency_original"], normalized["condition"],
                    normalized["condition_score"], normalized["has_box"],
                    normalized["has_papers"], normalized["has_warranty"],
                    normalized["completeness_score"], normalized["dial_variant"],
                    normalized["seller_location"], normalized["url"],
                    normalized["image_urls"], normalized["status"]
                ))
                saved += 1
            except sqlite3.IntegrityError:
                # Listing già presente (source + external_id duplicato)
                skipped += 1

        conn.commit()
        conn.close()
        return saved, skipped

    def log_scrape(self, status, listings_found=0, listings_new=0, error=None):
        """Registra la sessione di scraping nel log."""
        conn = get_db_connection(self.db_path)
        conn.execute("""
            INSERT INTO scrape_log 
                (source, status, listings_found, listings_new, error_message, finished_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (self.source_name, status, listings_found, listings_new, error))
        conn.commit()
        conn.close()

    def scrape_all_references(self):
        """Scrape tutte le referenze nel catalogo."""
        references = get_all_references(self.db_path)
        total_found = 0
        total_new = 0

        logger.info(f"[{self.source_name}] Inizio scrape di {len(references)} referenze")

        for ref in references:
            try:
                logger.info(f"  → {ref['brand']} {ref['model']} ({ref['reference']})")
                raw_listings = self.search_reference(
                    ref["brand"], ref["model"], ref["reference"]
                )
                if raw_listings:
                    saved, skipped = self.save_listings(ref["id"], raw_listings)
                    total_found += len(raw_listings)
                    total_new += saved
                    logger.info(f"    Trovati: {len(raw_listings)}, Nuovi: {saved}, Duplicati: {skipped}")

                self.rate_limit_pause()

            except Exception as e:
                logger.error(f"    ERRORE: {e}")
                continue

        self.log_scrape("success", total_found, total_new)
        logger.info(
            f"[{self.source_name}] Completato — "
            f"Trovati: {total_found}, Nuovi: {total_new}"
        )
        return total_found, total_new


# ============================================================
# 3. EBAY SCRAPER — API ufficiale Browse
# ============================================================

class EBayScraper(BaseScraper):
    """
    Scraper eBay tramite API ufficiale Browse.
    Richiede app_id e cert_id configurati in config.json.
    Gratuito fino a 5000 chiamate/giorno.
    """

    source_name = "ebay"

    BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"

    def __init__(self, config, db_path="watchquant.db"):
        super().__init__(config, db_path)
        ebay_cfg = config.get("scraping", {}).get("ebay", {})
        self.app_id = ebay_cfg.get("app_id", "")
        self.cert_id = ebay_cfg.get("cert_id", "")
        self.marketplace = ebay_cfg.get("marketplace", "EBAY_IT")
        self.category_id = ebay_cfg.get("category_id", "31387")
        self._token = None
        self._token_expiry = None

    def is_configured(self):
        """Verifica se le API key sono presenti."""
        return bool(self.app_id and self.cert_id)

    def _get_token(self):
        """Ottieni OAuth token da eBay."""
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token

        resp = requests.post(
            self.TOKEN_URL,
            auth=(self.app_id, self.cert_id),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope"
            }
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = datetime.now() + timedelta(
            seconds=data.get("expires_in", 7200) - 300
        )
        logger.info("[eBay] Token OAuth ottenuto")
        return self._token

    def search_reference(self, brand, model, reference, limit=50):
        """Cerca listing eBay per una referenza."""
        if not self.is_configured():
            logger.warning("[eBay] API key non configurate. Salta.")
            return []

        token = self._get_token()
        query = f"{brand} {reference}"

        headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace,
            "X-EBAY-C-ENDUSERCTX": "contextualLocation=country=IT"
        }
        params = {
            "q": query,
            "category_ids": self.category_id,
            "filter": "conditionIds:{1000|1500|2000|2500|3000}",
            "limit": min(limit, 200),
            "sort": "newlyListed",
        }

        try:
            resp = self.session.get(
                self.BROWSE_URL, headers=headers, params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[eBay] Errore API per {reference}: {e}")
            return []

        listings = []
        for item in data.get("itemSummaries", []):
            price_data = item.get("price", {})
            condition = item.get("condition", "")
            location = item.get("itemLocation", {})

            listings.append({
                "source": "ebay",
                "external_id": item.get("itemId", ""),
                "title": item.get("title", ""),
                "price": float(price_data.get("value", 0)),
                "currency": price_data.get("currency", "EUR"),
                "condition": condition,
                "seller_location": location.get("country", ""),
                "url": item.get("itemWebUrl", ""),
                "image_urls": [
                    img.get("imageUrl", "")
                    for img in item.get("additionalImages", [])[:3]
                ] if item.get("additionalImages") else [],
            })

        return listings


# ============================================================
# 4. CHRONO24 SCRAPER — requests + BeautifulSoup
# ============================================================

class Chrono24Scraper(BaseScraper):
    """
    Scraper Chrono24 con requests + BeautifulSoup.
    
    ATTENZIONE: Chrono24 ha protezioni anti-bot.
    Per l'MVP usiamo requests semplici con cautela.
    Se bloccati, sarà necessario:
    - Proxy rotanti (v1)
    - Selenium/Playwright (v1)
    - Headers realistici + cookie management
    """

    source_name = "chrono24"

    BASE_URL = "https://www.chrono24.it"

    def __init__(self, config, db_path="watchquant.db"):
        super().__init__(config, db_path)
        c24_cfg = config.get("scraping", {}).get("chrono24", {})
        self.enabled = c24_cfg.get("enabled", False)
        self.proxy = c24_cfg.get("proxy", "")
        self.max_pages = c24_cfg.get("max_pages", 2)
        self.delay_min = c24_cfg.get("delay_min_sec", 4)
        self.delay_max = c24_cfg.get("delay_max_sec", 10)

        # Rotazione user agent per sembrare più realistici
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        ]

    def _get_proxies(self):
        if self.proxy:
            return {"http": self.proxy, "https": self.proxy}
        return None

    def _rotate_headers(self):
        self.session.headers["User-Agent"] = random.choice(self.user_agents)

    def search_reference(self, brand, model, reference, **kwargs):
        """Cerca listing su Chrono24 per una referenza."""
        if not self.enabled:
            logger.info("[Chrono24] Disabilitato in config. Salta.")
            return []

        if BeautifulSoup is None:
            logger.error("[Chrono24] beautifulsoup4 non installato.")
            return []

        all_listings = []

        for page in range(1, self.max_pages + 1):
            self._rotate_headers()
            url = (
                f"{self.BASE_URL}/search/index.htm"
                f"?query={reference}&pageSize=60&showpage={page}"
            )

            try:
                resp = self.session.get(
                    url,
                    proxies=self._get_proxies(),
                    timeout=20
                )

                if resp.status_code == 403:
                    logger.warning(f"[Chrono24] Bloccato (403). Probabile anti-bot.")
                    break
                if resp.status_code == 429:
                    logger.warning(f"[Chrono24] Rate limited (429). Pausa lunga.")
                    time.sleep(60)
                    break

                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Parsing delle card — i selettori CSS possono cambiare
                # Chrono24 aggiorna spesso il layout, quindi questo va monitorato
                cards = soup.select("div.article-item-container")
                if not cards:
                    # Prova selettore alternativo
                    cards = soup.select("[class*='article-item']")

                if not cards:
                    logger.info(f"[Chrono24] Nessun risultato pagina {page} per {reference}")
                    break

                for card in cards:
                    listing = self._parse_card(card, reference)
                    if listing:
                        all_listings.append(listing)

                logger.info(
                    f"[Chrono24] Pagina {page}/{self.max_pages}: "
                    f"{len(cards)} card trovate per {reference}"
                )

            except requests.RequestException as e:
                logger.error(f"[Chrono24] Errore pagina {page} per {reference}: {e}")
                break

            # Pausa anti-detection tra le pagine
            time.sleep(random.uniform(self.delay_min, self.delay_max))

        return all_listings

    def _parse_card(self, card, reference):
        """Parsa una card HTML di Chrono24 in un dict grezzo."""
        try:
            # Titolo
            title_el = card.select_one(".article-title") or card.select_one("a[class*='title']")
            title = title_el.text.strip() if title_el else ""

            # Prezzo
            price_el = card.select_one(".article-price") or card.select_one("[class*='price']")
            price_text = price_el.text.strip() if price_el else ""
            price, currency = self.normalizer.parse_price_text(price_text)

            if not price:
                return None

            # URL
            link_el = card.select_one("a[href]")
            url = ""
            if link_el and link_el.get("href"):
                href = link_el["href"]
                url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

            # ID esterno dall'URL
            external_id = ""
            if url:
                match = re.search(r"--id(\d+)\.htm", url)
                if match:
                    external_id = f"c24_{match.group(1)}"

            # Condizione (se presente)
            condition_el = card.select_one("[class*='condition']")
            condition = condition_el.text.strip() if condition_el else ""

            return {
                "source": "chrono24",
                "external_id": external_id or f"c24_{hashlib.md5(url.encode()).hexdigest()[:8]}",
                "title": title,
                "price": price,
                "currency": currency,
                "condition": condition,
                "url": url,
                "seller_location": "",
                "image_urls": [],
            }
        except Exception as e:
            logger.debug(f"[Chrono24] Errore parsing card: {e}")
            return None


# ============================================================
# 5. DEMO SCRAPER — Per testare il flusso senza API key
# ============================================================

class DemoScraper(BaseScraper):
    """
    Genera dati simulati ma realistici per testare
    l'intero flusso senza API key.
    Prezzi basati su range reali del mercato secondario.
    """

    source_name = "demo"

    # Range di prezzo realistici per il secondario (EUR)
    PRICE_RANGES = {
        "Seiko": {"Presage": (180, 380), "Prospex": (250, 500),
                   "Seiko 5": (100, 220), "SKX": (250, 550)},
        "Orient": {"Bambino": (80, 170), "Kamasu": (130, 250),
                    "Ray": (100, 190)},
        "Casio": {"Casioak": (60, 150), "G-Shock 5000": (280, 480),
                   "G-Shock 5600": (35, 80)},
        "Tissot": {"PRX": (200, 550), "Gentleman": (280, 520)},
        "Hamilton": {"Khaki Field": (250, 500), "Khaki Aviation": (350, 600)},
        "Swatch": {"MoonSwatch": (200, 450)},
        "Citizen": {"Promaster": (100, 250)},
        "Vostok": {"Amphibia": (40, 120)},
        "Timex": {"Marlin": (120, 230)},
    }

    CONDITIONS = ["new", "like_new", "like_new", "good", "good", "good", "fair"]
    SOURCES_FAKE = ["ebay", "chrono24", "vinted", "subito"]
    LOCATIONS = ["IT", "DE", "FR", "UK", "US", "JP", "CH"]

    def search_reference(self, brand, model, reference, **kwargs):
        """Genera 5-15 listing finti ma realistici."""
        family = kwargs.get("model_family", model.split()[0])
        brand_ranges = self.PRICE_RANGES.get(brand, {})
        price_range = brand_ranges.get(family, (100, 400))

        num_listings = random.randint(5, 15)
        listings = []

        for i in range(num_listings):
            condition = random.choice(self.CONDITIONS)
            base_price = random.uniform(*price_range)

            # Aggiusta prezzo per condizione
            condition_multiplier = {
                "new": 1.15, "like_new": 1.0, "good": 0.85, "fair": 0.70
            }
            price = base_price * condition_multiplier.get(condition, 1.0)
            price = round(price + random.uniform(-20, 20), 2)

            has_box = random.random() > 0.4
            has_papers = random.random() > 0.5

            source = random.choice(self.SOURCES_FAKE)
            listings.append({
                "source": source,
                "external_id": f"demo_{reference}_{source}_{i}_{random.randint(1000,9999)}",
                "title": f"{brand} {model} Ref. {reference} - {condition}",
                "price": max(price, 20),
                "currency": "EUR",
                "condition": condition,
                "has_box": has_box,
                "has_papers": has_papers,
                "seller_location": random.choice(self.LOCATIONS),
                "url": f"https://example.com/demo/{reference}/{i}",
                "image_urls": [],
            })

        return listings


# ============================================================
# 6. ORCHESTRATOR — Coordina tutti gli scraper
# ============================================================

class ScraperOrchestrator:
    """Coordina l'esecuzione degli scraper e la raccolta dati."""

    def __init__(self, config_path="config.json", db_path="watchquant.db"):
        self.config = load_config(config_path)
        self.db_path = db_path

        # Inizializza tutti gli scraper
        self.scrapers = {
            "ebay": EBayScraper(self.config, db_path),
            "chrono24": Chrono24Scraper(self.config, db_path),
            "demo": DemoScraper(self.config, db_path),
        }

    def run(self, sources=None, demo=False):
        """
        Esegue lo scraping.
        
        sources: lista di nomi scraper da eseguire (es. ['ebay', 'chrono24'])
        demo: se True, usa il DemoScraper per test
        """
        if demo:
            sources = ["demo"]
        elif sources is None:
            # Esegui tutti gli scraper configurati
            sources = []
            if self.scrapers["ebay"].is_configured():
                sources.append("ebay")
            if self.scrapers["chrono24"].enabled:
                sources.append("chrono24")

        if not sources:
            logger.warning(
                "Nessuna fonte attiva. Usa --demo per testare, "
                "oppure configura le API key in config.json"
            )
            return

        results = {}
        for source_name in sources:
            scraper = self.scrapers.get(source_name)
            if not scraper:
                logger.warning(f"Scraper '{source_name}' non trovato. Salto.")
                continue

            try:
                found, new = scraper.scrape_all_references()
                results[source_name] = {"found": found, "new": new}
            except Exception as e:
                logger.error(f"Errore con scraper {source_name}: {e}")
                scraper.log_scrape("error", error=str(e))
                results[source_name] = {"found": 0, "new": 0, "error": str(e)}

        # Report finale
        self._print_report(results)
        return results

    def _print_report(self, results):
        """Stampa il report di fine scraping."""
        print("\n" + "=" * 50)
        print("  📊 REPORT SCRAPING")
        print("=" * 50)
        total_found = 0
        total_new = 0
        for source, data in results.items():
            found = data.get("found", 0)
            new = data.get("new", 0)
            total_found += found
            total_new += new
            err = data.get("error", "")
            status = "❌ " + err if err else "✅"
            print(f"  {source:.<20} Trovati: {found:>4}  Nuovi: {new:>4}  {status}")
        print("  " + "─" * 46)
        print(f"  {'TOTALE':.<20} Trovati: {total_found:>4}  Nuovi: {total_new:>4}")
        print()


# ============================================================
# MAIN — Esecuzione da riga di comando
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchQuant Scraper — Raccolta dati mercato orologi"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Esegui con dati simulati (per test senza API key)"
    )
    parser.add_argument(
        "--source", type=str, choices=["ebay", "chrono24"],
        help="Esegui solo una fonte specifica"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Esegui tutte le fonti configurate"
    )
    args = parser.parse_args()

    orchestrator = ScraperOrchestrator()

    if args.demo:
        orchestrator.run(demo=True)
    elif args.source:
        orchestrator.run(sources=[args.source])
    elif args.all:
        orchestrator.run()
    else:
        # Default: se nessuna fonte è configurata, suggerisci demo
        print("\n  🕐 WatchQuant Scraper")
        print("  " + "─" * 40)
        print("  Nessuna opzione specificata.\n")
        print("  Opzioni:")
        print("    python scrapers.py --demo       → Test con dati simulati")
        print("    python scrapers.py --source ebay → Solo eBay")
        print("    python scrapers.py --all         → Tutte le fonti attive")
        print()


if __name__ == "__main__":
    main()

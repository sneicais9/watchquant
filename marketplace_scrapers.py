"""
WatchQuant — Marketplace Scrapers (Vinted + Subito)
=====================================================
Scraper per i marketplace generalisti italiani/europei dove si trovano
spesso orologi sottovalutati perché i venditori non conoscono il valore.

1. VintedScraper — Vinted.it/Vinted.fr (API interna)
2. SubitoScraper — Subito.it (requests + BeautifulSoup)
3. MarketplaceOrchestrator — coordina tutto

PERCHÉ QUESTI MARKETPLACE:
- I venditori spesso non sono collezionisti → prezzi più bassi
- Meno competizione rispetto a Chrono24/eBay
- Listing non standardizzati = opportunità per chi ha un modello AI
- Vinted ha commissioni basse → margini migliori

COME USARE:
    # Scan completo di tutti i marketplace:
    python marketplace_scrapers.py --all

    # Solo Vinted:
    python marketplace_scrapers.py --vinted

    # Solo Subito:
    python marketplace_scrapers.py --subito

    # Report risultati:
    python marketplace_scrapers.py --report

NOTE LEGALI:
    - Usiamo solo dati pubblicamente visibili
    - Rate limiting rispettoso (max 10 req/min)
    - Nessun login richiesto
    - Uso personale, non commerciale
"""

import re
import json
import time
import random
import hashlib
import logging
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installa: python3 -m pip install requests beautifulsoup4")
    exit(1)

from foundation import get_db_connection, load_config, get_all_references
from scrapers import BaseScraper, Normalizer

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/marketplace.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WatchQuant.Marketplace")


# ============================================================
# 1. VINTED SCRAPER
# ============================================================

class VintedScraper(BaseScraper):
    """
    Scraper per Vinted tramite la loro API interna.
    
    Vinted ha un'API pubblica non documentata ma accessibile:
    - Non richiede autenticazione per la ricerca
    - Ritorna JSON strutturato con prezzi, foto, condizioni
    - Rate limit gentile se non si esagera
    
    I listing su Vinted sono spesso sottovalutati perché:
    - I venditori non sono collezionisti
    - La piattaforma è percepita come "low-end"
    - Le condizioni sono spesso descritte male
    """

    source_name = "vinted"

    CATALOG_URL = "https://www.vinted.it/api/v2/catalog/items"
    
    # Vinted richiede un cookie di sessione per l'API
    # Lo otteniamo visitando la homepage
    HOME_URL = "https://www.vinted.it"

    # Category ID per orologi su Vinted
    WATCH_CATEGORY = "2066"  # Accessori > Orologi

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    ]

    def __init__(self, config, db_path="watchquant.db"):
        super().__init__(config, db_path)
        self._session_cookie = None

    def _init_session(self):
        """Ottieni un cookie di sessione visitando la homepage."""
        if self._session_cookie:
            return True

        self.session.headers["User-Agent"] = random.choice(self.USER_AGENTS)

        try:
            resp = self.session.get(self.HOME_URL, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                # Vinted setta dei cookie necessari per l'API
                self._session_cookie = True
                logger.info("[Vinted] Sessione inizializzata")
                return True
            else:
                logger.warning(f"[Vinted] Homepage status {resp.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"[Vinted] Errore init sessione: {e}")
            return False

    def search_reference(self, brand, model, reference, **kwargs):
        """Cerca listing su Vinted per una referenza."""
        if not self._init_session():
            return []

        # Vinted funziona meglio con ricerche per brand + modello
        # piuttosto che per referenza esatta
        queries = [
            f"{brand} {reference}",
            f"{brand} {model}",
        ]

        all_listings = []
        seen_ids = set()

        for query in queries:
            try:
                self.session.headers["User-Agent"] = random.choice(self.USER_AGENTS)

                params = {
                    "search_text": query,
                    "catalog_ids": self.WATCH_CATEGORY,
                    "order": "newest_first",
                    "per_page": 20,
                    "page": 1,
                }

                resp = self.session.get(
                    self.CATALOG_URL,
                    params=params,
                    timeout=15
                )

                if resp.status_code == 401:
                    # Cookie scaduto, riprova
                    self._session_cookie = None
                    self._init_session()
                    resp = self.session.get(
                        self.CATALOG_URL, params=params, timeout=15
                    )

                if resp.status_code != 200:
                    logger.debug(f"[Vinted] HTTP {resp.status_code} per '{query}'")
                    continue

                data = resp.json()
                items = data.get("items", [])

                for item in items:
                    item_id = str(item.get("id", ""))
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)

                    listing = self._parse_item(item, reference)
                    if listing:
                        all_listings.append(listing)

            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.debug(f"[Vinted] Errore per '{query}': {e}")

            time.sleep(random.uniform(2, 5))

        return all_listings

    def _parse_item(self, item, reference):
        """Parsa un item Vinted in formato standard."""
        try:
            price = item.get("price", "0")
            if isinstance(price, str):
                price = float(price.replace(",", "."))
            elif isinstance(price, dict):
                price = float(price.get("amount", "0"))
            else:
                price = float(price)

            if price < 10 or price > 10000:
                return None

            title = item.get("title", "")
            description = item.get("description", "")

            # URL
            url = item.get("url", "")
            if url and not url.startswith("http"):
                url = f"https://www.vinted.it{url}"

            # Foto
            photos = item.get("photos", [])
            image_urls = [p.get("url", "") for p in photos[:3] if p.get("url")]

            # Posizione
            user = item.get("user", {})
            location = user.get("city", "")
            country = user.get("country_title", "IT")

            return {
                "source": "vinted",
                "external_id": f"vinted_{item.get('id', '')}",
                "title": title,
                "price": price,
                "currency": item.get("currency", "EUR"),
                "condition": self._map_condition(item.get("status", "")),
                "seller_location": f"{location}, {country}" if location else country,
                "url": url,
                "image_urls": image_urls,
                "description": description,
            }
        except Exception as e:
            logger.debug(f"[Vinted] Errore parsing: {e}")
            return None

    def _map_condition(self, vinted_status):
        """Mappa le condizioni Vinted al nostro standard."""
        status_map = {
            "new_with_tags": "new",
            "new_without_tags": "like_new",
            "very_good": "like_new",
            "good": "good",
            "satisfactory": "fair",
        }
        return status_map.get(vinted_status, "good")


# ============================================================
# 2. SUBITO.IT SCRAPER
# ============================================================

class SubitoScraper(BaseScraper):
    """
    Scraper per Subito.it tramite requests + BeautifulSoup.
    
    Subito è interessante perché:
    - Molti venditori non conoscono il valore degli orologi
    - Nessuna commissione → i prezzi riflettono il valore percepito
    - Mercato prevalentemente italiano
    - Listing spesso sottovalutati nella fascia €50-€500
    
    NOTA: Subito ha protezioni anti-bot moderate.
    Usiamo rate limiting rispettoso e user-agent rotation.
    """

    source_name = "subito"

    SEARCH_URL = "https://www.subito.it/annunci-italia/vendita/accessori/"
    
    # Subito ha anche un endpoint API-like per la ricerca
    API_URL = "https://hades.subito.it/v1/search/items"

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
    ]

    def search_reference(self, brand, model, reference, **kwargs):
        """Cerca listing su Subito.it per una referenza."""
        queries = [
            f"orologio {brand} {reference}",
            f"{brand} {model}",
        ]

        all_listings = []
        seen_urls = set()

        for query in queries:
            # Metodo 1: API Hades (più strutturato)
            listings = self._search_api(query)

            # Metodo 2: Fallback HTML scraping
            if not listings:
                listings = self._search_html(query)

            for listing in listings:
                url = listing.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_listings.append(listing)

            time.sleep(random.uniform(3, 7))

        return all_listings

    def _search_api(self, query):
        """Cerca via API Hades di Subito."""
        listings = []
        try:
            self.session.headers.update({
                "User-Agent": random.choice(self.USER_AGENTS),
                "Accept": "application/json",
                "X-Subito-Channel": "www",
            })

            params = {
                "q": query,
                "lim": "30",
                "start": "0",
                "sort": "datedesc",
            }

            resp = self.session.get(
                self.API_URL,
                params=params,
                timeout=15,
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            ads = data.get("ads", [])

            for ad in ads:
                listing = self._parse_api_item(ad)
                if listing:
                    listings.append(listing)

            logger.info(f"[Subito] '{query}': {len(ads)} annunci, {len(listings)} con prezzo valido")

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.debug(f"[Subito/API] Errore per '{query}': {e}")

        return listings

    def _parse_api_item(self, ad):
        """Parsa un annuncio dall'API Hades."""
        try:
            features = ad.get("features", [])

            # Estrai prezzo da features uri="/price"
            price = None
            for feat in features:
                if feat.get("uri") == "/price":
                    values = feat.get("values", [])
                    if values:
                        price_str = values[0].get("value", "")
                        price = self._parse_price(price_str)
                    break

            if not price or price < 10 or price > 10000:
                return None

            # Condizione da features uri="/item_condition"
            condition = "good"
            for feat in features:
                if feat.get("uri") == "/item_condition":
                    values = feat.get("values", [])
                    if values:
                        cond_text = values[0].get("value", "").lower()
                        if "nuovo" in cond_text or "mai" in cond_text:
                            condition = "new"
                        elif "ottime" in cond_text or "come nuovo" in cond_text:
                            condition = "like_new"
                        elif "buono" in cond_text:
                            condition = "good"
                        elif "discreto" in cond_text or "usato" in cond_text:
                            condition = "fair"
                    break

            title = ad.get("subject", "")
            body = ad.get("body", "")

            # URL
            urls = ad.get("urls", {})
            url = urls.get("default", "")

            # Immagini
            images = ad.get("images", [])
            image_urls = [img.get("base_url", "") for img in images[:3] if img.get("base_url")]

            # Posizione
            geo = ad.get("geo", {})
            city = geo.get("city", {}).get("value", "")
            region = geo.get("region", {}).get("value", "")

            return {
                "source": "subito",
                "external_id": f"subito_{ad.get('urn', '')}",
                "title": title,
                "price": price,
                "currency": "EUR",
                "condition": condition,
                "seller_location": f"{city}, {region}" if city else region,
                "url": url,
                "image_urls": image_urls,
                "description": body,
            }
        except Exception as e:
            logger.debug(f"[Subito] Errore parsing API: {e}")
            return None

    def _search_html(self, query):
        """Fallback: cerca via HTML scraping."""
        listings = []
        try:
            self.session.headers["User-Agent"] = random.choice(self.USER_AGENTS)

            url = f"https://www.subito.it/annunci-italia/vendita/accessori/?q={quote_plus(query)}"
            resp = self.session.get(url, timeout=15)

            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            # Subito usa diverse strutture CSS, proviamo i pattern comuni
            items = soup.select("div.items__item, div.item-card, a.items__item")

            if not items:
                # Pattern alternativo
                items = soup.select("[class*='ItemCard'], [class*='item-card']")

            for item in items[:20]:
                listing = self._parse_html_item(item)
                if listing:
                    listings.append(listing)

        except requests.RequestException as e:
            logger.debug(f"[Subito/HTML] Errore per '{query}': {e}")

        return listings

    def _parse_html_item(self, item):
        """Parsa un elemento HTML di Subito."""
        try:
            # Titolo
            title_el = (
                item.select_one("[class*='title'], h2, .item-title") or
                item.select_one("a")
            )
            title = title_el.text.strip() if title_el else ""
            if not title:
                return None

            # Prezzo
            price_el = item.select_one("[class*='price'], .price, p.price")
            price_text = price_el.text.strip() if price_el else ""
            price = self._parse_price(price_text)

            if not price or price < 10 or price > 10000:
                return None

            # URL
            link = item.select_one("a[href]")
            url = ""
            if link and link.get("href"):
                href = link["href"]
                url = href if href.startswith("http") else f"https://www.subito.it{href}"

            # Posizione
            loc_el = item.select_one("[class*='town'], [class*='location'], .city")
            location = loc_el.text.strip() if loc_el else ""

            return {
                "source": "subito",
                "external_id": f"subito_{hashlib.md5(url.encode()).hexdigest()[:10]}",
                "title": title,
                "price": price,
                "currency": "EUR",
                "condition": "good",
                "seller_location": location,
                "url": url,
                "image_urls": [],
            }
        except Exception as e:
            logger.debug(f"[Subito/HTML] Errore parsing: {e}")
            return None

    def _parse_price(self, text):
        """Estrae prezzo numerico da testo italiano."""
        if not text:
            return None
        clean = re.sub(r"[^\d.,]", "", str(text))
        if not clean:
            return None
        # Formato italiano: 1.234,56
        if "," in clean and "." in clean:
            clean = clean.replace(".", "").replace(",", ".")
        elif "," in clean:
            parts = clean.split(",")
            if len(parts[-1]) <= 2:
                clean = clean.replace(",", ".")
            else:
                clean = clean.replace(",", "")
        try:
            return float(clean)
        except ValueError:
            return None


# ============================================================
# 3. MARKETPLACE ORCHESTRATOR
# ============================================================

class MarketplaceOrchestrator:
    """Coordina Vinted e Subito scrapers."""

    def __init__(self, config_path="config.json", db_path="watchquant.db"):
        self.config = load_config(config_path)
        self.db_path = db_path
        self.scrapers = {
            "vinted": VintedScraper(self.config, db_path),
            "subito": SubitoScraper(self.config, db_path),
        }

    def run(self, sources=None):
        """Esegue scraping su marketplace selezionati."""
        if sources is None:
            sources = ["vinted", "subito"]

        results = {}
        for name in sources:
            scraper = self.scrapers.get(name)
            if not scraper:
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"  SCRAPING: {name.upper()}")
            logger.info(f"{'='*50}")

            try:
                found, new = scraper.scrape_all_references()
                results[name] = {"found": found, "new": new}
            except Exception as e:
                logger.error(f"  Errore {name}: {e}")
                scraper.log_scrape("error", error=str(e))
                results[name] = {"found": 0, "new": 0, "error": str(e)}

        self._print_report(results)
        return results

    def _print_report(self, results):
        print("\n" + "=" * 55)
        print("  📊 REPORT MARKETPLACE")
        print("=" * 55)
        total_found = 0
        total_new = 0
        for source, data in results.items():
            found = data.get("found", 0)
            new = data.get("new", 0)
            total_found += found
            total_new += new
            err = data.get("error", "")
            status = f"❌ {err[:40]}" if err else "✅"
            print(f"  {source:.<20} Trovati: {found:>4}  Nuovi: {new:>4}  {status}")
        print(f"  {'─' * 51}")
        print(f"  {'TOTALE':.<20} Trovati: {total_found:>4}  Nuovi: {total_new:>4}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchQuant — Marketplace Scrapers (Vinted + Subito)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Scrape tutti i marketplace")
    parser.add_argument("--vinted", action="store_true",
                        help="Solo Vinted")
    parser.add_argument("--subito", action="store_true",
                        help="Solo Subito.it")
    parser.add_argument("--report", action="store_true",
                        help="Mostra ultimo report")
    args = parser.parse_args()

    orch = MarketplaceOrchestrator()

    if args.all:
        orch.run()
    elif args.vinted:
        orch.run(sources=["vinted"])
    elif args.subito:
        orch.run(sources=["subito"])
    elif args.report:
        conn = get_db_connection()
        for source in ["vinted", "subito"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM listings WHERE source = ?", (source,)
            ).fetchone()[0]
            print(f"  {source}: {count} listing")
        conn.close()
    else:
        print("\n  🛒 WatchQuant — Marketplace Scrapers")
        print("  " + "─" * 45)
        print("  Opzioni:")
        print("    python marketplace_scrapers.py --all      → Tutti i marketplace")
        print("    python marketplace_scrapers.py --vinted   → Solo Vinted")
        print("    python marketplace_scrapers.py --subito   → Solo Subito.it")
        print("    python marketplace_scrapers.py --report   → Report listing salvati")
        print()


if __name__ == "__main__":
    main()

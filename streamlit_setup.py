"""Auto-setup per Streamlit Cloud: crea DB e config se mancanti."""
import os, json, sqlite3

def ensure_cloud_ready():
    # Config
    if not os.path.exists("config.json"):
        if os.path.exists("config_example.json"):
            import shutil
            shutil.copy("config_example.json", "config.json")
        else:
            config = {
                "project_name": "WatchQuant MVP",
                "version": "0.1.0",
                "database": {"path": "watchquant.db"},
                "scraping": {"ebay": {"app_id": "", "cert_id": ""},
                             "chrono24": {"enabled": False},
                             "rate_limits": {"requests_per_minute": 10, "pause_on_error_sec": 60}},
                "alerts": {"telegram_token": "", "telegram_chat_id": "", "email_enabled": False},
                "strategy": {
                    "underval_alert_threshold": 0.15,
                    "underval_strong_buy_threshold": 0.25,
                    "underval_suspicious_threshold": 0.40,
                    "max_position_pct": 0.20, "min_position_eur": 50,
                    "cash_reserve_pct": 0.30, "stop_loss_pct": -0.15,
                    "take_profit_pct": 0.25, "max_single_brand_pct": 0.40,
                    "platform_fee_avg": 0.10, "shipping_cost_eur": 15,
                    "slippage_pct": 0.03
                },
                "currency": {"base": "EUR", "converter_api": "https://api.frankfurter.app"}
            }
            with open("config.json", "w") as f:
                json.dump(config, f, indent=2)

    # Database con dati demo
    if not os.path.exists("watchquant.db"):
        from foundation import create_database, populate_catalog
        create_database()
        populate_catalog()
        # Genera dati demo
        try:
            from scrapers import ScraperOrchestrator
            orch = ScraperOrchestrator()
            orch.run(demo=True)
        except:
            pass
        # Train modello
        try:
            from models import PricingModel
            model = PricingModel()
            model.train(min_listings=2)
        except:
            pass

ensure_cloud_ready()

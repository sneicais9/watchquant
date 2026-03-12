"""
WatchQuant MVP — Pricing Model & Undervaluation Detection
===========================================================
Il cervello del sistema, ora con integrazione dei segnali macro:
1. FeatureBuilder — costruisce le feature dal database + segnali esterni
2. PricingModel — addestra e usa modelli per stimare il fair value
3. UndervalDetector — trova opportunità sottovalutate e assegna segnali
4. ModelTrainer — orchestrazione di training, validazione, e salvataggio

NOVITÀ v0.2: Integrazione con signals.py
  - Il modello usa i segnali macro (oro, forex, crypto, VIX) come feature
  - L'undervaluation detector tiene conto del contesto macro
  - Il signal classifier usa il trend dei rialzi retail come boost

COME USARE:
    # Step 1: assicurati di avere dati nel DB
    python scrapers.py --demo

    # Step 2: raccogli segnali macro (opzionale ma consigliato)
    python signals.py --collect

    # Step 3: addestra il modello e genera segnali
    python models.py --train

    # Step 4: mostra le opportunità trovate
    python models.py --opportunities

    # Step 5: statistiche del modello
    python models.py --stats

FILE RICHIESTI: foundation.py e scrapers.py già eseguiti, DB con listing.
OPZIONALE: signals.py per segnali macro (migliora le predizioni).
"""

import json
import os
import pickle
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import (
        mean_absolute_error, mean_absolute_percentage_error,
        mean_squared_error, r2_score
    )
except ImportError:
    print("Installa dipendenze: pip install -r requirements.txt")
    exit(1)

from foundation import get_db_connection, load_config

# Prova a importare signals (opzionale)
try:
    from signals import SignalsOrchestrator, setup_signals_db
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False

warnings.filterwarnings("ignore", category=UserWarning)

Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/models.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WatchQuant.Models")


# ============================================================
# 1. FEATURE BUILDER — Estrae e costruisce feature dal DB
# ============================================================

class FeatureBuilder:
    """
    Costruisce il dataset di feature per il modello ML.
    
    Legge dal database e produce un DataFrame con:
    - Feature categoriche: brand, model_family, case_material, source, condition
    - Feature numeriche: condition_score, completeness_score, case_size_mm, 
      age_years, num_active_listings, avg_price_ref
    - Feature macro (da signals.py): oro, forex, crypto, VIX
    - Target: price (EUR)
    """

    # Feature numeriche base (sempre presenti)
    NUMERIC_FEATURES_BASE = [
        "condition_score",
        "completeness_score",
        "case_size_mm",
        "age_years",
        "num_active_listings",
        "avg_price_ref",
        "median_price_ref",
        "price_spread_ref",
        "retail_price_eur",
    ]

    # Feature macro da signals.py (aggiunte se disponibili)
    NUMERIC_FEATURES_MACRO = [
        "signal_gold_usd",
        "signal_gold_usd_chg30d",
        "signal_eur_usd",
        "signal_chf_eur",
        "signal_chf_eur_chg30d",
        "signal_sp500_chg30d",
        "signal_vix",
        "signal_btc_usd_chg30d",
        "retail_increases_90d",
        "retail_avg_increase_90d",
    ]

    # Feature categoriche a bassa cardinalità (one-hot encoding)
    CATEGORICAL_LOW = [
        "brand",
        "case_material",
        "movement_type",
        "condition",
        "source",
    ]

    # Feature categoriche ad alta cardinalità (ordinal encoding)
    CATEGORICAL_HIGH = [
        "model_family",
    ]

    def __init__(self, db_path="watchquant.db"):
        self.db_path = db_path
        self.macro_features = {}
        self.has_macro = False

    def _load_macro_features(self):
        """
        Carica i segnali macro da signals.py.
        Questi sono gli stessi per TUTTI i listing (contesto di mercato globale),
        quindi ogni riga del dataset riceve gli stessi valori macro.
        
        Questo è corretto perché il modello impara cose come:
        "quando il VIX è alto E l'orologio è in buone condizioni,
         il fair value è più basso del solito perché la domanda cala"
        """
        if not HAS_SIGNALS:
            logger.info("signals.py non disponibile. Training senza feature macro.")
            return {}

        try:
            setup_signals_db(self.db_path)
            orch = SignalsOrchestrator(self.db_path)
            features = orch.get_signal_features()

            if features:
                self.has_macro = True
                logger.info(f"  ✅ {len(features)} feature macro caricate dai segnali esterni")
                # Log delle feature più importanti
                for key in ["signal_gold_usd", "signal_vix", "signal_btc_usd_chg30d"]:
                    if key in features:
                        logger.info(f"     {key}: {features[key]:.2f}")
            else:
                logger.info("  Nessun segnale macro nel DB. Esegui: python signals.py --collect")

            return features

        except Exception as e:
            logger.warning(f"Errore caricamento segnali macro: {e}")
            return {}

    def get_numeric_features(self):
        """Ritorna la lista di feature numeriche da usare (base + macro se disponibili)."""
        features = self.NUMERIC_FEATURES_BASE.copy()
        if self.has_macro:
            features.extend(self.NUMERIC_FEATURES_MACRO)
        return features

    def build_dataset(self, min_listings_per_ref=3):
        """
        Costruisce il dataset completo per il training.
        
        Returns:
            X (DataFrame): feature
            y (Series): target (prezzo EUR)
            metadata (DataFrame): info aggiuntive (url, title, etc.)
        """
        conn = get_db_connection(self.db_path)

        # Query principale: join listings + watches
        query = """
            SELECT 
                l.id as listing_id,
                l.price,
                l.source,
                l.condition,
                l.condition_score,
                l.completeness_score,
                l.has_box,
                l.has_papers,
                l.seller_location,
                l.url,
                l.title,
                l.scraped_at,
                l.status,
                w.brand,
                w.model,
                w.reference,
                w.model_family,
                w.case_material,
                w.case_size_mm,
                w.movement_type,
                w.year_production,
                w.retail_price_eur,
                w.id as watch_id
            FROM listings l
            JOIN watches w ON l.watch_id = w.id
            WHERE l.price > 0 AND l.price < 50000
            ORDER BY l.scraped_at
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("Nessun listing nel database. Esegui prima gli scraper.")
            return None, None, None

        logger.info(f"Listing caricati dal DB: {len(df)}")

        # --- Feature base ---

        current_year = datetime.now().year
        df["age_years"] = df["year_production"].apply(
            lambda y: current_year - y if pd.notna(y) and y > 1900 else np.nan
        )

        # Statistiche per referenza
        ref_stats = df.groupby("reference").agg(
            avg_price_ref=("price", "mean"),
            median_price_ref=("price", "median"),
            min_price_ref=("price", "min"),
            max_price_ref=("price", "max"),
            num_active_listings=("price", "count"),
        ).reset_index()

        ref_stats["price_spread_ref"] = (
            (ref_stats["max_price_ref"] - ref_stats["min_price_ref"])
            / ref_stats["median_price_ref"].clip(lower=1)
        )

        df = df.merge(ref_stats, on="reference", how="left")

        # Filtra referenze con troppi pochi dati
        df = df[df["num_active_listings"] >= min_listings_per_ref].copy()

        if df.empty:
            logger.warning(
                f"Nessuna referenza con almeno {min_listings_per_ref} listing. "
                "Raccogli più dati o abbassa la soglia."
            )
            return None, None, None

        # --- Feature macro (da signals.py) ---
        self.macro_features = self._load_macro_features()

        if self.macro_features:
            # Aggiungi le feature macro a ogni riga (sono globali)
            for feat_name in self.NUMERIC_FEATURES_MACRO:
                df[feat_name] = self.macro_features.get(feat_name, 0)
            logger.info(f"  Feature macro aggiunte al dataset: {len(self.NUMERIC_FEATURES_MACRO)}")

        # --- Fill NaN ---

        df["case_size_mm"] = df["case_size_mm"].fillna(df["case_size_mm"].median())
        df["retail_price_eur"] = df["retail_price_eur"].fillna(0)
        df["age_years"] = df["age_years"].fillna(0)
        df["condition_score"] = df["condition_score"].fillna(2)
        df["completeness_score"] = df["completeness_score"].fillna(0)

        for col in self.CATEGORICAL_LOW + self.CATEGORICAL_HIGH:
            df[col] = df[col].fillna("unknown")

        # Fill NaN per feature macro
        for col in self.NUMERIC_FEATURES_MACRO:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # --- Separa feature, target, metadata ---
        numeric_features = self.get_numeric_features()
        all_features = numeric_features + self.CATEGORICAL_LOW + self.CATEGORICAL_HIGH
        
        # Assicurati che tutte le colonne esistano
        for col in all_features:
            if col not in df.columns:
                df[col] = 0

        X = df[all_features].copy()
        y = df["price"].copy()
        metadata = df[["listing_id", "watch_id", "reference", "brand", "model",
                        "title", "url", "source", "price", "scraped_at",
                        "has_box", "has_papers", "completeness_score"]].copy()

        logger.info(
            f"Dataset pronto — {len(X)} campioni, {len(X.columns)} feature "
            f"({len(numeric_features)} numeriche + "
            f"{len(self.CATEGORICAL_LOW) + len(self.CATEGORICAL_HIGH)} categoriche), "
            f"{df['reference'].nunique()} referenze uniche"
        )
        if self.has_macro:
            logger.info(f"  🌍 Include {len(self.NUMERIC_FEATURES_MACRO)} feature macro dai segnali esterni")

        return X, y, metadata


# ============================================================
# 2. PRICING MODEL — Stima del fair value
# ============================================================

class PricingModel:
    """
    Modello di pricing per orologi.
    
    MVP: usa Ridge/GradientBoosting con feature base + macro.
    Il preprocessor gestisce automaticamente feature numeriche e categoriche.
    Le feature macro (oro, forex, VIX, crypto) vengono trattate come numeriche.
    """

    MODEL_DIR = Path("models")
    MODEL_FILE = "pricing_model.pkl"
    METADATA_FILE = "model_metadata.json"

    def __init__(self):
        self.model = None
        self.feature_builder = FeatureBuilder()
        self.metadata = {}
        self.is_fitted = False

    def _build_preprocessor(self, numeric_features):
        """Costruisce il preprocessor scikit-learn con le feature corrette."""
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),

            ("cat_low", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), FeatureBuilder.CATEGORICAL_LOW),

            ("cat_high", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                            unknown_value=-1)),
            ]), FeatureBuilder.CATEGORICAL_HIGH),
        ])

    def _build_models(self):
        """Costruisce i modelli candidati."""
        return {
            "ridge": Ridge(alpha=10.0),
            "lasso": Lasso(alpha=1.0, max_iter=5000),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
        }

    def train(self, min_listings=3):
        """
        Addestra il modello completo.
        
        1. Costruisce il dataset (con feature macro se disponibili)
        2. Preprocessing
        3. Prova diversi modelli con cross-validation
        4. Seleziona il migliore
        5. Salva su disco
        """
        logger.info("=" * 50)
        logger.info("INIZIO TRAINING")
        logger.info("=" * 50)

        # 1. Costruisci dataset
        X, y, metadata = self.feature_builder.build_dataset(
            min_listings_per_ref=min_listings
        )
        if X is None:
            return False

        # 2. Preprocessor (con le feature numeriche corrette)
        numeric_features = self.feature_builder.get_numeric_features()
        preprocessor = self._build_preprocessor(numeric_features)

        # 3. Cross-validation per ogni modello candidato
        models = self._build_models()
        results = {}

        logger.info(f"\nValutazione {len(models)} modelli con 5-fold CV...\n")

        kfold = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                scores_mae = cross_val_score(
                    pipeline, X, y,
                    cv=kfold,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1
                )
                scores_r2 = cross_val_score(
                    pipeline, X, y,
                    cv=kfold,
                    scoring="r2",
                    n_jobs=-1
                )

                mae = -scores_mae.mean()
                mae_std = scores_mae.std()
                r2 = scores_r2.mean()
                mape_approx = mae / y.mean() * 100

                results[name] = {
                    "mae": mae,
                    "mae_std": mae_std,
                    "r2": r2,
                    "mape_approx": mape_approx
                }

                logger.info(
                    f"  {name:<22} MAE: €{mae:>8.2f} (±{mae_std:.2f})  "
                    f"R²: {r2:.3f}  MAPE~: {mape_approx:.1f}%"
                )

            except Exception as e:
                logger.error(f"  {name:<22} ERRORE: {e}")
                results[name] = {"mae": float("inf"), "r2": 0, "error": str(e)}

        # 4. Seleziona il modello migliore
        best_name = min(results, key=lambda k: results[k].get("mae", float("inf")))
        best_stats = results[best_name]

        logger.info(f"\n{'='*50}")
        logger.info(f"  MIGLIOR MODELLO: {best_name}")
        logger.info(f"  MAE: €{best_stats['mae']:.2f}  R²: {best_stats['r2']:.3f}")
        logger.info(f"{'='*50}")

        # 5. Retraining finale su tutti i dati
        preprocessor = self._build_preprocessor(numeric_features)
        best_model = models[best_name]

        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])
        final_pipeline.fit(X, y)

        self.model = final_pipeline
        self.is_fitted = True

        # Predizioni sul training set per statistiche
        y_pred = self.model.predict(X)
        final_mae = mean_absolute_error(y, y_pred)
        final_mape = mean_absolute_percentage_error(y, y_pred) * 100
        final_r2 = r2_score(y, y_pred)

        # 6. Metadata del modello
        self.metadata = {
            "model_name": best_name,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_features_numeric": len(numeric_features),
            "n_features_macro": len(self.feature_builder.NUMERIC_FEATURES_MACRO) if self.feature_builder.has_macro else 0,
            "has_macro_signals": self.feature_builder.has_macro,
            "n_references": X["model_family"].nunique(),
            "numeric_features_used": numeric_features,
            "cv_results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                               for kk, vv in v.items()}
                           for k, v in results.items()},
            "final_metrics": {
                "mae": round(final_mae, 2),
                "mape": round(final_mape, 2),
                "r2": round(final_r2, 4),
            },
            "price_stats": {
                "mean": round(y.mean(), 2),
                "median": round(y.median(), 2),
                "min": round(y.min(), 2),
                "max": round(y.max(), 2),
            },
            "macro_snapshot": self.feature_builder.macro_features if self.feature_builder.has_macro else {}
        }

        # 7. Salva tutto
        self._save()

        logger.info(f"\n  Metriche finali (full dataset):")
        logger.info(f"    MAE:  €{final_mae:.2f}")
        logger.info(f"    MAPE: {final_mape:.1f}%")
        logger.info(f"    R²:   {final_r2:.4f}")
        if self.feature_builder.has_macro:
            logger.info(f"    🌍 Segnali macro: ATTIVI ({self.metadata['n_features_macro']} feature)")
        else:
            logger.info(f"    ⚠️  Segnali macro: NON ATTIVI (esegui signals.py --collect per attivarli)")

        return True

    def predict(self, X):
        """Predice il fair value per un DataFrame di feature."""
        if not self.is_fitted:
            self.load()
        return self.model.predict(X)

    def predict_with_confidence(self, X):
        """
        Predice fair value con intervallo di confidenza approssimato.
        
        La confidenza tiene conto di:
        - MAE del modello (base)
        - Se i segnali macro sono disponibili (boost)
        - Numero di listing per la referenza (più dati = più confidenza)
        """
        if not self.is_fitted:
            self.load()

        predictions = self.model.predict(X)
        mae = self.metadata.get("final_metrics", {}).get("mae", 50)

        # Confidenza base
        confidence = np.clip(1.0 - (mae / (predictions + 1e-6)), 0.3, 0.95)

        # Boost se abbiamo segnali macro (il modello ha più informazione)
        if self.metadata.get("has_macro_signals"):
            confidence = np.clip(confidence * 1.08, 0.3, 0.95)

        return predictions, confidence

    def _save(self):
        """Salva modello e metadata su disco."""
        self.MODEL_DIR.mkdir(exist_ok=True)

        with open(self.MODEL_DIR / self.MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)

        with open(self.MODEL_DIR / self.METADATA_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"  Modello salvato in {self.MODEL_DIR}/")

    def load(self):
        """Carica il modello da disco."""
        model_path = self.MODEL_DIR / self.MODEL_FILE
        meta_path = self.MODEL_DIR / self.METADATA_FILE

        if not model_path.exists():
            logger.error("Nessun modello salvato. Esegui prima --train")
            return False

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        self.is_fitted = True
        macro_str = "🌍 con segnali macro" if self.metadata.get("has_macro_signals") else "base"
        logger.info(
            f"  Modello caricato: {self.metadata.get('model_name', '?')} ({macro_str}) "
            f"— trainato: {self.metadata.get('trained_at', '?')[:16]}"
        )
        return True

    def get_feature_importance(self, top_n=15):
        """
        Restituisce le feature più importanti.
        Funziona con GradientBoosting e RandomForest.
        Ora include anche le feature macro.
        """
        if not self.is_fitted:
            self.load()

        inner_model = self.model.named_steps["model"]
        preprocessor = self.model.named_steps["preprocessor"]

        if not hasattr(inner_model, "feature_importances_"):
            logger.info("Il modello attuale non supporta feature importance diretta.")
            return None

        # Ricostruisci i nomi delle feature dopo il preprocessing
        feature_names = []

        # Numeriche (base + macro)
        numeric_used = self.metadata.get("numeric_features_used",
                                          FeatureBuilder.NUMERIC_FEATURES_BASE)
        feature_names.extend(numeric_used)

        # Categoriche one-hot
        try:
            ohe = preprocessor.named_transformers_["cat_low"].named_steps["encoder"]
            feature_names.extend(ohe.get_feature_names_out(FeatureBuilder.CATEGORICAL_LOW))
        except Exception:
            feature_names.extend([f"cat_{i}" for i in range(20)])

        # Categoriche ordinali
        feature_names.extend(FeatureBuilder.CATEGORICAL_HIGH)

        importances = inner_model.feature_importances_

        n = min(len(feature_names), len(importances))
        pairs = list(zip(feature_names[:n], importances[:n]))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[:top_n]


# ============================================================
# 3. UNDERVALUATION DETECTOR — Trova le opportunità
# ============================================================

class UndervalDetector:
    """
    Analizza tutti i listing attivi e assegna:
    - fair_value: stima del modello
    - underval_score: quanto è sottovalutato
    - confidence: quanto è affidabile la stima
    - signal: STRONG_BUY, BUY, WATCH, HOLD, SUSPICIOUS
    
    NOVITÀ: Il detector ora tiene conto del contesto macro.
    Se ci sono rialzi retail recenti, abbassa la soglia di BUY
    (il mercato sta per salire → conviene comprare prima).
    Se il VIX è alto, alza le soglie (più prudenza).
    """

    def __init__(self, config_path="config.json", db_path="watchquant.db"):
        self.config = load_config(config_path)
        self.db_path = db_path
        self.model = PricingModel()
        self.feature_builder = FeatureBuilder(db_path)

        # Soglie base dalla configurazione
        strategy = self.config.get("strategy", {})
        self.threshold_alert = strategy.get("underval_alert_threshold", 0.15)
        self.threshold_strong = strategy.get("underval_strong_buy_threshold", 0.25)
        self.threshold_suspicious = strategy.get("underval_suspicious_threshold", 0.40)

        # Carica contesto macro per aggiustare le soglie
        self.macro_adjustment = self._calc_macro_adjustment()

    def _calc_macro_adjustment(self):
        """
        Calcola un aggiustamento delle soglie basato sul contesto macro.
        
        Ritorna un moltiplicatore:
        - < 1.0 → mercato favorevole, abbassa le soglie (compra di più)
        - = 1.0 → neutro
        - > 1.0 → mercato sfavorevole, alza le soglie (più prudenza)
        
        Fattori:
        - Rialzi retail recenti → abbassa soglie (il secondario salirà)
        - VIX alto → alza soglie (incertezza)
        - Oro in salita → abbassa soglie per orologi in metallo prezioso
        - BTC in calo → alza soglie (meno domanda speculativa)
        """
        if not HAS_SIGNALS:
            return 1.0

        try:
            setup_signals_db(self.db_path)
            orch = SignalsOrchestrator(self.db_path)
            features = orch.get_signal_features()

            if not features:
                return 1.0

            adjustment = 1.0

            # Rialzi retail recenti → il secondario salirà
            retail_count = features.get("retail_increases_90d", 0)
            retail_avg = features.get("retail_avg_increase_90d", 0)
            if retail_count >= 3 and retail_avg > 5:
                adjustment -= 0.08  # Abbassa soglie (più aggressivo)
                logger.info(f"  📊 Macro: {retail_count} rialzi retail recenti → soglie abbassate")

            # VIX alto → più prudenza
            vix = features.get("signal_vix", 0)
            if vix > 25:
                adjustment += 0.10  # Alza soglie
                logger.info(f"  📊 Macro: VIX alto ({vix:.0f}) → soglie alzate (prudenza)")
            elif vix > 0 and vix < 15:
                adjustment -= 0.03  # Mercato calmo → leggermente più aggressivo
                logger.info(f"  📊 Macro: VIX basso ({vix:.0f}) → soglie leggermente abbassate")

            # BTC in forte calo → meno domanda speculativa
            btc_chg = features.get("signal_btc_usd_chg30d", 0)
            if btc_chg < -15:
                adjustment += 0.05
                logger.info(f"  📊 Macro: BTC in calo ({btc_chg:+.1f}%) → più prudenza")
            elif btc_chg > 20:
                adjustment -= 0.03
                logger.info(f"  📊 Macro: BTC in rialzo ({btc_chg:+.1f}%) → domanda speculativa attesa")

            # Oro in forte rialzo → opportunità su orologi in metallo
            gold_chg = features.get("signal_gold_usd_chg30d", 0)
            if gold_chg > 5:
                adjustment -= 0.03
                logger.info(f"  📊 Macro: Oro in rialzo ({gold_chg:+.1f}%) → opportunità su orologi in metallo")

            adjustment = max(0.80, min(1.20, adjustment))
            logger.info(f"  📊 Aggiustamento macro finale: {adjustment:.2f}x")
            return adjustment

        except Exception as e:
            logger.debug(f"Errore calcolo aggiustamento macro: {e}")
            return 1.0

    def score_all_listings(self):
        """
        Valuta tutti i listing attivi nel database.
        Aggiorna fair_value, underval_score, confidence, signal nel DB.
        """
        logger.info("Inizio scoring di tutti i listing...")

        X, y, metadata = self.feature_builder.build_dataset(min_listings_per_ref=2)
        if X is None:
            logger.warning("Nessun dato disponibile per lo scoring.")
            return None

        if not self.model.is_fitted:
            if not self.model.load():
                logger.error("Modello non disponibile. Esegui prima --train")
                return None

        # Predizioni con confidenza
        fair_values, confidences = self.model.predict_with_confidence(X)

        # Calcola undervaluation score
        actual_prices = y.values
        underval_scores = (fair_values - actual_prices) / np.clip(fair_values, 1, None)

        # Assegna segnali (con aggiustamento macro)
        signals = []
        for uv, conf in zip(underval_scores, confidences):
            signals.append(self._classify_signal(uv, conf))

        # Assembla risultato
        results = metadata.copy()
        results["fair_value"] = np.round(fair_values, 2)
        results["underval_score"] = np.round(underval_scores, 4)
        results["confidence"] = np.round(confidences, 3)
        results["signal"] = signals
        results["scored_at"] = datetime.now().isoformat()
        results["discount_eur"] = np.round(fair_values - actual_prices, 2)
        results["discount_pct"] = np.round(underval_scores * 100, 1)

        # Aggiorna il database
        self._update_db(results)

        # Ordina per opportunità
        results["opportunity_rank"] = (
            results["underval_score"] * results["confidence"]
        )
        results = results.sort_values("opportunity_rank", ascending=False)

        # Report
        n_strong = (results["signal"] == "STRONG_BUY").sum()
        n_buy = (results["signal"] == "BUY").sum()
        n_watch = (results["signal"] == "WATCH").sum()
        n_suspicious = (results["signal"] == "SUSPICIOUS").sum()

        logger.info(f"\n  Scoring completato — {len(results)} listing valutati")
        logger.info(f"  🔥 STRONG BUY: {n_strong}")
        logger.info(f"  ⭐ BUY:        {n_buy}")
        logger.info(f"  👀 WATCH:      {n_watch}")
        logger.info(f"  ⚠️  SUSPICIOUS: {n_suspicious}")

        if self.macro_adjustment != 1.0:
            direction = "abbassate" if self.macro_adjustment < 1.0 else "alzate"
            logger.info(f"  🌍 Soglie {direction} del {abs(1 - self.macro_adjustment)*100:.0f}% per contesto macro")

        return results

    def _classify_signal(self, underval_score, confidence):
        """Classifica il segnale con aggiustamento macro."""
        # Applica l'aggiustamento macro alle soglie
        adj = self.macro_adjustment
        t_suspicious = self.threshold_suspicious * adj
        t_strong = self.threshold_strong * adj
        t_alert = self.threshold_alert * adj

        if underval_score > t_suspicious:
            return "SUSPICIOUS"
        elif underval_score > t_strong and confidence > 0.6:
            return "STRONG_BUY"
        elif underval_score > t_alert and confidence > 0.5:
            return "BUY"
        elif underval_score > 0.05:
            return "WATCH"
        elif underval_score > -0.05:
            return "FAIR_PRICE"
        else:
            return "OVERPRICED"

    def _update_db(self, results):
        """Aggiorna i listing nel database con i valori calcolati."""
        conn = get_db_connection(self.db_path)
        updated = 0

        for _, row in results.iterrows():
            try:
                conn.execute("""
                    UPDATE listings SET
                        fair_value = ?,
                        underval_score = ?,
                        confidence = ?,
                        signal = ?,
                        scored_at = ?
                    WHERE id = ?
                """, (
                    row["fair_value"],
                    row["underval_score"],
                    row["confidence"],
                    row["signal"],
                    row["scored_at"],
                    row["listing_id"]
                ))
                updated += 1
            except Exception as e:
                logger.debug(f"Errore update listing {row['listing_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"  Database aggiornato: {updated} listing scored")

    def get_opportunities(self, min_signal="BUY", limit=30):
        """Restituisce le migliori opportunità dal database."""
        conn = get_db_connection(self.db_path)

        if min_signal == "STRONG_BUY":
            signal_filter = "('STRONG_BUY')"
        else:
            signal_filter = "('STRONG_BUY', 'BUY')"

        query = f"""
            SELECT 
                l.id,
                w.brand,
                w.model,
                w.reference,
                l.price,
                l.fair_value,
                l.underval_score,
                l.confidence,
                l.signal,
                l.condition,
                l.has_box,
                l.has_papers,
                l.completeness_score,
                l.source,
                l.url,
                l.title,
                l.scored_at
            FROM listings l
            JOIN watches w ON l.watch_id = w.id
            WHERE l.signal IN {signal_filter}
              AND l.fair_value IS NOT NULL
            ORDER BY (l.underval_score * l.confidence) DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if df.empty:
            logger.info("Nessuna opportunità trovata con i criteri attuali.")
        return df

    def print_opportunities(self, limit=20):
        """Stampa le opportunità in formato leggibile."""
        df = self.get_opportunities(limit=limit)

        if df.empty:
            print("\n  Nessuna opportunità trovata.")
            print("  Esegui prima: python models.py --train")
            return

        print("\n" + "=" * 80)
        print("  🎯 OPPORTUNITÀ DI ACQUISTO — WatchQuant")
        if self.macro_adjustment != 1.0:
            direction = "📈 FAVOREVOLE" if self.macro_adjustment < 1.0 else "📉 CAUTO"
            print(f"  🌍 Contesto macro: {direction} (soglie x{self.macro_adjustment:.2f})")
        print("=" * 80)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            signal_emoji = {
                "STRONG_BUY": "🔥", "BUY": "⭐", "WATCH": "👀"
            }.get(row["signal"], "")

            discount = (row["fair_value"] - row["price"])
            discount_pct = row["underval_score"] * 100

            comp = int(row["completeness_score"])
            comp_str = "📦" * min(comp, 3) + "  " * (3 - min(comp, 3))

            print(f"\n  {signal_emoji} #{i} — {row['signal']}")
            print(f"  {row['brand']} {row['model']} ({row['reference']})")
            print(f"  💰 Prezzo: €{row['price']:,.0f}  →  Fair Value: €{row['fair_value']:,.0f}")
            print(f"  📉 Sconto: €{discount:,.0f} ({discount_pct:+.1f}%)")
            print(f"  📊 Confidenza: {row['confidence']:.0%}  |  Condizione: {row['condition']}")
            print(f"  {comp_str} Completezza: {comp}/3  |  Fonte: {row['source']}")
            if row.get("url") and not str(row["url"]).startswith("https://example.com"):
                print(f"  🔗 {row['url']}")
            print(f"  {'─' * 60}")

        print(f"\n  Totale opportunità: {len(df)}")
        print()


# ============================================================
# 4. MODEL TRAINER — Orchestrazione completa
# ============================================================

class ModelTrainer:
    """
    Orchestratore per training + scoring in un unico flusso.
    Ora include la raccolta segnali macro prima del training.
    """

    def __init__(self, config_path="config.json", db_path="watchquant.db"):
        self.config_path = config_path
        self.db_path = db_path

    def full_pipeline(self, collect_signals=True):
        """
        Esegue la pipeline completa:
        1. (Opzionale) Raccolta segnali macro
        2. Training modello
        3. Feature importance
        4. Scoring di tutti i listing
        5. Stampa opportunità
        """
        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETA: SIGNALS → TRAINING → SCORING")
        logger.info("=" * 60)

        # Step 0: Raccolta segnali macro (se disponibile)
        if collect_signals and HAS_SIGNALS:
            logger.info("\n  [0/4] Raccolta segnali macro...")
            try:
                orch = SignalsOrchestrator(self.db_path)
                orch.collect_all()
            except Exception as e:
                logger.warning(f"Errore raccolta segnali (non bloccante): {e}")
        elif not HAS_SIGNALS:
            logger.info("\n  [0/4] signals.py non trovato — training senza feature macro")

        # Step 1: Training
        logger.info("\n  [1/4] Training modello...")
        model = PricingModel()
        success = model.train(min_listings=2)

        if not success:
            logger.error("Training fallito. Verifica i dati.")
            return False

        # Step 2: Feature importance
        logger.info("\n  [2/4] Analisi feature importance...")
        fi = model.get_feature_importance(top_n=15)
        if fi:
            logger.info("\n  📊 TOP 15 FEATURE IMPORTANCE:")
            for name, importance in fi:
                bar = "█" * int(importance * 100)
                # Evidenzia feature macro
                macro_tag = " 🌍" if name.startswith("signal_") or name.startswith("retail_") else ""
                logger.info(f"    {name:<35} {importance:.4f} {bar}{macro_tag}")

        # Step 3: Scoring
        logger.info("\n  [3/4] Scoring listing...")
        detector = UndervalDetector(self.config_path, self.db_path)
        results = detector.score_all_listings()

        # Step 4: Mostra opportunità
        if results is not None:
            logger.info("\n  [4/4] Opportunità trovate:")
            detector.print_opportunities()

        return True

    def show_model_stats(self):
        """Mostra statistiche del modello corrente."""
        model = PricingModel()
        if not model.load():
            print("\n  Nessun modello trovato. Esegui: python models.py --train")
            return

        meta = model.metadata
        print("\n" + "=" * 50)
        print("  📊 STATISTICHE MODELLO")
        print("=" * 50)
        print(f"\n  Modello:        {meta.get('model_name', '?')}")
        print(f"  Addestrato:     {meta.get('trained_at', '?')[:16]}")
        print(f"  Campioni:       {meta.get('n_samples', '?')}")
        print(f"  Feature totali: {meta.get('n_features', '?')}")
        print(f"  Feature macro:  {meta.get('n_features_macro', 0)}")
        print(f"  Segnali macro:  {'🌍 ATTIVI' if meta.get('has_macro_signals') else '⚠️ NON ATTIVI'}")
        print(f"  Referenze:      {meta.get('n_references', '?')}")

        fm = meta.get("final_metrics", {})
        print(f"\n  --- Metriche ---")
        print(f"  MAE:   €{fm.get('mae', '?')}")
        print(f"  MAPE:  {fm.get('mape', '?')}%")
        print(f"  R²:    {fm.get('r2', '?')}")

        ps = meta.get("price_stats", {})
        print(f"\n  --- Distribuzione prezzi ---")
        print(f"  Media:   €{ps.get('mean', '?')}")
        print(f"  Mediana: €{ps.get('median', '?')}")
        print(f"  Min:     €{ps.get('min', '?')}")
        print(f"  Max:     €{ps.get('max', '?')}")

        # Snapshot macro al momento del training
        macro = meta.get("macro_snapshot", {})
        if macro:
            print(f"\n  --- Snapshot Macro (al momento del training) ---")
            key_signals = [
                ("signal_gold_usd", "Oro (USD/oz)"),
                ("signal_vix", "VIX"),
                ("signal_btc_usd_chg30d", "BTC var. 30d (%)"),
                ("signal_chf_eur", "CHF/EUR"),
                ("retail_increases_90d", "Rialzi retail 90d"),
            ]
            for key, label in key_signals:
                val = macro.get(key)
                if val is not None:
                    print(f"  {label:<25} {val:>12.2f}")

        # CV results
        cv = meta.get("cv_results", {})
        if cv:
            print(f"\n  --- Cross-Validation ---")
            for name, stats in cv.items():
                if "error" in stats:
                    print(f"  {name:<22} ❌ {stats['error']}")
                else:
                    print(
                        f"  {name:<22} MAE: €{stats.get('mae', '?'):>8}  "
                        f"R²: {stats.get('r2', '?')}  "
                        f"MAPE~: {stats.get('mape_approx', '?')}%"
                    )
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchQuant — Pricing Model & Undervaluation Detection"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Pipeline completa: segnali → training → scoring"
    )
    parser.add_argument(
        "--train-no-signals", action="store_true",
        help="Training senza raccolta segnali (usa quelli già nel DB)"
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Solo scoring dei listing (usa modello esistente)"
    )
    parser.add_argument(
        "--opportunities", action="store_true",
        help="Mostra le opportunità trovate"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Mostra statistiche del modello corrente"
    )
    args = parser.parse_args()

    trainer = ModelTrainer()

    if args.train:
        trainer.full_pipeline(collect_signals=True)
    elif args.train_no_signals:
        trainer.full_pipeline(collect_signals=False)
    elif args.score:
        detector = UndervalDetector()
        detector.score_all_listings()
        detector.print_opportunities()
    elif args.opportunities:
        detector = UndervalDetector()
        detector.print_opportunities()
    elif args.stats:
        trainer.show_model_stats()
    else:
        print("\n  🧠 WatchQuant — Pricing Model (con Segnali Macro)")
        print("  " + "─" * 50)
        print("  Opzioni:")
        print("    python models.py --train              → Segnali + Training + Scoring")
        print("    python models.py --train-no-signals   → Training senza raccolta segnali")
        print("    python models.py --score              → Solo scoring (modello esistente)")
        print("    python models.py --opportunities      → Mostra opportunità")
        print("    python models.py --stats              → Statistiche modello")
        print()
        print("  Flusso consigliato:")
        print("    1. python foundation.py")
        print("    2. python scrapers.py --demo")
        print("    3. python signals.py --collect        ← NUOVO: raccoglie segnali macro")
        print("    4. python models.py --train           ← ora usa anche i segnali macro")
        print()
        if HAS_SIGNALS:
            print("  ✅ signals.py trovato — feature macro disponibili")
        else:
            print("  ⚠️  signals.py non trovato — il modello funzionerà senza feature macro")
        print()


if __name__ == "__main__":
    main()

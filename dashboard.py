import streamlit_setup
"""
WatchQuant MVP — Dashboard Streamlit
======================================
Dashboard completa con 5 sezioni:
1. Overview — metriche chiave, salute sistema
2. Opportunità — listing sottovalutati con filtri
3. Mercato — analisi prezzi per brand/referenza
4. Portafoglio — gestione pezzi comprati/venduti, P&L
5. Sistema — log scraping, stato modello, configurazione

COME USARE:
    # Assicurati di avere dati:
    python foundation.py
    python scrapers.py --demo
    python models.py --train

    # Lancia la dashboard:
    streamlit run dashboard.py

    # Si aprirà nel browser su http://localhost:8501
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from foundation import get_db_connection, load_config

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WatchQuant",
    page_icon="🕐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Metriche più compatte */
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem; }
    
    /* Tabelle più leggibili */
    .dataframe { font-size: 0.85rem; }
    
    /* Signal badges */
    .signal-strong { 
        background: #dc2626; color: white; 
        padding: 2px 8px; border-radius: 4px; font-weight: bold; 
    }
    .signal-buy { 
        background: #f59e0b; color: white; 
        padding: 2px 8px; border-radius: 4px; font-weight: bold; 
    }
    .signal-watch { 
        background: #3b82f6; color: white; 
        padding: 2px 8px; border-radius: 4px; 
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_data(ttl=60)  # Cache per 60 secondi
def load_listings():
    """Carica tutti i listing con info orologio."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT 
            l.*,
            w.brand, w.model, w.reference, w.model_family,
            w.case_material, w.case_size_mm, w.movement_type,
            w.retail_price_eur
        FROM listings l
        JOIN watches w ON l.watch_id = w.id
        ORDER BY l.scraped_at DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_portfolio():
    """Carica il portafoglio."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT 
            p.*,
            w.brand, w.model, w.reference, w.model_family
        FROM portfolio p
        JOIN watches w ON p.watch_id = w.id
        ORDER BY p.buy_date DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_watches():
    """Carica catalogo referenze."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM watches ORDER BY brand, model", conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_scrape_logs():
    """Carica log scraping."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT * FROM scrape_log ORDER BY started_at DESC LIMIT 50
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_model_metadata():
    """Carica metadata del modello ML."""
    meta_path = Path("models/model_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def signal_badge(signal):
    """Restituisce emoji + testo per un segnale."""
    badges = {
        "STRONG_BUY": "🔥 STRONG BUY",
        "BUY": "⭐ BUY",
        "WATCH": "👀 WATCH",
        "FAIR_PRICE": "➡️ FAIR",
        "OVERPRICED": "📈 OVERPRICED",
        "SUSPICIOUS": "⚠️ SUSPICIOUS",
    }
    return badges.get(signal, signal or "—")


def format_eur(val):
    """Formatta un numero come EUR."""
    if pd.isna(val) or val is None:
        return "—"
    return f"€{val:,.0f}"


def completeness_icons(score):
    """Icone per completezza (box/papers/warranty)."""
    if pd.isna(score):
        return "—"
    score = int(score)
    return "📦" * score + "⬜" * (3 - score)


# ============================================================
# SIDEBAR — Navigazione
# ============================================================

st.sidebar.title("🕐 WatchQuant")
st.sidebar.caption("Quantitative Trading System per Orologi")

# Immagine personalizzata nella sidebar
import os
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.jpg")
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

page = st.sidebar.radio(
    "Navigazione",
    ["📊 Overview", "🎯 Opportunità", "📈 Mercato", "💼 Portafoglio", "📝 Diario di Bordo", "⚙️ Sistema"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Aggiorna dati"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "📊 Overview":
    st.title("📊 Overview")

    listings = load_listings()
    portfolio = load_portfolio()
    model_meta = load_model_metadata()

    # --- Metriche top ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Listing Totali", f"{len(listings):,}")
    with col2:
        scored = listings[listings["signal"].notna()]
        n_opportunities = scored[scored["signal"].isin(["STRONG_BUY", "BUY"])].shape[0]
        st.metric("Opportunità Attive", n_opportunities)
    with col3:
        n_holding = portfolio[portfolio["status"] == "holding"].shape[0] if not portfolio.empty else 0
        st.metric("Pezzi in Portafoglio", n_holding)
    with col4:
        if not portfolio.empty and "net_profit" in portfolio.columns:
            total_pnl = portfolio["net_profit"].sum()
            st.metric("P&L Totale", format_eur(total_pnl))
        else:
            st.metric("P&L Totale", "€0")
    with col5:
        if model_meta:
            mae = model_meta.get("final_metrics", {}).get("mae", "?")
            st.metric("Precisione Modello (MAE)", f"€{mae}")
        else:
            st.metric("Precisione Modello", "Non addestrato")

    st.markdown("---")

    # --- Distribuzione segnali ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribuzione Segnali")
        if not listings.empty and listings["signal"].notna().any():
            signal_counts = listings["signal"].value_counts()
            colors = {
                "STRONG_BUY": "#dc2626", "BUY": "#f59e0b", "WATCH": "#3b82f6",
                "FAIR_PRICE": "#6b7280", "OVERPRICED": "#9333ea", "SUSPICIOUS": "#ef4444"
            }
            fig = px.bar(
                x=signal_counts.index,
                y=signal_counts.values,
                color=signal_counts.index,
                color_discrete_map=colors,
                labels={"x": "Segnale", "y": "Conteggio"},
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessun listing valutato. Esegui: `python models.py --train`")

    with col_right:
        st.subheader("Listing per Brand")
        if not listings.empty:
            brand_counts = listings["brand"].value_counts().head(10)
            fig = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # --- Prezzi medi per brand ---
    st.subheader("Prezzo Medio per Brand")
    if not listings.empty:
        brand_stats = listings.groupby("brand").agg(
            prezzo_medio=("price", "mean"),
            prezzo_mediano=("price", "median"),
            num_listing=("price", "count"),
            prezzo_min=("price", "min"),
            prezzo_max=("price", "max"),
        ).round(0).sort_values("num_listing", ascending=False)

        fig = px.bar(
            brand_stats.reset_index(),
            x="brand", y="prezzo_medio",
            error_y=brand_stats["prezzo_max"].values - brand_stats["prezzo_medio"].values,
            color="num_listing",
            color_continuous_scale="Blues",
            labels={"brand": "Brand", "prezzo_medio": "Prezzo Medio (€)", "num_listing": "N. Listing"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: OPPORTUNITÀ
# ============================================================

elif page == "🎯 Opportunità":
    st.title("🎯 Opportunità di Acquisto")

    listings = load_listings()
    scored = listings[listings["signal"].notna()].copy()

    if scored.empty:
        st.warning("Nessun listing valutato. Esegui prima: `python models.py --train`")
        st.stop()

    # --- Filtri ---
    st.sidebar.markdown("### Filtri Opportunità")

    signal_filter = st.sidebar.multiselect(
        "Segnale",
        options=["STRONG_BUY", "BUY", "WATCH", "FAIR_PRICE", "OVERPRICED", "SUSPICIOUS"],
        default=["STRONG_BUY", "BUY"]
    )

    brand_filter = st.sidebar.multiselect(
        "Brand",
        options=sorted(scored["brand"].unique()),
        default=[]
    )

    price_range = st.sidebar.slider(
        "Range Prezzo (€)",
        min_value=0,
        max_value=int(scored["price"].max()) + 100,
        value=(0, int(scored["price"].max()) + 100)
    )

    min_confidence = st.sidebar.slider(
        "Confidenza Minima",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    # Applica filtri
    mask = scored["signal"].isin(signal_filter) if signal_filter else pd.Series(True, index=scored.index)
    if brand_filter:
        mask &= scored["brand"].isin(brand_filter)
    mask &= scored["price"].between(*price_range)
    mask &= scored["confidence"].fillna(0) >= min_confidence

    filtered = scored[mask].sort_values("underval_score", ascending=False)

    # --- Metriche filtrate ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Risultati", len(filtered))
    if not filtered.empty:
        col2.metric("Sconto Medio", f"{filtered['underval_score'].mean()*100:.1f}%")
        best = filtered.iloc[0]
        col3.metric("Miglior Sconto", f"{best['underval_score']*100:.1f}% — {best['brand']} {best['reference']}")

    st.markdown("---")

    # --- Cards opportunità ---
    for i, (_, row) in enumerate(filtered.head(30).iterrows()):
        with st.container():
            cols = st.columns([1, 3, 2, 2, 1])

            with cols[0]:
                st.markdown(f"**#{i+1}**")
                st.markdown(signal_badge(row["signal"]))

            with cols[1]:
                st.markdown(f"**{row['brand']} {row['model']}**")
                st.caption(f"Ref: {row['reference']}  |  {row['condition'] or '—'}  |  {completeness_icons(row.get('completeness_score'))}")

            with cols[2]:
                st.metric(
                    "Prezzo → Fair Value",
                    format_eur(row["price"]),
                    f"FV: {format_eur(row.get('fair_value'))}"
                )

            with cols[3]:
                discount = row.get("underval_score", 0)
                discount_eur = row.get("fair_value", 0) - row["price"] if row.get("fair_value") else 0
                st.metric(
                    "Sconto",
                    f"{discount*100:.1f}%",
                    f"€{discount_eur:,.0f} potenziale"
                )

            with cols[4]:
                st.caption(f"📊 {row.get('confidence', 0):.0%}")
                st.caption(f"📍 {row['source']}")
                if row.get("url") and "example.com" not in str(row.get("url", "")):
                    st.markdown(f"[🔗 Link]({row['url']})")

            st.markdown("---")

    # --- Tabella completa esportabile ---
    with st.expander("📋 Tabella completa (esportabile)"):
        export_cols = [
            "brand", "model", "reference", "price", "fair_value",
            "underval_score", "confidence", "signal", "condition",
            "source", "url"
        ]
        available_cols = [c for c in export_cols if c in filtered.columns]
        st.dataframe(
            filtered[available_cols].reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        csv = filtered[available_cols].to_csv(index=False)
        st.download_button(
            "📥 Scarica CSV",
            csv,
            "watchquant_opportunities.csv",
            "text/csv"
        )


# ============================================================
# PAGE 3: MERCATO
# ============================================================

elif page == "📈 Mercato":
    st.title("📈 Analisi di Mercato")

    listings = load_listings()
    watches = load_watches()

    if listings.empty:
        st.warning("Nessun dato disponibile. Esegui prima gli scraper.")
        st.stop()

    # --- Selettore brand/referenza ---
    col1, col2 = st.columns(2)
    with col1:
        selected_brand = st.selectbox(
            "Brand",
            options=["Tutti"] + sorted(listings["brand"].unique().tolist())
        )
    with col2:
        if selected_brand != "Tutti":
            refs = sorted(listings[listings["brand"] == selected_brand]["reference"].unique().tolist())
        else:
            refs = sorted(listings["reference"].unique().tolist())
        selected_ref = st.selectbox(
            "Referenza",
            options=["Tutte"] + refs
        )

    # Filtra
    market_data = listings.copy()
    if selected_brand != "Tutti":
        market_data = market_data[market_data["brand"] == selected_brand]
    if selected_ref != "Tutte":
        market_data = market_data[market_data["reference"] == selected_ref]

    # --- Statistiche di mercato ---
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Listing", len(market_data))
    col2.metric("Prezzo Medio", format_eur(market_data["price"].mean()))
    col3.metric("Prezzo Mediano", format_eur(market_data["price"].median()))
    col4.metric("Min", format_eur(market_data["price"].min()))
    col5.metric("Max", format_eur(market_data["price"].max()))

    # --- Distribuzione prezzi ---
    st.subheader("Distribuzione Prezzi")
    fig = px.histogram(
        market_data, x="price",
        nbins=30,
        color="condition",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"price": "Prezzo (€)", "condition": "Condizione"},
        marginal="box"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Prezzo per condizione ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Prezzo per Condizione")
        fig = px.box(
            market_data, x="condition", y="price",
            color="condition",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"condition": "Condizione", "price": "Prezzo (€)"},
            category_orders={"condition": ["new", "like_new", "good", "fair", "poor"]}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Prezzo per Completezza")
        fig = px.box(
            market_data, x="completeness_score", y="price",
            color="completeness_score",
            labels={"completeness_score": "Completezza (0-3)", "price": "Prezzo (€)"}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Prezzo per fonte ---
    st.subheader("Confronto Prezzi per Piattaforma")
    fig = px.box(
        market_data, x="source", y="price",
        color="source",
        labels={"source": "Piattaforma", "price": "Prezzo (€)"}
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap brand x condizione ---
    st.subheader("Heatmap: Prezzo Medio per Brand × Condizione")
    if listings["brand"].nunique() > 1:
        pivot = listings.pivot_table(
            values="price", index="brand", columns="condition",
            aggfunc="median"
        ).reindex(columns=["new", "like_new", "good", "fair", "poor"])

        fig = px.imshow(
            pivot,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            labels={"color": "Prezzo Mediano (€)"},
            aspect="auto"
        )
        fig.update_layout(height=max(300, len(pivot) * 40))
        st.plotly_chart(fig, use_container_width=True)

    # --- Scatter: prezzo vs fair value ---
    if "fair_value" in market_data.columns and market_data["fair_value"].notna().any():
        st.subheader("Prezzo vs Fair Value (sotto la diagonale = sottovalutato)")
        scatter_data = market_data[market_data["fair_value"].notna()].copy()
        fig = px.scatter(
            scatter_data, x="fair_value", y="price",
            color="signal",
            hover_data=["brand", "model", "reference", "condition"],
            color_discrete_map={
                "STRONG_BUY": "#dc2626", "BUY": "#f59e0b", "WATCH": "#3b82f6",
                "FAIR_PRICE": "#6b7280", "OVERPRICED": "#9333ea", "SUSPICIOUS": "#ef4444"
            },
            labels={"fair_value": "Fair Value (€)", "price": "Prezzo Listing (€)"}
        )
        # Linea diagonale (fair price)
        max_val = max(scatter_data["fair_value"].max(), scatter_data["price"].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(dash="dash", color="gray"),
            showlegend=False, name="Fair Price"
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: PORTAFOGLIO
# ============================================================

elif page == "💼 Portafoglio":
    st.title("💼 Gestione Portafoglio")

    portfolio = load_portfolio()
    watches = load_watches()
    listings = load_listings()

    # --- Aggiungi pezzo al portafoglio ---
    with st.expander("➕ Aggiungi Orologio al Portafoglio", expanded=portfolio.empty):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                watch_options = watches.apply(
                    lambda r: f"{r['brand']} {r['model']} ({r['reference']})", axis=1
                ).tolist()
                selected_watch = st.selectbox("Orologio", options=watch_options)
                buy_price = st.number_input("Prezzo di acquisto (€)", min_value=1.0, value=200.0, step=10.0)
                buy_date = st.date_input("Data acquisto", value=datetime.now())

            with col2:
                buy_source = st.selectbox("Fonte acquisto", ["ebay", "chrono24", "vinted", "subito", "privato", "altro"])
                shipping_cost = st.number_input("Spedizione (€)", min_value=0.0, value=15.0, step=1.0)
                auth_cost = st.number_input("Autenticazione (€)", min_value=0.0, value=0.0, step=10.0)
                notes = st.text_input("Note (opzionale)")

            if st.button("💾 Aggiungi al Portafoglio", type="primary"):
                # Trova watch_id
                if selected_watch:
                    ref = selected_watch.split("(")[-1].replace(")", "").strip()
                    conn = get_db_connection()
                    watch_row = conn.execute(
                        "SELECT id FROM watches WHERE reference = ?", (ref,)
                    ).fetchone()

                    if watch_row:
                        costs = json.dumps({"shipping": shipping_cost, "auth": auth_cost})
                        total_cost = buy_price + shipping_cost + auth_cost

                        conn.execute("""
                            INSERT INTO portfolio 
                                (watch_id, buy_price, buy_date, buy_source, costs, total_cost, status, notes)
                            VALUES (?, ?, ?, ?, ?, ?, 'holding', ?)
                        """, (watch_row["id"], buy_price, buy_date.isoformat(),
                              buy_source, costs, total_cost, notes))
                        conn.commit()
                        st.success(f"Aggiunto: {selected_watch} a €{buy_price:,.0f}")
                        st.cache_data.clear()
                        st.rerun()
                    conn.close()

    st.markdown("---")

    if portfolio.empty:
        st.info("Il portafoglio è vuoto. Aggiungi il tuo primo orologio qui sopra.")
        st.stop()

    # --- Metriche portafoglio ---
    holding = portfolio[portfolio["status"] == "holding"]
    sold = portfolio[portfolio["status"] == "sold"]

    col1, col2, col3, col4 = st.columns(4)

    total_invested = holding["total_cost"].sum() if not holding.empty else 0
    col1.metric("Capitale Investito", format_eur(total_invested))

    # Stima valore corrente (usa fair value medio dal modello se disponibile)
    if not holding.empty and not listings.empty:
        current_values = []
        for _, pos in holding.iterrows():
            ref_listings = listings[
                (listings["watch_id"] == pos["watch_id"]) &
                (listings["fair_value"].notna())
            ]
            if not ref_listings.empty:
                current_values.append(ref_listings["fair_value"].median())
            else:
                ref_listings_price = listings[listings["watch_id"] == pos["watch_id"]]
                if not ref_listings_price.empty:
                    current_values.append(ref_listings_price["price"].median())
                else:
                    current_values.append(pos["buy_price"])

        estimated_nav = sum(current_values)
        unrealized_pnl = estimated_nav - total_invested
        col2.metric("Valore Stimato", format_eur(estimated_nav), f"{unrealized_pnl:+,.0f}€")
    else:
        col2.metric("Valore Stimato", "—")

    if not sold.empty and sold["net_profit"].notna().any():
        realized = sold["net_profit"].sum()
        win_rate = (sold["net_profit"] > 0).mean() * 100
        col3.metric("P&L Realizzato", format_eur(realized))
        col4.metric("Win Rate", f"{win_rate:.0f}%")
    else:
        col3.metric("P&L Realizzato", "€0")
        col4.metric("Win Rate", "—")

    # --- Tabella portafoglio ---
    st.subheader("Posizioni Aperte")
    if not holding.empty:
        display_cols = ["brand", "model", "reference", "buy_price", "total_cost",
                        "buy_date", "buy_source", "notes"]
        available = [c for c in display_cols if c in holding.columns]
        st.dataframe(holding[available].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Nessuna posizione aperta.")

    # --- Treemap composizione ---
    if not holding.empty and len(holding) > 1:
        st.subheader("Composizione Portafoglio")
        fig = px.treemap(
            holding,
            path=["brand", "model"],
            values="total_cost",
            color="total_cost",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Registra vendita ---
    if not holding.empty:
        st.markdown("---")
        with st.expander("💰 Registra Vendita"):
            hold_options = holding.apply(
                lambda r: f"#{r['id']} — {r['brand']} {r['model']} (comprato €{r['buy_price']:,.0f})",
                axis=1
            ).tolist()
            sell_selection = st.selectbox("Pezzo venduto", options=hold_options)
            sell_price = st.number_input("Prezzo di vendita (€)", min_value=1.0, step=10.0)
            sell_source = st.selectbox("Venduto su", ["ebay", "chrono24", "vinted", "subito", "privato", "altro"])
            sell_commission = st.number_input("Commissione piattaforma (€)", min_value=0.0, step=5.0)

            if st.button("✅ Conferma Vendita"):
                portfolio_id = int(sell_selection.split("#")[1].split(" ")[0])
                conn = get_db_connection()
                pos = conn.execute(
                    "SELECT total_cost FROM portfolio WHERE id = ?", (portfolio_id,)
                ).fetchone()

                if pos:
                    net_profit = sell_price - sell_commission - pos["total_cost"]
                    roi = (net_profit / pos["total_cost"]) * 100

                    conn.execute("""
                        UPDATE portfolio SET
                            sell_price = ?, sell_date = ?, sell_source = ?,
                            net_profit = ?, roi_pct = ?, status = 'sold'
                        WHERE id = ?
                    """, (sell_price, datetime.now().date().isoformat(),
                          sell_source, net_profit, roi, portfolio_id))
                    conn.commit()
                    st.success(f"Vendita registrata. Profitto netto: €{net_profit:,.0f} (ROI: {roi:+.1f}%)")
                    st.cache_data.clear()
                    st.rerun()
                conn.close()

    # --- Storico vendite ---
    if not sold.empty:
        st.markdown("---")
        st.subheader("Storico Vendite")
        sold_display = ["brand", "model", "reference", "buy_price", "sell_price",
                        "net_profit", "roi_pct", "buy_date", "sell_date"]
        available = [c for c in sold_display if c in sold.columns]
        st.dataframe(sold[available].reset_index(drop=True), use_container_width=True)


# ============================================================
# PAGE 5: DIARIO DI BORDO
# ============================================================

elif page == "📝 Diario di Bordo":
    st.title("📝 Diario di Bordo")
    st.caption("Annota decisioni, osservazioni, e lezioni imparate")

    # Crea tabella note se non esiste
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logbook (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT DEFAULT 'nota',
            title TEXT,
            content TEXT NOT NULL,
            mood TEXT DEFAULT '😐',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

    # --- Nuova nota ---
    with st.expander("✍️ Scrivi una nuova nota", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            note_title = st.text_input("Titolo (opzionale)", placeholder="Es: Comprato Seiko SKX007 su eBay")
        with col2:
            note_category = st.selectbox("Categoria", [
                "🛒 Acquisto", "💰 Vendita", "📊 Analisi",
                "💡 Idea", "⚠️ Lezione", "📝 Nota generica"
            ])

        note_content = st.text_area(
            "Contenuto",
            height=150,
            placeholder="Cosa è successo? Perché hai preso questa decisione? Cosa hai imparato?"
        )

        col_mood1, col_mood2 = st.columns([1, 3])
        with col_mood1:
            mood = st.selectbox("Mood", ["🚀", "😊", "😐", "😰", "🤬", "🧠", "💰", "🎉"])

        if st.button("💾 Salva nota", type="primary"):
            if note_content.strip():
                conn = get_db_connection()
                conn.execute("""
                    INSERT INTO logbook (category, title, content, mood)
                    VALUES (?, ?, ?, ?)
                """, (note_category, note_title, note_content, mood))
                conn.commit()
                conn.close()
                st.success("Nota salvata!")
                st.rerun()
            else:
                st.warning("Scrivi qualcosa prima di salvare")

    st.markdown("---")

    # --- Storico note ---
    conn = get_db_connection()
    notes = pd.read_sql_query("""
        SELECT * FROM logbook ORDER BY created_at DESC
    """, conn)
    conn.close()

    if notes.empty:
        st.info("Il diario è vuoto. Scrivi la tua prima nota!")
    else:
        # Filtro per categoria
        categories = ["Tutte"] + notes["category"].unique().tolist()
        filter_cat = st.selectbox("Filtra per categoria", categories)

        if filter_cat != "Tutte":
            notes = notes[notes["category"] == filter_cat]

        st.caption(f"{len(notes)} note totali")

        for _, note in notes.iterrows():
            with st.container():
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    title_display = note["title"] if note["title"] else "Senza titolo"
                    st.markdown(f"**{note['mood']} {title_display}**")
                    st.caption(f"{note['category']} — {note['created_at'][:16]}")
                with header_col2:
                    if st.button("🗑️", key=f"del_{note['id']}"):
                        conn = get_db_connection()
                        conn.execute("DELETE FROM logbook WHERE id = ?", (note["id"],))
                        conn.commit()
                        conn.close()
                        st.rerun()

                st.markdown(note["content"])
                st.markdown("---")


# ============================================================
# PAGE 6: SISTEMA
# ============================================================

elif page == "⚙️ Sistema":
    st.title("⚙️ Stato del Sistema")

    # --- Stato modello ---
    st.subheader("🧠 Modello ML")
    model_meta = load_model_metadata()

    if model_meta:
        col1, col2, col3, col4 = st.columns(4)
        fm = model_meta.get("final_metrics", {})
        col1.metric("Modello", model_meta.get("model_name", "?"))
        col2.metric("MAE", f"€{fm.get('mae', '?')}")
        col3.metric("MAPE", f"{fm.get('mape', '?')}%")
        col4.metric("R²", f"{fm.get('r2', '?')}")

        st.caption(f"Addestrato: {model_meta.get('trained_at', '?')[:16]}  |  "
                   f"Campioni: {model_meta.get('n_samples', '?')}  |  "
                   f"Referenze: {model_meta.get('n_references', '?')}")

        # CV results
        cv = model_meta.get("cv_results", {})
        if cv:
            with st.expander("📊 Risultati Cross-Validation"):
                cv_rows = []
                for name, stats in cv.items():
                    cv_rows.append({
                        "Modello": name,
                        "MAE (€)": stats.get("mae", "—"),
                        "R²": stats.get("r2", "—"),
                        "MAPE~(%)": stats.get("mape_approx", "—"),
                    })
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)
    else:
        st.warning("Nessun modello addestrato. Esegui: `python models.py --train`")

    st.markdown("---")

    # --- Log scraping ---
    st.subheader("🔄 Log Scraping")
    logs = load_scrape_logs()
    if not logs.empty:
        # Rendi più leggibile
        logs_display = logs[["source", "status", "listings_found", "listings_new",
                             "started_at", "error_message"]].copy()
        logs_display["status"] = logs_display["status"].map(
            lambda s: "✅" if s == "success" else "❌" if s == "error" else "🔄"
        )
        st.dataframe(logs_display.head(20), use_container_width=True)
    else:
        st.info("Nessun log di scraping disponibile.")

    st.markdown("---")

    # --- Stato database ---
    st.subheader("🗄️ Database")
    conn = get_db_connection()
    tables_info = {}
    for table in ["watches", "listings", "price_history", "sentiment_data", "portfolio", "scrape_log"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        tables_info[table] = count
    conn.close()

    cols = st.columns(len(tables_info))
    for col, (table, count) in zip(cols, tables_info.items()):
        col.metric(table, f"{count:,}")

    st.markdown("---")

    # --- Configurazione ---
    st.subheader("📝 Configurazione")
    config = load_config()

    with st.expander("Parametri Strategia"):
        strategy = config.get("strategy", {})
        for key, val in strategy.items():
            st.text(f"{key}: {val}")

    with st.expander("Impostazioni Scraping"):
        scraping = config.get("scraping", {})
        st.json(scraping)

    with st.expander("Stato API Key"):
        ebay_cfg = config.get("scraping", {}).get("ebay", {})
        tg_cfg = config.get("alerts", {})

        ebay_ok = bool(ebay_cfg.get("app_id"))
        tg_ok = bool(tg_cfg.get("telegram_token"))

        st.markdown(f"- eBay API: {'✅ Configurata' if ebay_ok else '❌ Non configurata'}")
        st.markdown(f"- Telegram Bot: {'✅ Configurato' if tg_ok else '❌ Non configurato'}")
        st.markdown(f"- Chrono24: {'✅ Abilitato' if config.get('scraping',{}).get('chrono24',{}).get('enabled') else '❌ Disabilitato'}")

        if not ebay_ok or not tg_ok:
            st.info("Modifica `config.json` per configurare le API key.")

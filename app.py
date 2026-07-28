"""
Coral-Sync | Marine Intelligence Dashboard
-------------------------------------------
Prototype UI (dummy data) untuk platform B2B Data-as-a-Service prediksi
risiko coral bleaching & rekomendasi zona tangkap ikan.

Menu:
  1. Peta & Ringkasan     -> peta interaktif zona + KPI nasional
  2. Rekomendasi Zona Tangkap -> ranking zona layak tangkap
  3. Early Warning System -> daftar alert zona berisiko tinggi
  4. Detail Zona          -> drill-down time series & feature importance

Jalankan dengan:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# --------------------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Coral-Sync | Marine Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0E4F66"       # deep ocean teal
ACCENT = "#12A594"        # coral-sync teal accent
WARN = "#F5A623"          # amber - medium risk
DANGER = "#E5484D"        # red - high risk
SAFE = "#3DD68C"          # green - low risk / optimal zone
BG = "#F4FAF9"

st.markdown(f"""
<style>
    .main {{ background-color: {BG}; }}
    #MainMenu, footer {{visibility: hidden;}}

    .csync-header {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #0A7C82 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .csync-header h1 {{ margin: 0; font-size: 26px; font-weight: 700; }}
    .csync-header p {{ margin: 2px 0 0 0; font-size: 13px; opacity: 0.85; }}
    .csync-badge {{
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        text-align: right;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #E3ECEA;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricLabel"] {{ font-size: 13px; color: #567; }}

    .zone-card {{
        background: white;
        border: 1px solid #E3ECEA;
        border-left: 5px solid {ACCENT};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .zone-card.high {{ border-left-color: {DANGER}; }}
    .zone-card.medium {{ border-left-color: {WARN}; }}
    .zone-card.low {{ border-left-color: {SAFE}; }}

    .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        color: white;
    }}
    .pill-high {{ background: {DANGER}; }}
    .pill-medium {{ background: {WARN}; }}
    .pill-low {{ background: {SAFE}; }}

    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
    }}
    section[data-testid="stSidebar"] * {{ color: #EAF6F4 !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 14px; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# DUMMY DATA GENERATION (cached so it's stable within a session)
# --------------------------------------------------------------------------------------

ZONE_META = [
    # name, province, lat, lon
    ("Raja Ampat", "Papua Barat", -0.50, 130.50),
    ("Bunaken", "Sulawesi Utara", 1.60, 124.75),
    ("Wakatobi", "Sulawesi Tenggara", -5.30, 123.75),
    ("Komodo - Labuan Bajo", "Nusa Tenggara Timur", -8.55, 119.45),
    ("Nusa Penida", "Bali", -8.72, 115.54),
    ("Gili Matra", "Nusa Tenggara Barat", -8.35, 116.04),
    ("Karimunjawa", "Jawa Tengah", -5.85, 110.45),
    ("Kepulauan Derawan", "Kalimantan Timur", 2.28, 118.24),
    ("Pulau Weh", "Aceh", 5.85, 95.32),
    ("Kepulauan Seribu", "DKI Jakarta", -5.60, 106.60),
    ("Selat Sunda", "Banten - Lampung", -6.05, 105.50),
    ("Takabonerate", "Sulawesi Selatan", -6.50, 121.00),
    ("Teluk Cenderawasih", "Papua", -2.50, 135.50),
    ("Kepulauan Alor", "Nusa Tenggara Timur", -8.30, 124.50),
    ("Banda Neira", "Maluku", -4.50, 129.90),
    ("Pulau Morotai", "Maluku Utara", 2.30, 128.30),
    ("Kepulauan Anambas", "Kepulauan Riau", 3.00, 106.20),
    ("Kepulauan Natuna", "Kepulauan Riau", 4.00, 108.20),
    ("Ujung Kulon", "Banten", -6.75, 105.40),
    ("Pulau Bangka", "Sulawesi Utara", 1.75, 125.15),
]


@st.cache_data
def generate_zone_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, (name, prov, lat, lon) in enumerate(ZONE_META):
        grid_id = f"{lat:.1f}_{lon:.1f}"
        dhw = round(float(np.clip(rng.gamma(2.0, 1.8), 0, 12)), 2)
        sst = round(float(rng.normal(29.2, 1.1) + dhw * 0.15), 2)
        chlor_a = round(float(np.clip(rng.gamma(2.0, 0.15), 0.02, 1.5)), 3)
        wind = round(float(rng.uniform(2, 14)), 1)
        precip = round(float(rng.gamma(2.0, 15)), 1)

        # bleaching risk probability derived mainly from DHW (mirrors NOAA BAA logic)
        base_risk = np.clip(dhw / 10.0, 0, 1)
        noise = rng.normal(0, 0.06)
        risk_proba = float(np.clip(base_risk * 0.8 + noise + 0.05, 0.02, 0.98))

        if risk_proba >= 0.66:
            risk_level = "Tinggi"
        elif risk_proba >= 0.33:
            risk_level = "Sedang"
        else:
            risk_level = "Rendah"

        # fishing suitability: higher chlorophyll & lower bleaching risk -> better zone
        fishing_score = float(np.clip(
            (chlor_a / 1.5) * 55 + (1 - risk_proba) * 45 + rng.normal(0, 4), 5, 99
        ))

        last_alert_days = int(rng.integers(0, 21)) if risk_level != "Rendah" else int(rng.integers(15, 90))

        rows.append({
            "zone_id": f"CS-{i+1:03d}",
            "grid_id": grid_id,
            "zone_name": name,
            "province": prov,
            "lat": lat,
            "lon": lon,
            "dhw": dhw,
            "sst": sst,
            "chlor_a": chlor_a,
            "wind_speed": wind,
            "precip_mm": precip,
            "risk_proba": round(risk_proba, 3),
            "risk_level": risk_level,
            "fishing_score": round(fishing_score, 1),
            "last_alert_days": last_alert_days,
            "last_updated": (datetime.now() - timedelta(hours=int(rng.integers(1, 20)))),
        })
    return pd.DataFrame(rows)


@st.cache_data
def generate_timeseries(zone_id: str, dhw_now: float, sst_now: float, seed_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(zone_id)) % (2**32) + seed_offset)
    weeks = pd.date_range(end=datetime.now(), periods=12, freq="W")
    trend = np.linspace(-1, 1, 12) * rng.uniform(0.3, 1.2)
    dhw_series = np.clip(dhw_now + trend * dhw_now * 0.5 + rng.normal(0, 0.3, 12), 0, None)
    dhw_series[-1] = dhw_now
    sst_series = sst_now - (dhw_series.max() - dhw_series) * 0.12 + rng.normal(0, 0.15, 12)
    sst_series[-1] = sst_now
    proba_series = np.clip(dhw_series / 10.0 * 0.8 + rng.normal(0, 0.04, 12) + 0.05, 0.02, 0.98)
    return pd.DataFrame({
        "week": weeks, "dhw": dhw_series.round(2), "sst": sst_series.round(2),
        "risk_proba": proba_series.round(3),
    })


df = generate_zone_data()

RISK_COLOR = {"Tinggi": DANGER, "Sedang": WARN, "Rendah": SAFE}
RISK_PILL = {"Tinggi": "pill-high", "Sedang": "pill-medium", "Rendah": "pill-low"}
RISK_CARD = {"Tinggi": "high", "Sedang": "medium", "Rendah": "low"}

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown(f"""
<div class="csync-header">
    <div>
        <h1> Coral-Sync <span style="font-weight:400; font-size:16px;">Marine Intelligence Dashboard</span></h1>
        <p>Prediksi Risiko Coral Bleaching &amp; Rekomendasi Zona Tangkap Ikan &nbsp;•&nbsp; Data as a Service (B2B)</p>
    </div>
    <div class="csync-badge">
        Klien: <b>PT Nusantara Marine Export</b><br/>
        Paket: <b>Enterprise API</b> · Diperbarui {datetime.now().strftime('%d %b %Y, %H:%M')} WIB
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("###  Coral-Sync")
    st.caption("Marine Intelligence Technology · Stacking Ensemble AI")
    page = st.radio(
        "Navigasi",
        ["🗺️ Peta & Ringkasan", "🎣 Rekomendasi Zona Tangkap", "🚨 Early Warning System", "🔍 Detail Zona"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("**Model aktif:** RF + XGBoost + LogReg (Stacking)")
    st.caption("**Sumber data (demo):** NOAA CRW, MODIS Chlor-a, ERA5-Land")
    st.markdown("---")
    st.caption(" Seluruh data pada dashboard ini adalah *dummy* untuk keperluan demo antarmuka.")

# --------------------------------------------------------------------------------------
# PAGE 1 — PETA & RINGKASAN
# --------------------------------------------------------------------------------------
if page == " Peta & Ringkasan":

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Zona Dipantau", f"{len(df)}")
    k2.metric("Zona Risiko Tinggi", f"{(df.risk_level == 'Tinggi').sum()}", delta="perlu perhatian", delta_color="inverse")
    k3.metric("Rata-rata DHW Nasional", f"{df.dhw.mean():.2f} °C-weeks")
    k4.metric("Zona Layak Tangkap Optimal", f"{(df.fishing_score >= 65).sum()}")
    k5.metric("Akurasi Model (Test Set)", "91.4%", delta="PR-AUC 0.87")

    st.markdown("### Peta Sebaran Zona")
    c1, c2 = st.columns([3, 1])
    with c2:
        view_mode = st.radio("Mode tampilan", ["Risiko Bleaching", "Rekomendasi Tangkap"], index=0)
        risk_filter = st.multiselect("Filter tingkat risiko", ["Tinggi", "Sedang", "Rendah"],
                                      default=["Tinggi", "Sedang", "Rendah"])
        prov_filter = st.multiselect("Filter provinsi", sorted(df.province.unique()))

    map_df = df[df.risk_level.isin(risk_filter)]
    if prov_filter:
        map_df = map_df[map_df.province.isin(prov_filter)]

    with c1:
        if view_mode == "Risiko Bleaching":
            fig = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", color="risk_level",
                size=(map_df["risk_proba"] * 30 + 8),
                color_discrete_map=RISK_COLOR,
                hover_name="zone_name",
                hover_data={"province": True, "dhw": True, "sst": True, "risk_proba": True,
                            "lat": False, "lon": False},
                category_orders={"risk_level": ["Tinggi", "Sedang", "Rendah"]},
                zoom=3.4, height=560,
            )
        else:
            fig = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", color="fishing_score",
                size=(map_df["fishing_score"] / 2 + 8),
                color_continuous_scale=["#E5484D", "#F5A623", "#3DD68C"],
                hover_name="zone_name",
                hover_data={"province": True, "chlor_a": True, "fishing_score": True,
                            "lat": False, "lon": False},
                zoom=3.4, height=560,
            )
        fig.update_layout(mapbox_style="carto-positron",
                           margin=dict(l=0, r=0, t=0, b=0),
                           mapbox_center={"lat": -2.5, "lon": 118})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ringkasan Distribusi")
    c3, c4 = st.columns(2)
    with c3:
        dist = df.risk_level.value_counts().reindex(["Tinggi", "Sedang", "Rendah"]).fillna(0)
        fig_bar = px.bar(dist, x=dist.index, y=dist.values,
                          color=dist.index, color_discrete_map=RISK_COLOR,
                          labels={"x": "Tingkat Risiko", "y": "Jumlah Zona"},
                          title="Distribusi Tingkat Risiko Bleaching")
        fig_bar.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_bar, use_container_width=True)
    with c4:
        fig_scatter = px.scatter(df, x="dhw", y="chlor_a", color="risk_level",
                                  color_discrete_map=RISK_COLOR, size="fishing_score",
                                  hover_name="zone_name",
                                  labels={"dhw": "Degree Heating Weeks", "chlor_a": "Klorofil-a (mg/m³)"},
                                  title="DHW vs Klorofil-a per Zona")
        fig_scatter.update_layout(height=340)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Tabel Seluruh Zona")
    show_df = df[["zone_name", "province", "grid_id", "dhw", "sst", "chlor_a",
                   "risk_level", "risk_proba", "fishing_score"]].sort_values("risk_proba", ascending=False)
    st.dataframe(
        show_df.rename(columns={
            "zone_name": "Zona", "province": "Provinsi", "grid_id": "Grid ID",
            "dhw": "DHW (°C-wk)", "sst": "SST (°C)", "chlor_a": "Klorofil-a",
            "risk_level": "Risiko", "risk_proba": "Prob. Bleaching", "fishing_score": "Skor Tangkap",
        }),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------------------
# PAGE 2 — REKOMENDASI ZONA TANGKAP
# --------------------------------------------------------------------------------------
elif page == " Rekomendasi Zona Tangkap":
    st.markdown("###  Rekomendasi Wilayah Penangkapan Ikan Optimal")
    st.caption("Ranking dihitung dari kombinasi ketersediaan klorofil-a (proksi kelimpahan ikan) "
               "dan rendahnya risiko stres termal karang di sekitar zona.")

    min_score = st.slider("Ambang batas skor kelayakan minimum", 0, 100, 50)
    reco_df = df[df.fishing_score >= min_score].sort_values("fishing_score", ascending=False)

    st.info(f" Ditemukan **{len(reco_df)} zona** yang direkomendasikan untuk operasi penangkapan "
            f"berdasarkan kondisi oseanografi terkini.")

    for _, row in reco_df.iterrows():
        cls = RISK_CARD[row.risk_level]
        pill = RISK_PILL[row.risk_level]
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"""
            <div class="zone-card {cls}">
                <b>{row.zone_name}</b> — {row.province} &nbsp; <span class="pill {pill}">Risiko {row.risk_level}</span><br/>
                <span style="font-size:13px; color:#556;">
                Skor Kelayakan Tangkap: <b>{row.fishing_score:.1f}/100</b> ·
                Klorofil-a: {row.chlor_a} mg/m³ ·
                SST: {row.sst}°C ·
                DHW: {row.dhw}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.progress(int(row.fishing_score))

    fig = px.bar(reco_df.head(15), x="fishing_score", y="zone_name", orientation="h",
                 color="risk_level", color_discrete_map=RISK_COLOR,
                 labels={"fishing_score": "Skor Kelayakan Tangkap", "zone_name": ""},
                 title="Top 15 Zona Rekomendasi Tangkap")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# PAGE 3 — EARLY WARNING SYSTEM
# --------------------------------------------------------------------------------------
elif page == " Early Warning System":
    st.markdown("###  Early Warning System — Stres Termal Karang")
    st.caption("Notifikasi otomatis dikirim ke email/API klien ketika probabilitas bleaching "
               "suatu zona melewati ambang batas alert (BAA ≥ 3, standar NOAA).")

    alerts = df[df.risk_level.isin(["Tinggi", "Sedang"])].sort_values("risk_proba", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Alert Aktif", len(alerts))
    c2.metric("Alert Kritis (Risiko Tinggi)", (alerts.risk_level == "Tinggi").sum())
    c3.metric("Rata-rata Waktu Sejak Alert", f"{alerts.last_alert_days.mean():.0f} hari")

    st.markdown("---")

    for _, row in alerts.iterrows():
        cls = RISK_CARD[row.risk_level]
        pill = RISK_PILL[row.risk_level]
        severity_icon = "🔴" if row.risk_level == "Tinggi" else "🟠"
        action = ("Rekomendasi: alihkan armada dari zona ini, tingkatkan frekuensi pemantauan SST harian."
                   if row.risk_level == "Tinggi" else
                   "Rekomendasi: pantau perkembangan mingguan, siapkan rencana kontingensi rute.")
        st.markdown(f"""
        <div class="zone-card {cls}">
            {severity_icon} <b>{row.zone_name}</b>, {row.province}
            &nbsp; <span class="pill {pill}">{row.risk_level}</span>
            &nbsp; <span style="font-size:12px; color:#789;">terdeteksi {row.last_alert_days} hari lalu</span>
            <br/>
            <span style="font-size:13px; color:#556;">
            Probabilitas bleaching: <b>{row.risk_proba*100:.1f}%</b> ·
            DHW: {row.dhw} °C-weeks · SST: {row.sst}°C
            </span><br/>
            <span style="font-size:13px; color:#0E4F66;"> {action}</span>
        </div>
        """, unsafe_allow_html=True)

    if alerts.empty:
        st.success("Tidak ada zona dengan status alert saat ini.")

# --------------------------------------------------------------------------------------
# PAGE 4 — DETAIL ZONA
# --------------------------------------------------------------------------------------
elif page == " Detail Zona":
    st.markdown("###  Detail & Riwayat Zona")
    zone_name = st.selectbox("Pilih zona", df.zone_name.tolist())
    row = df[df.zone_name == zone_name].iloc[0]
    ts = generate_timeseries(row.zone_id, row.dhw, row.sst, seed_offset=7)

    pill = RISK_PILL[row.risk_level]
    st.markdown(f"""
    <div class="zone-card {RISK_CARD[row.risk_level]}">
        <b style="font-size:18px;">{row.zone_name}</b> — {row.province}
        &nbsp; <span class="pill {pill}">Risiko {row.risk_level}</span><br/>
        <span style="font-size:13px; color:#556;">
        Grid ID: {row.grid_id} · Koordinat: {row.lat:.2f}, {row.lon:.2f} ·
        Diperbarui: {row.last_updated.strftime('%d %b %Y %H:%M')} WIB
        </span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilitas Bleaching", f"{row.risk_proba*100:.1f}%")
    m2.metric("DHW", f"{row.dhw} °C-weeks")
    m3.metric("SST", f"{row.sst} °C")
    m4.metric("Skor Kelayakan Tangkap", f"{row.fishing_score:.0f}/100")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.week, y=ts.dhw, name="DHW (°C-weeks)",
                                  line=dict(color=DANGER, width=3)))
        fig.update_layout(title="Tren Degree Heating Weeks (12 minggu)", height=340,
                           yaxis_title="DHW", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts.week, y=ts.sst, name="SST (°C)",
                                   line=dict(color=PRIMARY, width=3)))
        fig2.update_layout(title="Tren Sea Surface Temperature (12 minggu)", height=340,
                            yaxis_title="°C", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.area(ts, x="week", y="risk_proba",
                        title="Tren Probabilitas Risiko Bleaching Mingguan")
        fig3.update_traces(line_color=ACCENT, fillcolor="rgba(18,165,148,0.2)")
        fig3.add_hline(y=0.66, line_dash="dot", line_color=DANGER, annotation_text="Ambang Tinggi")
        fig3.add_hline(y=0.33, line_dash="dot", line_color=WARN, annotation_text="Ambang Sedang")
        fig3.update_layout(height=340, yaxis_title="Probabilitas", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        importance = pd.DataFrame({
            "Fitur": ["DHW 12w mean", "SST", "DHW momentum", "Klorofil-a std",
                      "Wind stress", "Curah hujan 4w"],
            "Kontribusi": [0.29, 0.24, 0.16, 0.13, 0.10, 0.08],
        })
        fig4 = px.bar(importance, x="Kontribusi", y="Fitur", orientation="h",
                      title="Feature Importance Model (Stacking Ensemble)",
                      color_discrete_sequence=[ACCENT])
        fig4.update_layout(yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("##### Parameter Oseanografi Terkini")
    st.dataframe(pd.DataFrame({
        "Parameter": ["Sea Surface Temperature", "Degree Heating Weeks", "Klorofil-a",
                      "Kecepatan Angin", "Curah Hujan"],
        "Nilai": [f"{row.sst} °C", f"{row.dhw} °C-weeks", f"{row.chlor_a} mg/m³",
                  f"{row.wind_speed} m/s", f"{row.precip_mm} mm"],
        "Sumber Data (demo)": ["NOAA Coral Reef Watch", "NOAA Coral Reef Watch",
                                "MODIS-Aqua L3SMI", "ERA5-Land", "ERA5-Land"],
    }), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
st.markdown("---")
st.caption("Coral-Sync API · Marine Intelligence Technology · Prototipe UI dengan dummy data — "
           "Universitas Airlangga, Fakultas Teknologi Maju dan Multidisiplin, 2026.")

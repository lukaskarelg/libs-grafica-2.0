# app.py
"""
LIBS — versión con picos discretos tipo histograma
- Gráfico de barras verticales (histograma de picos)
- Etiquetas en cada barra
- Color más oscuro para longitudes de onda repetidas
- Permite cargar base de datos desde archivo o ingresarla manualmente
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io

st.set_page_config(layout="wide", page_title="LIBS — Picos discretos")

st.title("🔬 LIBS — Espectros discretos tipo histograma con comparación")

st.sidebar.header("📂 Cargar archivos")

# --- Entradas de datos ---
upload_measure = st.sidebar.file_uploader("Archivo de medidas (CSV o TXT)", type=["csv", "txt"])
paste_measure = st.sidebar.text_area("O pega los datos de medición (wavelength,counts)", height=120)

upload_db = st.sidebar.file_uploader("Base de datos (CSV o TXT)", type=["csv", "txt"])
paste_db = st.sidebar.text_area("O pega la base de datos (formato: Wavelength (nm),Sum,...)", height=120)

tolerance_nm = st.sidebar.number_input("Tolerancia de coincidencia (nm)", min_value=0.0, value=0.2, step=0.01)

# --- Lectura flexible ---
def read_measurements(file, pasted_text):
    df = None
    if file is not None:
        df = pd.read_csv(file)
    elif pasted_text.strip():
        df = pd.read_csv(io.StringIO(pasted_text), header=None, names=["wavelength", "counts"])
    if df is None:
        return None
    # Normalizar columnas
    cols = [c.lower().strip() for c in df.columns]
    if "wavelength" not in cols or "counts" not in cols:
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["wavelength", "counts"]
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["counts"] = pd.to_numeric(df["counts"], errors="coerce")
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)
    return df

def read_db(file, pasted_text):
    df = None
    if file is not None:
        df = pd.read_csv(file)
    elif pasted_text.strip():
        df = pd.read_csv(io.StringIO(pasted_text))
    if df is None:
        return None
    col_names = [c.lower() for c in df.columns]
    if "wavelength (nm)" in col_names and "sum" in col_names:
        df = df.rename(columns={
            df.columns[col_names.index("wavelength (nm)")]: "wavelength",
            df.columns[col_names.index("sum")]: "counts"
        })
    else:
        df = df.iloc[:, :2]
        df.columns = ["wavelength", "counts"]
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["counts"] = pd.to_numeric(df["counts"], errors="coerce")
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)
    return df

# --- Cargar datos ---
meas_df = read_measurements(upload_measure, paste_measure)
db_df = read_db(upload_db, paste_db)

if meas_df is None:
    st.warning("Cargue un archivo o pegue datos de medidas para continuar.")
    st.stop()

# --- Filtro 10% del máximo ---
max_counts = meas_df["counts"].max()
threshold = 0.1 * max_counts
filtered = meas_df[meas_df["counts"] >= threshold].reset_index(drop=True)

st.markdown(f"### 🔎 Filtro aplicado: se eliminaron valores menores al 10% del máximo ({threshold:.2e})")
st.write(f"{len(filtered)} puntos conservados de {len(meas_df)} totales")

# --- Colores según repetición ---
counts_repeats = filtered["wavelength"].duplicated(keep=False)
colors = ["#0040ff" if not rep else "#001a66" for rep in counts_repeats]  # azul oscuro si se repite

# --- Gráfico tipo histograma ---
st.subheader("Gráfica de espectro (picos discretos tipo histograma)")
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(filtered["wavelength"], filtered["counts"], color=colors, width=0.05)

# Etiquetas sobre cada barra
for idx, row in filtered.iterrows():
    y = row["counts"]
    wl = row["wavelength"]
    label_color = "black" if not counts_repeats[idx] else "#333333"
    ax.text(wl, y * 1.02, f"{wl:.2f}", ha="center", va="bottom", fontsize=8, rotation=90, color=label_color)

ax.set_xlabel("Longitud de onda (nm)")
ax.set_ylabel("Cuentas")
ax.set_title("Espectro LIBS — Picos discretos (≥10% del máximo)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
st.pyplot(fig)

# --- Coincidencias con base ---
def match_peaks(df_meas, df_db, tol):
    if df_db is None or df_db.empty:
        return pd.DataFrame(columns=["wavelength_meas", "counts_meas", "wavelength_db", "delta_nm"])
    matches = []
    wl_m = df_meas["wavelength"].values
    cnt_m = df_meas["counts"].values
    wl_db = df_db["wavelength"].values
    cnt_db = df_db["counts"].values
    for i in range(len(wl_m)):
        diffs = np.abs(wl_db - wl_m[i])
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            matches.append({
                "wavelength_meas": wl_m[i],
                "counts_meas": cnt_m[i],
                "wavelength_db": wl_db[j],
                "delta_nm": diffs[j],
                "counts_db": cnt_db[j]
            })
    return pd.DataFrame(matches)

st.markdown("---")
st.subheader("Resultados de comparación con base de datos")

if db_df is None:
    st.info("No se cargó ni pegó una base de datos de referencia.")
else:
    matches = match_peaks(filtered, db_df, tolerance_nm)
    if matches.empty:
        st.warning("No se encontraron coincidencias dentro de la tolerancia.")
    else:
        st.success(f"Se encontraron {len(matches)} coincidencias.")
        st.dataframe(matches)
        csv = matches.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar coincidencias (CSV)", data=csv, file_name="coincidencias.csv", mime="text/csv")


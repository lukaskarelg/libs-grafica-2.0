# app.py
"""
Versión modificada: LIBS espectros discretos sin suavizado
- Muestra picos discretos según los datos cargados
- Filtra los valores menores al 10% del máximo de cuentas
- Acepta bases de datos con formato "Wavelength (nm), Sum, ..."
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io

st.set_page_config(layout="wide", page_title="LIBS — Espectros discretos")

st.title("🔬 LIBS — Gráfica de picos discretos y comparación con base de datos")

st.sidebar.header("Cargar archivos")

# --- Subir archivos ---
upload_measure = st.sidebar.file_uploader("Archivo de medidas (CSV)", type=["csv", "txt"])
paste_measure = st.sidebar.text_area("O pega los datos (wavelength,counts)", height=120)

upload_db = st.sidebar.file_uploader("Base de datos (CSV con columnas Wavelength (nm), Sum, ...)", type=["csv", "txt"])

tolerance_nm = st.sidebar.number_input("Tolerancia de coincidencia (nm)", min_value=0.0, value=0.2, step=0.01)

st.write("### Ejemplo de archivo de medidas")
st.code("400.12,123\n400.25,130\n400.38,95", language="text")

# --- Lectura flexible ---
def read_measurements(file, pasted_text):
    df = None
    if file is not None:
        df = pd.read_csv(file)
    elif pasted_text.strip():
        df = pd.read_csv(io.StringIO(pasted_text), header=None, names=["wavelength","counts"])
    if df is None:
        return None
    # Normalizar nombres
    cols = [c.lower().strip() for c in df.columns]
    if "wavelength" not in cols or "counts" not in cols:
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["wavelength","counts"]
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["counts"] = pd.to_numeric(df["counts"], errors="coerce")
    df = df.dropna()
    df = df.sort_values("wavelength").reset_index(drop=True)
    return df

def read_db(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    # Buscar columnas correctas
    col_names = [c.lower() for c in df.columns]
    if "wavelength (nm)" in col_names and "sum" in col_names:
        df = df.rename(columns={
            df.columns[col_names.index("wavelength (nm)")]: "wavelength",
            df.columns[col_names.index("sum")]: "counts"
        })
    else:
        # Tomar primeras dos columnas si los nombres no coinciden
        df = df.iloc[:, :2]
        df.columns = ["wavelength", "counts"]
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["counts"] = pd.to_numeric(df["counts"], errors="coerce")
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)
    return df

# --- Cargar datos ---
meas_df = read_measurements(upload_measure, paste_measure)
db_df = read_db(upload_db)

if meas_df is None:
    st.warning("Cargue un archivo o pegue datos de medidas para continuar.")
    st.stop()

# --- Filtro 10% del máximo ---
max_counts = meas_df["counts"].max()
threshold = 0.1 * max_counts
filtered = meas_df[meas_df["counts"] >= threshold].reset_index(drop=True)

st.markdown(f"### 🔎 Filtro aplicado: se eliminaron valores menores al 10% del máximo ({threshold:.2e})")
st.write(f"{len(filtered)} puntos conservados de {len(meas_df)} totales")

# --- Gráfica discreta ---
st.subheader("Gráfica de espectro (picos discretos)")
fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(filtered["wavelength"], filtered["counts"], color="blue", s=30)
ax.set_xlabel("Longitud de onda (nm)")
ax.set_ylabel("Cuentas")
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
ax.set_title("Espectro LIBS — Picos filtrados (≥10% del máximo)")
st.pyplot(fig)

# --- Coincidencias con base ---
def match_peaks(df_meas, df_db, tol):
    if df_db is None or df_db.empty:
        return pd.DataFrame(columns=["wavelength_meas","counts_meas","wavelength_db","delta_nm"])
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
    st.info("No se cargó una base de datos de referencia.")
else:
    matches = match_peaks(filtered, db_df, tolerance_nm)
    if matches.empty:
        st.warning("No se encontraron coincidencias dentro de la tolerancia.")
    else:
        st.success(f"Se encontraron {len(matches)} coincidencias.")
        st.dataframe(matches)
        csv = matches.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar coincidencias (CSV)", data=csv, file_name="coincidencias.csv", mime="text/csv")

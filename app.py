# app.py
"""
App Streamlit para visualizar datos LIBS y comparar picos con una base de datos de líneas.
Formato esperado:
 - Medidas CSV: columnas -> wavelength, counts
 - Base de datos CSV: columns -> element, wavelength
Instalación:
 pip install streamlit pandas numpy matplotlib
Ejecutar:
 streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

st.set_page_config(layout="wide", page_title="LIBS: graficar y comparar")

st.title("🔬 LIBS — Gráfica y comparación con base de datos")

# Sidebar: subir archivos o pegar
st.sidebar.header("Cargar datos")
upload_measure = st.sidebar.file_uploader("Sube archivo de medidas (CSV) con columnas wavelength,counts", type=["csv","txt"])
paste_measure = st.sidebar.text_area("O pega aquí datos de medidas (wavelength,counts) - opcional", height=120)
upload_db = st.sidebar.file_uploader("Sube base de datos (CSV) con columnas element,wavelength", type=["csv","txt"])
paste_db = st.sidebar.text_area("O pega aquí base de datos (element,wavelength) - opcional", height=120)

st.sidebar.markdown("---")
st.sidebar.header("Parámetros de análisis")
smoothing_window = st.sidebar.number_input("Ventana de suavizado (puntos) — 1 = sin suavizar", min_value=1, value=3, step=1)
peak_prominence = st.sidebar.number_input("Umbral mínimo (prominence) relativo para picos (0-1)", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
tolerance_nm = st.sidebar.number_input("Tolerancia para emparejar líneas (nm)", min_value=0.0, value=0.2, step=0.01)

st.write("Formato de ejemplo (medidas): `wavelength,counts`")
st.code("400.12, 123\n400.25, 130\n400.38, 95\n...", language="text")
st.write("Formato de ejemplo (base de datos): `element,wavelength`")
st.code("Fe, 404.58\nCa, 422.67\nMg, 285.21\n...", language="text")

# ---- funciones utilitarias ----
def read_measurements(file, pasted_text):
    df = None
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python')  # intento flexible
    elif pasted_text and pasted_text.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted_text), header=None, names=["wavelength","counts"])
        except Exception:
            df = None
    if df is not None:
        # intentar encontrar/normalizar columnas
        cols = [c.lower().strip() for c in df.columns]
        if "wavelength" not in cols or "counts" not in cols:
            # intentar asumir dos columnas
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["wavelength","counts"]
            else:
                raise ValueError("No se encontraron columnas wavelength y counts")
        df = df[["wavelength","counts"]].dropna()
        df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
        df["counts"] = pd.to_numeric(df["counts"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("wavelength").reset_index(drop=True)
    return df

def read_db(file, pasted_text):
    df = None
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python')
    elif pasted_text and pasted_text.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted_text), header=None, names=["element","wavelength"])
        except Exception:
            df = None
    if df is not None:
        cols = [c.lower().strip() for c in df.columns]
        if "wavelength" not in cols or "element" not in cols:
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["element","wavelength"]
            else:
                raise ValueError("La base de datos debe tener columnas element y wavelength")
        df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
        df = df.dropna(subset=["wavelength"]).reset_index(drop=True)
    return df

def smooth_counts(counts, window):
    if window <= 1:
        return counts
    # media móvil simple
    kernel = np.ones(window) / window
    return np.convolve(counts, kernel, mode='same')

def detect_peaks(wavelengths, counts, rel_prominence=0.08):
    # algoritmo simple: un punto es pico si es mayor que vecinos y supera un umbral relativo
    peaks_idx = []
    if len(counts) < 3:
        return np.array(peaks_idx, dtype=int)
    # Umbral absoluto basado en rango
    minc, maxc = np.min(counts), np.max(counts)
    threshold = minc + rel_prominence * (maxc - minc)
    for i in range(1, len(counts)-1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1] and counts[i] >= threshold:
            peaks_idx.append(i)
    return np.array(peaks_idx, dtype=int)

def match_peaks(meas_wl, meas_counts, db_df, peaks_idx, tol_nm):
    matches = []
    if db_df is None or db_df.empty or len(peaks_idx)==0:
        return pd.DataFrame(columns=["meas_wavelength","meas_counts","element","db_wavelength","delta_nm"])
    db_wl = db_df["wavelength"].values
    db_elem = db_df["element"].values
    for idx in peaks_idx:
        wl = meas_wl[idx]
        cnt = meas_counts[idx]
        # buscar la línea más cercana en la base
        diffs = np.abs(db_wl - wl)
        best_i = np.argmin(diffs)
        if diffs[best_i] <= tol_nm:
            matches.append({
                "meas_wavelength": wl,
                "meas_counts": cnt,
                "element": db_elem[best_i],
                "db_wavelength": db_wl[best_i],
                "delta_nm": diffs[best_i]
            })
    matches_df = pd.DataFrame(matches)
if not matches_df.empty and "meas_wavelength" in matches_df.columns:
    matches_df = matches_df.sort_values("meas_wavelength").reset_index(drop=True)
return matches_df

# ---- cargar datos ----
try:
    meas_df = read_measurements(upload_measure, paste_measure)
except Exception as e:
    st.error(f"Error leyendo medidas: {e}")
    st.stop()

try:
    db_df = read_db(upload_db, paste_db)
except Exception as e:
    st.error(f"Error leyendo base de datos: {e}")
    st.stop()

if meas_df is None:
    st.info("Carga tus datos de medidas o pégalos en el panel lateral. (wavelength,counts)")
    st.stop()

# procesamiento
wavelengths = meas_df["wavelength"].values
counts_raw = meas_df["counts"].values
counts_sm = smooth_counts(counts_raw, smoothing_window)
peaks_idx = detect_peaks(wavelengths, counts_sm, rel_prominence=peak_prominence)

# Gráfica
st.subheader("Gráfica de espectro")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(wavelengths, counts_sm)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Counts (smoothed)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
ax.set_title("Espectro LIBS (suavizado)")
# marcar picos
if len(peaks_idx)>0:
    ax.scatter(wavelengths[peaks_idx], counts_sm[peaks_idx], marker='v', s=40)
    for i in peaks_idx:
        ax.text(wavelengths[i], counts_sm[i]*1.02, f"{wavelengths[i]:.2f}", fontsize=8, rotation=45, ha='left', va='bottom')
st.pyplot(fig)

st.markdown("---")
st.subheader("Resultados de detección y comparación")

if db_df is None:
    st.info("No se cargó una base de datos. Solo se mostrará la lista de picos detectados.")
    peaks_table = pd.DataFrame({
        "meas_wavelength": wavelengths[peaks_idx],
        "meas_counts": counts_sm[peaks_idx]
    }).reset_index(drop=True)
    st.dataframe(peaks_table)
    csv = peaks_table.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar picos detectados (CSV)", data=csv, file_name="picos_detectados.csv", mime="text/csv")
else:
    matched = match_peaks(wavelengths, counts_sm, db_df, peaks_idx, tolerance_nm)
    if matched.empty:
        st.warning("No se encontraron coincidencias dentro de la tolerancia especificada.")
    else:
        st.write(f"Se encontraron {len(matched)} coincidencias (tolerancia = {tolerance_nm} nm).")
        st.dataframe(matched)
        csv = matched.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar coincidencias (CSV)", data=csv, file_name="coincidencias.csv", mime="text/csv")

st.markdown("### Tabla de picos detectados (para inspección)")
detected_df = pd.DataFrame({
    "meas_wavelength": wavelengths[peaks_idx],
    "meas_counts": counts_sm[peaks_idx]
}).reset_index(drop=True)
st.table(detected_df)

st.markdown("---")
st.caption("Nota: el detector de picos es simple y pensado para análisis rápido. Para análisis más robusto se recomienda usar métodos de detección de picos (por ejemplo scipy.signal.find_peaks) y calibración espectral previa.")

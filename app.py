# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Análisis LIBS & Comparación NIST", layout="wide")

st.title("🔬 Sistema de Análisis de Espectro LIBS")
st.markdown("""
Esta aplicación procesa datos espectrales, filtra el ruido de fondo y compara los picos encontrados 
con una base de datos de líneas de emisión teóricas.
""")

# --- 1. CARGA DE DATOS EXPERIMENTALES ---
st.sidebar.header("1. Carga de Datos")
archivo_exp = st.sidebar.file_uploader("Subir archivo del espectrómetro (.txt)", type=["txt", "csv"])

# --- 2. CARGA DE BASE DE DATOS (REFERENCIA) ---
st.sidebar.header("2. Base de Datos (Referencias)")
# Opción para subir la DB o usar una demo
archivo_db = st.sidebar.file_uploader("Subir Base de Datos (.csv)", type=["csv"])

def cargar_datos_experimentales(uploaded_file):
    """
    Lee el formato específico saltando 4 líneas y convirtiendo columnas.
    """
    try:
        df = pd.read_csv(
            uploaded_file,
            skiprows=4,
            sep='\t', 
            header=None,
            names=["Etiqueta", "Intensidad", "LongitudOnda"], # Renombrado para claridad
            engine="python"
        )
        # Limpieza y conversión
        df["Intensidad"] = pd.to_numeric(df["Intensidad"], errors='coerce')
        df["LongitudOnda"] = pd.to_numeric(df["LongitudOnda"], errors='coerce')
        df.dropna(inplace=True)
        return df.sort_values(by="LongitudOnda")
    except Exception as e:
        st.error(f"Error leyendo el archivo experimental: {e}")
        return None

def algoritmo_comparacion(picos_exp, df_ref, tolerancia=0.5):
    """
    Compara los picos experimentales con la base de datos dentro de una tolerancia (nm).
    Retorna un DataFrame con los "matches".
    """
    resultados = []

    # Iteramos por cada pico experimental encontrado
    for idx, row_exp in picos_exp.iterrows():
        wl_exp = row_exp['LongitudOnda']
        int_exp = row_exp['Intensidad']

        # Filtramos la DB buscando líneas cercanas (± tolerancia)
        # Asumimos que la DB tiene columnas: ['Wavelength', 'Element', 'Ionization']
        matches = df_ref[
            (df_ref['Wavelength'] >= wl_exp - tolerancia) & 
            (df_ref['Wavelength'] <= wl_exp + tolerancia)
        ].copy()

        if not matches.empty:
            # Calculamos la diferencia absoluta
            matches['Diferencia_nm'] = (matches['Wavelength'] - wl_exp).abs()
            matches['Error_Relativo'] = matches['Diferencia_nm'] / matches['Wavelength']
            
            # Nos quedamos con el match más cercano (menor diferencia)
            # Ojo: En LIBS real a veces hay solapamiento, aquí tomamos el "mejor candidato"
            best_match = matches.loc[matches['Diferencia_nm'].idxmin()]

            resultados.append({
                "Wavelength Exp (nm)": wl_exp,
                "Intensidad Exp": int_exp,
                "Elemento": best_match['Element'],
                "Ionización": best_match.get('Ionization', '-'),
                "Wavelength Teórico (nm)": best_match['Wavelength'],
                "Diferencia (nm)": best_match['Diferencia_nm'], # |Teorico - Exp|
                "Probabilidad Relativa": best_match.get('Rel_Intensity', 'N/A') # Si la DB tiene intensidad relativa
            })
    
    return pd.DataFrame(resultados)

# --- LÓGICA PRINCIPAL ---

if archivo_exp is not None:
    # 1. Procesar Experimental
    df_raw = cargar_datos_experimentales(archivo_exp)
    
    if df_raw is not None:
        # 2. Filtrado (Tu lógica original)
        max_val = df_raw["Intensidad"].max()
        mean_val = df_raw["Intensidad"].mean()
        
        # Sliders para ajustar umbrales en tiempo real (Interactivo)
        col1, col2 = st.columns(2)
        factor_umbral = col1.slider("Factor de Umbral (sobre la media)", 0.1, 5.0, 0.9, 0.1)
        umbral = factor_umbral * mean_val
        
        col2.metric("Umbral de Intensidad", f"{umbral:.2f}", delta_color="inverse")
        
        datos_filtrados = df_raw[df_raw["Intensidad"] >= umbral].copy()
        
        st.subheader("📊 Visualización del Espectro")
        
       # --- GRAFICACIÓN CON PLOTLY (ESTILO DISCRETO / VLINES) ---
        fig = go.Figure()

        # 1. Las líneas verticales (Usamos Bar con ancho muy fino para simular vlines)
        fig.add_trace(go.Bar(
            x=datos_filtrados["LongitudOnda"],
            y=datos_filtrados["Intensidad"],
            width=0.3,                # <--- Ajusta esto: grosor de la línea en nm (muy fino)
            marker_color='black',     # Color de la línea vertical
            name='Magnitud',
            hoverinfo='skip'          # Ocultamos info aquí para no duplicar con el punto
        ))

        # 2. Los puntos rojos encima (Las cabezas de los picos)
        fig.add_trace(go.Scatter(
            x=datos_filtrados["LongitudOnda"],
            y=datos_filtrados["Intensidad"],
            mode='markers',
            name='Pico Detectado',
            marker=dict(
                color='red', 
                size=7,               # Tamaño del punto
                symbol='circle',
                line=dict(color='white', width=1) # Borde blanco para resaltar
            ),
            # Tooltip profesional
            hovertemplate="<b>Longitud de Onda:</b> %{x:.2f} nm<br><b>Intensidad:</b> %{y:.2f} a.u.<extra></extra>"
        ))

        # Configuración del Layout
        fig.update_layout(
            title="Espectro Discreto (Picos Filtrados)",
            xaxis_title="Longitud de Onda [nm]",
            yaxis_title="Intensidad [a.u]",
            template="plotly_white",
            height=550,
            showlegend=True,
            bargap=0  # Asegura que las barras no se agrupen raro
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- SECCIÓN DE COMPARACIÓN DE BASE DE DATOS ---
        st.markdown("---")
        st.subheader("📚 Comparación con Base de Datos")

        if archivo_db is not None:
            # Asumimos que la DB tiene columnas: Wavelength, Element, Ionization, Rel_Intensity
            # Ejemplo de formato CSV esperado:
            # Wavelength,Element,Ionization,Rel_Intensity
            # 589.00,Na,I,1000
            
            try:
                df_ref = pd.read_csv(archivo_db)
                # Normalizar nombres de columnas si es necesario
                # Asegúrate de que tu CSV tenga al menos 'Wavelength' y 'Element'
                
                tolerancia = st.slider("Tolerancia de Búsqueda (nm)", 0.01, 1.0, 0.1)
                
                # Ejecutar algoritmo
                df_resultados = algoritmo_comparacion(datos_filtrados, df_ref, tolerancia)
                
                if not df_resultados.empty:
                    # Formateo de la tabla para resaltar diferencias
                    st.write(f"Se encontraron **{len(df_resultados)}** coincidencias probables.")
                    
                    # Estilizar la tabla (Pandas Styler)
                    st.dataframe(
                        df_resultados.style.format({
                            "Wavelength Exp (nm)": "{:.3f}",
                            "Wavelength Teórico (nm)": "{:.3f}",
                            "Diferencia (nm)": "{:.4f}",
                            "Intensidad Exp": "{:.1f}"
                        }).background_gradient(subset=["Diferencia (nm)"], cmap="Reds"),
                        use_container_width=True
                    )
                    
                    # Descargar reporte
                    csv_res = df_resultados.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar Tabla de Comparación",
                        csv_res,
                        "reporte_libs.csv",
                        "text/csv"
                    )
                    
                    # Resumen de elementos encontrados
                    st.write("#### 🧪 Elementos Detectados (Conteo de Picos)")
                    conteo = df_resultados["Elemento"].value_counts().reset_index()
                    conteo.columns = ["Elemento", "Cantidad de Picos"]
                    st.bar_chart(conteo.set_index("Elemento"))
                    
                else:
                    st.warning("No se encontraron coincidencias con la tolerancia actual.")

            except Exception as e:
                st.error(f"Error procesando la base de datos: {e}. Revisa que las columnas sean: Wavelength, Element, Ionization")
        else:
            st.info("👆 Por favor sube un archivo CSV con la Base de Datos para realizar la comparación.")
            
            # Generar CSV de ejemplo para el usuario
            st.markdown("**¿No tienes una base de datos a mano?**")
            ejemplo_db = pd.DataFrame({
                "Wavelength": [589.0, 589.6, 309.2, 308.2, 656.3],
                "Element": ["Na", "Na", "Al", "Al", "H"],
                "Ionization": ["I", "I", "I", "I", "I"],
                "Rel_Intensity": [1000, 500, 800, 600, 200]
            })
            csv_ejemplo = ejemplo_db.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar plantilla CSV ejemplo", csv_ejemplo, "db_referencia_ejemplo.csv")

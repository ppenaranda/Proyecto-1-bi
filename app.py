import streamlit as st
import requests

st.set_page_config(page_title="Detector de noticias falsas", layout="centered")

st.title("Detector de noticias falsas")
st.markdown("Ingresa una noticia para detectar si es falsa o verdadera")

# Formulario de predicción
with st.form(key="form_predict"):
    texto = st.text_area("Texto de la noticia")
    submit = st.form_submit_button("Analizar")

if submit:
    if texto.strip():
        respuesta = requests.post("http://localhost:8080/predict", json={"textos": [texto]})
        if respuesta.status_code == 200:
            pred = respuesta.json()["resultados"][0]
            label = "Verdadera" if pred["prediccion"] == 1 else "Falsa"
            st.subheader(f"Resultado: {label}")
            st.write(f"Probabilidad de ser falsa: **{pred['probabilidad_fake']:.2f}**")
            st.write(f"Probabilidad de ser verdadera: **{pred['probabilidad_real']:.2f}**")
        else:
            st.error("Error al conectarse con la API.")
    else:
        st.warning("Por favor ingresa un texto.")

# Sección de reentrenamiento
st.markdown("---")
st.subheader("Reentrenamiento del modelo")

with st.form(key="form_retrain"):
    nuevos_textos = st.text_area("Nuevos textos")
    etiquetas = st.text_input("Etiquetas correspondientes (0 falsa, 1 verdadera, separadas por coma)")
    entrenar = st.form_submit_button("Reentrenar modelo")

if entrenar:
    try:
        textos_lista = [t.strip() for t in nuevos_textos.split("\n") if t.strip()]
        etiquetas_lista = [int(e.strip()) for e in etiquetas.split(",")]

        if len(textos_lista) != len(etiquetas_lista):
            st.error("El número de textos y etiquetas no coincide.")
        else:
            data = {"textos": textos_lista, "etiquetas": etiquetas_lista}
            respuesta = requests.post("http://localhost:8080/retrain", json=data)
            if respuesta.status_code == 200:
                resultado = respuesta.json()
                st.success("Modelo reentrenado exitosamente.")
                st.write(f"Precisión: **{resultado['precision']:.2f}**")
                st.write(f"Recall: **{resultado['recall']:.2f}**")
                st.write(f"F1-score: **{resultado['f1_score']:.2f}**")
            else:
                st.error("Ocurrió un error al reentrenar.")
    except:
        st.error("Error en el formato de las etiquetas o textos.")

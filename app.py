# ───── Instalación automática de paquetes necesarios ─────
import subprocess
import sys

def instalar_paquete(nombre):
    try:
        __import__(nombre)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", nombre])

for paquete in ['seaborn', 'tensorflow', 'matplotlib', 'reportlab', 'pillow', 'streamlit']:
    instalar_paquete(paquete)

# ───── Imports ─────
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO
import tempfile

# ───── Configuración de la app ─────
st.set_page_config(page_title="Diagnóstico Caña de Azúcar", layout="centered")
st.title("🌿 Diagnóstico con Deep Learning - Caña de Azúcar")
st.markdown("Sube una imagen de hoja para predecir si está sana o enferma usando tres modelos entrenados.")

MODEL_PATHS = {
    "MobileNetV2": "MobileNetV2_model.h5",
    "ResNet50": "ResNet50_model.h5",
    "EfficientNetB0": "EfficientNetB0_model.h5"
}
CLASS_NAMES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# ───── Funciones ─────
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def plot_prediction_bar(predictions, class_names):
    probs = predictions[0]
    fig, ax = plt.subplots()
    ax.barh(class_names, probs, color='green')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilidad')
    ax.set_title('Distribución de Predicción')
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')
    st.pyplot(fig)

def plot_heatmap(predictions, class_names):
    fig, ax = plt.subplots()
    sns.heatmap(np.array([predictions[0]]), annot=True, cmap="YlGnBu", xticklabels=class_names, yticklabels=["Probabilidades"], ax=ax)
    ax.set_title("Mapa de Calor de Predicción")
    st.pyplot(fig)

def plot_scatter(predictions, class_names):
    probs = predictions[0]
    fig, ax = plt.subplots()
    ax.scatter(class_names, probs, color='blue', s=100)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidad')
    ax.set_title('Mapa de Dispersión de Confianza')
    for i, txt in enumerate(probs):
        ax.annotate(f"{txt:.2f}", (class_names[i], probs[i]), textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(fig)

def generar_pdf(imagen_pil, pred_class, probabilidad, model_name, class_names, predictions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph(f"📄 Diagnóstico con {model_name}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Resultado:</b> {pred_class}", styles['Normal']))
    story.append(Paragraph(f"<b>Confianza:</b> {probabilidad:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))

    temp_img_path = tempfile.mktemp(suffix=".png")
    imagen_pil.save(temp_img_path)
    story.append(Paragraph("📸 Imagen analizada:", styles['Normal']))
    story.append(RLImage(temp_img_path, width=300, height=300))
    story.append(Spacer(1, 12))

    story.append(Paragraph("📊 Distribución de probabilidades:", styles['Normal']))
    prob_text = ""
    for i, clase in enumerate(class_names):
        prob = predictions[0][i] * 100
        prob_text += f"{clase}: {prob:.2f}%<br/>"
    story.append(Paragraph(prob_text, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ───── Carga y predicción ─────
uploaded_file = st.file_uploader("📷 Sube imagen de hoja", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📸 Imagen cargada", use_container_width=True)
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = preprocess_image(image)
    resultados = []

    for model_name, model_path in MODEL_PATHS.items():
        with st.spinner(f"🔎 Evaluando con {model_name}..."):
            model = load_model(model_path)
            predictions = model.predict(image_array)
            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

            resultados.append({
                "modelo": model_name,
                "clase": predicted_class,
                "confianza": confidence,
                "predictions": predictions
            })

            with st.expander(f"🧠 Resultados del modelo: {model_name}", expanded=True):
                st.success(f"🔍 Predicción: **{predicted_class}** con **{confidence:.2f}%** de confianza")
                
                if st.button(f"📊 Ver gráfico de barras - {model_name}"):
                    plot_prediction_bar(predictions, CLASS_NAMES)
                if st.button(f"🔥 Ver mapa de calor - {model_name}"):
                    plot_heatmap(predictions, CLASS_NAMES)
                if st.button(f"🔵 Ver dispersión - {model_name}"):
                    plot_scatter(predictions, CLASS_NAMES)

                pdf = generar_pdf(image, predicted_class, confidence, model_name, CLASS_NAMES, predictions)
                st.download_button(
                    label=f"📄 Descargar PDF - {model_name}",
                    data=pdf,
                    file_name=f"{model_name}_diagnostico.pdf",
                    mime="application/pdf"
                )

    mejor_modelo = max(resultados, key=lambda x: x['confianza'])
    st.subheader("🏆 Modelo más confiable")
    st.success(f"✅ El modelo **{mejor_modelo['modelo']}** fue el más confiable con una predicción de **{mejor_modelo['clase']}** ({mejor_modelo['confianza']:.2f}%)")

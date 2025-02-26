import streamlit as st
import os
import re
from PIL import Image
import tempfile

# Extraer ID del archivo desde la URL de Google Drive
def extract_file_id(url):
    pattern = r'\/d\/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# URL de Google Drive
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/19pnfyKQJafNtInRRvsCS-b1D6AmGpbkH/view?usp=drive_link"
GOOGLE_DRIVE_FILE_ID = extract_file_id(GOOGLE_DRIVE_URL)

# Configuración de la página
st.set_page_config(
    page_title="Detector de Imágenes IA",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado con tamaño de imagen fijo
def add_custom_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .glassmorphism {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .title {
            font-weight: 700;
            color: #1E1E1E;
            font-size: 3.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #7928CA, #FF0080);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            font-weight: 400;
            color: #4A4A4A;
            font-size: 1.2rem;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #7928CA, #FF0080);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .prediction-result {
            font-weight: 600;
            font-size: 1.5rem;
            color: #7928CA;
            margin-top: 1rem;
        }
        
        /* Contenedor de imagen de tamaño fijo */
        .image-container {
            width: 100%;
            height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            margin: 1rem 0;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 300px;
            object-fit: contain;
        }
        
        /* Mejor organización de secciones */
        .upload-section {
            margin-bottom: 20px;
        }
        
        .analyze-section {
            margin-top: 20px;
        }
        
        /* Estilo del pie de página */
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.8rem;
            margin-top: 40px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Añadir gradiente de fondo
def add_bg_gradient():
    bg_gradient = """
    <div class="main" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
    """
    st.markdown(bg_gradient, unsafe_allow_html=True)

# Aplicar estilos
add_custom_css()
add_bg_gradient()

# Sección de encabezado
st.markdown('<p class="title">Detector de Imágenes Falsas</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Carga una imagen para determinar si fue generada por IA o es real</p>', unsafe_allow_html=True)

# Descargar pesos del modelo desde Google Drive
@st.cache_resource
def download_model_weights():
    try:
        import gdown
        
        # Crear un directorio temporal para los pesos
        temp_dir = tempfile.gettempdir()
        weights_path = os.path.join(temp_dir, "model_weights.pth")
        
        # Verificar si el archivo ya existe (para evitar volver a descargarlo)
        if os.path.exists(weights_path):
            st.sidebar.success("✅ Usando pesos del modelo previamente descargados")
            return weights_path
            
        # Verificar si el ID del archivo es válido
        if not GOOGLE_DRIVE_FILE_ID:
            st.sidebar.error("⚠️ ID de archivo de Google Drive no válido")
            return None
            
        # Descargar archivo desde Google Drive
        with st.sidebar.status("Descargando el modelo predictor de IA..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            result = gdown.download(url, weights_path, quiet=False)
            
        if result and os.path.exists(weights_path):
            st.sidebar.success("✅ Se descargaron los pesos del modelo correctamente")
            return weights_path
        else:
            st.sidebar.error("⚠️ Error al descargar los pesos del modelo")
            return None
    
    except ImportError:
        st.sidebar.error("⚠️ Por favor instala gdown: pip install gdown")
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ Error descargando pesos: {str(e)}")
        return None

# Funcionalidad de carga del modelo
@st.cache_resource
def load_model():
    # Importar PyTorch dentro de la función para evitar problemas con el observador de archivos
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # Usar siempre CPU para un despliegue coherente
    device = torch.device("cpu")
    
    # Crear modelo con arquitectura ConvNeXt (de test_model.py)
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # Clasificador personalizado como en test_model.py
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, 2)
    
    # Intentar descargar pesos desde Google Drive
    weights_path = download_model_weights()
    
    # Si tenemos una copia local como respaldo
    if not weights_path and os.path.exists("model_weights.pth"):
        weights_path = "model_weights.pth"
        st.sidebar.info("Usando pesos de manera local")
    
    # Cargar pesos si están disponibles
    if weights_path and os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            st.sidebar.success("✅ Pesos del modelo cargados correctamente")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error al cargar los pesos: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Pesos del modelo no disponibles. Usando pesos pre-entrenados de ImageNet.")
    
    model.eval()
    return model, device

# Preprocesamiento de imágenes
def preprocess_image(image):
    # Importar PyTorch dentro de la función para evitar problemas con el observador de archivos
    import torch
    from torchvision import transforms
    
    # Usar las mismas transformaciones que en test_model.py
    preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

# Crear dos columnas para el diseño
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    
    # Sección de carga
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Carga una imagen...", 
        type=["jpg", "jpeg", "png"],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Crear columnas para la imagen y el botón para organizar mejor el espacio
    if uploaded_file is not None:
        try:
            # Mostrar imagen con tamaño fijo
            image = Image.open(uploaded_file).convert('RGB')
            
            # Usando HTML/CSS para un contenedor de imagen de tamaño fijo
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Botón de análisis - ahora aparecerá directamente debajo de la imagen
            st.markdown('<div class="analyze-section">', unsafe_allow_html=True)
            analyze_button = st.button("Analizar imagen")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Procesar al hacer clic en el botón
            if analyze_button:
                with st.spinner("Procesando imagen..."):
                    # Importar PyTorch solo cuando sea necesario
                    import torch
                    
                    # Cargar modelo (en caché después de la primera carga)
                    model, device = load_model()
                    
                    # Procesar imagen
                    input_tensor = preprocess_image(image)
                    input_tensor = input_tensor.to(device)
                    
                    # Ejecutar inferencia
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    # Obtener resultados de predicción
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
                    pred_class = output.argmax(dim=1).item()
                    
                    # Nombres de clases
                    class_names = ["Generada por IA", "Creada por un humano"]
                    
                    # Mostrar resultados
                    st.markdown("### Resultado")
                    st.markdown(
                        f'<div class="prediction-result">Esta imagen fue {class_names[pred_class].lower()}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Barras de confianza
                    st.markdown("### Nivel de confianza")
                    for i, class_name in enumerate(class_names):
                        st.markdown(f"**{class_name}:** {probabilities[i]:.2f}%")
                        st.progress(float(probabilities[i]/100))
        
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    st.markdown("### Pasos")
    st.write("""
    1. Subir tu imagen al selector de archivos
    2. El modelo de IA analizará la imagen
    3. Visualiza los resultados del análisis
    
    Esta aplicación utiliza un modelo avanzado de aprendizaje profundo entrenado para distinguir entre imágenes generadas por IA y fotografías creadas por humanos.
    """)
    
    st.markdown("### Acerca del modelo")
    st.write("""
    Nuestro modelo está construido con PyTorch y utiliza una arquitectura ConvNeXt con capas clasificadoras personalizadas.
    
    Ha sido entrenado con miles de imágenes para reconocer los patrones y artefactos que diferencian las imágenes 
    generadas por IA de las fotografías creadas por humanos.
    """)
    
    # Añadir algunas métricas o estadísticas sobre el modelo
    st.markdown("### Rendimiento del modelo")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Precisión", value="91.3%")
    with col_b:
        st.metric(label="Clases", value="2")
    with col_c:
        st.metric(label="Velocidad", value="0.3s")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
footer = """
<div style="text-align: center; margin-top: 40px; padding: 20px;">
    <p style="color: #666; font-size: 0.8rem;">© 2025 Detector de Imágenes IA | Creado con Streamlit y PyTorch</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
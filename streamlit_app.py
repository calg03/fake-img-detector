import streamlit as st
import os
import re
from PIL import Image
import tempfile
import gc

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

# CSS personalizado (simplificado para evitar problemas con div)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #7928CA, #FF0080);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        margin-top: 20px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    h1 {
        font-weight: 700;
        color: #1E1E1E;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #7928CA, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        font-weight: 400;
        color: #4A4A4A;
        font-size: 1.2rem;
    }
    
    .prediction-text {
        font-weight: 600;
        font-size: 1.5rem;
        color: #7928CA;
        margin-top: 1rem;
    }
    
    .stProgress > div > div {
        background-color: #7928CA;
    }
</style>
""", unsafe_allow_html=True)

# Añadir gradiente de fondo
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
             background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); z-index: -1;">
</div>
""", unsafe_allow_html=True)

# Sección de encabezado usando HTML nativo de Streamlit
st.title("Detector de Imágenes Falsas")
st.subheader("Carga una imagen para determinar si fue generada por IA o es real")

@st.cache_data(persist=True)
def download_model_weights():
    try:
        import gdown
        
        # Crear un directorio temporal para los pesos
        temp_dir = tempfile.gettempdir()
        weights_path = os.path.join(temp_dir, "model_weights.pth")
        
        # Verificar si el archivo ya existe
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

@st.cache_resource
def load_model():
    # Importar PyTorch dentro de la función
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # Usar CPU para un despliegue coherente
    device = torch.device("cpu")
    
    # Crear modelo con arquitectura ConvNeXt
    model = models.convnext_large(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, 2)
    
    # Fijar todos los parámetros excepto clasificador
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
                
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
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            st.sidebar.success("✅ Pesos del modelo cargados correctamente")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error al cargar los pesos: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Pesos del modelo no disponibles. Usando pesos pre-entrenados.")
    
    model.eval()
    return model, device

@st.cache_data
def preprocess_image(image):
    import torch
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

# Layout con Streamlit nativo
col1, col2 = st.columns([3, 2])

with col1:
    # Contenedor principal con efecto de sombra
    with st.container():
        # Uploader de imágenes directamente sin divs HTML
        uploaded_file = st.file_uploader(
            "Carga una imagen...", 
            type=["jpg", "jpeg", "png"],
            help="Formatos aceptados: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                # Gestión de memoria
                if 'previous_image' not in st.session_state:
                    st.session_state.previous_image = None
                
                if st.session_state.previous_image and st.session_state.previous_image != uploaded_file.name:
                    gc.collect()
                
                st.session_state.previous_image = uploaded_file.name
                
                # Mostrar imagen
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_container_width=True)
                
                # Botón de análisis
                analyze_button = st.button("Analizar imagen")
                
                if analyze_button:
                    with st.spinner("Procesando imagen..."):
                        import torch
                        
                        # Cargar modelo
                        model, device = load_model()
                        
                        # Procesar imagen
                        input_tensor = preprocess_image(image)
                        input_tensor = input_tensor.to(device)
                        
                        # Inferencia
                        with torch.no_grad():
                            output = model(input_tensor)
                        
                        # Resultados
                        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
                        pred_class = output.argmax(dim=1).item()
                        
                        # Liberar memoria
                        del input_tensor, output
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Nombres de clases
                        class_names = ["Creada por un humano", "Generada por IA"]
                        
                        # Mostrar resultados
                        st.markdown("### Resultado")
                        st.markdown(f"<p class='prediction-text'>Esta imagen fue {class_names[pred_class].lower()}</p>", unsafe_allow_html=True)
                        
                        # Barras de confianza
                        st.markdown("### Nivel de confianza")
                        for i, class_name in enumerate(class_names):
                            st.markdown(f"**{class_name}:** {probabilities[i]:.2f}%")
                            st.progress(float(probabilities[i]/100))
            
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

with col2:
    # Contenedor de información
    with st.container():
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

# Pie de página
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px;">
    <p style="color: #666; font-size: 0.8rem;">© 2025 Detector de Imágenes IA | Creado con Streamlit y PyTorch</p>
</div>
""", unsafe_allow_html=True)

# Limpiar recursos temporales
if st.session_state.get('should_cleanup', False):
    temp_dir = tempfile.gettempdir()
    try:
        for item in os.listdir(temp_dir):
            if item.startswith("st_temp_"):
                file_path = os.path.join(temp_dir, item)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path, ignore_errors=True)
    except:
        pass
    
    # Marcar limpieza como completada
    st.session_state.should_cleanup = False
else:
    # Programar limpieza para la próxima ejecución
    st.session_state.should_cleanup = True
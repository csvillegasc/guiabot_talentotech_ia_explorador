from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("wordnet")

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Cargar el dataset de preguntas, respuestas y categorías
def load_data():
    df = pd.read_csv("Dataset/service_policies.csv")  # Asegúrate de que el archivo existe
    df["pregunta"] = df["pregunta"].str.strip().str.lower()  # Normalizar las preguntas
    df["respuesta"] = df["respuesta"].str.strip()  # Limpiar respuestas
    df["categoria"] = df["categoria"].str.strip().str.lower()  # Normalizar las categorías
    return df.to_dict("records")  # Convertir a lista de diccionarios

# Cargar las preguntas y respuestas
qa_list = load_data()

# Diccionario para almacenar la sesión del usuario
user_sessions = {}

# Función para obtener sinónimos
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

# Crear la aplicación FastAPI
app = FastAPI()

# Montar la carpeta estática para servir imágenes locales
app.mount("/static", StaticFiles(directory="static"), name="static")

# Nueva API para obtener categorías disponibles
@app.get("/categories/", tags=["Chatbot"])
def get_categories():
    """
    Retorna una lista de todas las categorías disponibles en el dataset.
    """
    categories = sorted(set(qa["categoria"] for qa in qa_list))
    return JSONResponse(content={"categorias": categories})

# Ruta para el chatbot con validación de consentimiento de datos y consulta
@app.get("/chatbot/", tags=["Chatbot"])
def chatbot(
    session_id: str = Query(..., title="Identificador único de usuario"),
    query: str = Query(None, title="Consulta del usuario"),
    category: str = Query(None, title="Categoría (opcional)")
):
    try:
        if session_id not in user_sessions:
            user_sessions[session_id] = {"consent": False, "first_message": True}

        # Si el usuario envía el primer mensaje, mostrar el mensaje de bienvenida
        if user_sessions[session_id]["first_message"]:
            user_sessions[session_id]["first_message"] = False
            return JSONResponse(content={
                "respuesta": "Hola soy 🤖GuIABot, tu asistente virtual. ¡Voy a acompañarte en tus solicitudes! "
                             "\nPara tu tranquilidad, te informamos que trataremos tus datos personales "
                             "de acuerdo con nuestras políticas de tratamiento de datos personales. "
                             "\n\n¿Estás de acuerdo con el tratamiento de tus datos personales?"
                             "\n1. SÍ"
                             "\n2. NO"
            })

        # Validar consentimiento antes de continuar
        if not user_sessions[session_id]["consent"]:
            if query.strip() == "1":
                user_sessions[session_id]["consent"] = True
                return JSONResponse(content={"respuesta": "¡Gracias por aceptar! Ahora puedes hacer tus consultas.", "show_categories": True})
            elif query.strip() == "2":
                del user_sessions[session_id]  # Reiniciar sesión
                return JSONResponse(content={"respuesta": "Gracias por tu tiempo. ¡Hasta luego! 🤖"})
            else:
                return JSONResponse(content={
                    "respuesta": "Por favor selecciona una opción válida:\n"
                                 "1. Si estás de acuerdo y continuar con tu solicitud\n"
                                 "2. Para cerrar el chatbot"
                })

        # Si el usuario ya aceptó el consentimiento, procesar su consulta
        print(f"📌 Pregunta recibida: {query} en la categoría: {category}")

        if not query:
            return JSONResponse(content={"respuesta": "Por favor, ingresa una consulta válida."})

        # Tokenizar la consulta y buscar sinónimos
        query_words = word_tokenize(query.lower())
        synonyms = set(query_words)
        for word in query_words:
            synonyms.update(get_synonyms(word))

        # Filtrar preguntas dentro de la categoría si se especifica
        filtered_qa_list = qa_list
        if category:
            filtered_qa_list = [qa for qa in qa_list if qa["categoria"].lower() == category.lower()]

        # Buscar coincidencias dentro de la categoría si se especifica o en todas si no hay categoría seleccionada
        results = [qa for qa in filtered_qa_list if any(s in qa["pregunta"] for s in synonyms)]

        if not results:
            return JSONResponse(content={"respuesta": "No se encontraron preguntas que coincidan con tu búsqueda", "encuestas": []})

        return JSONResponse(content={"respuesta": "Aquí tienes algunas preguntas que podrían ayudarte", "encuestas": results})

    except Exception as e:
        print(f"⚠️ Error en chatbot: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Mantener la interfaz de usuario original con funcionalidad corregida
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GuIAbot - Chatbot de Preguntas</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            background-color: #f4f4f4; 
            color: #333; 
            margin: 20px; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
        }

        /* Encabezado con imagen */
        #header-container { 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            background-color: #007bff; 
            color: white; 
            padding: 15px; 
            width: 80%; 
            max-width: 900px; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
        }

        h1 { 
            font-size: 24px; 
            margin-right: 15px; 
        }

        #chat-image { 
            width: 50px; 
            height: 50px; 
            border-radius: 50%; 
            background-color: white; 
            padding: 5px; 
        }

        /* Contenedor principal del chat */
        #chatbox-container { 
            width: 90%; 
            max-width: 750px; 
            margin-top: 20px; 
            background-color: white; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            padding: 15px; 
            text-align: left;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Espacio de chat */
        #chatbox { 
            width: 100%; 
            max-height: 250px; 
            overflow-y: auto; 
            border: 1px solid #ccc; 
            padding: 10px; 
            background-color: #f9f9f9; 
            color: black; 
            border-radius: 5px; 
            box-sizing: border-box; 
        }

        /* Mensajes */
        .user { 
            color: black; 
            font-weight: bold; 
        }

        .bot { 
            color: #808080;  
            font-weight: bold; 
        }

        .question { 
            color: #a9a9a9; 
            font-weight: bold; 
        }

        .answer { 
            color: #a9a9a9; 
            font-weight: bold; 
        }

        .error { 
            color: red; 
            font-weight: bold; 
        }

        /* Selector de categorías */
        #categorySelect { 
            width: 100%; 
            padding: 8px; 
            border-radius: 5px; 
            border: 1px solid #ccc; 
            display: none; 
            margin-top: 10px; 
            background-color: white; 
        }

        /* Entrada de usuario */
        #userInput { 
            width: calc(100% - 100px); 
            padding: 10px; 
            border: 1px solid #ccc; 
            border-radius: 5px; 
            margin-top: 10px; 
            font-size: 16px; 
        }

        /* Botón de enviar */
        button { 
            width: 90px; 
            padding: 10px; 
            background-color: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin-left: 10px; 
            font-size: 16px; 
        }

        button:hover { 
            background-color: #0056b3; 
        }
    </style>
</head>
<body>
    <!-- Encabezado -->
    <div id="header-container">
        <h1>Bienvenido a GuIAbot</h1>
        <img id="chat-image" src="/static/chatbot_image.jpg" alt="Chatbot">
    </div>

    <!-- Contenedor del chat -->
    <div id="chatbox-container">
        <div id="chatbox"></div>

        <!-- Selector de categorías -->
        <select id="categorySelect">
            <option value="">Todas las Categorías</option>
        </select>

        <!-- Entrada del usuario y botón de enviar -->
        <div style="display: flex; align-items: center; width: 100%;">
            <input type="text" id="userInput" placeholder="Escribe tu pregunta aquí...">
            <button onclick="sendMessage()">Enviar</button>
        </div>
    </div>

    <script>
        let sessionId = Math.random().toString(36).substring(7);

        function sendMessage() {
            let input = document.getElementById("userInput").value;
            let category = document.getElementById("categorySelect").value;

            let chatbox = document.getElementById("chatbox");
            chatbox.innerHTML += "<p class='user'><strong>Tú:</strong> " + input + "</p>";

            fetch("/chatbot/?session_id=" + sessionId + "&query=" + input + "&category=" + category)
                .then(response => response.json())
                .then(data => {
                    chatbox.innerHTML += "<p class='bot'><strong>GuIAbot:</strong> " + data.respuesta + "</p>";

                    // Activar el selector de categorías si aún no está activo
                    if (data.show_categories) {
                        document.getElementById("categorySelect").style.display = "block";

                        fetch("/categories/")
                            .then(response => response.json())
                            .then(catData => {
                                let categorySelect = document.getElementById("categorySelect");
                                categorySelect.innerHTML = "<option value=''>Todas las Categorías</option>";
                                catData.categorias.forEach(category => {
                                    let option = document.createElement("option");
                                    option.value = category;
                                    option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
                                    categorySelect.appendChild(option);
                                });
                            });
                    }

                    // Mostrar preguntas y respuestas con formato de color
                    if (data.encuestas && data.encuestas.length > 0) {
                        data.encuestas.forEach(encuesta => {
                            chatbox.innerHTML += "<p class='question'><strong>Pregunta:</strong> " + encuesta["pregunta"] + "</p>";
                            chatbox.innerHTML += "<p class='answer'><strong>Respuesta:</strong> " + encuesta["respuesta"] + "</p>";
                        });
                    }

                    // Desplazar automáticamente hacia abajo
                    chatbox.scrollTop = chatbox.scrollHeight;
                });

            document.getElementById("userInput").value = "";
        }
    </script>
</body>
</html>



"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=html_content)


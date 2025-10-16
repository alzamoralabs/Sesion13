### main.py ### UTEC - Sesion 13 - RAG
# por Boris Alzamora - 2025

from langchain_ollama.llms import OllamaLLM # Establece la conexión con modelos de Ollama
from langchain_openai import ChatOpenAI # Establece la conexión con modelos de OpenAI
from langchain_core.prompts import ChatPromptTemplate # Permite crear prompts para modelos de lenguaje
from vector import load_reviews # Función para cargar reviews y crear un retriever
from dotenv import load_dotenv # Carga variables de entorno desde un archivo .env

load_dotenv() # Cargar variables de entorno desde el archivo .env

# Selección de modelo
aimode = input("Elige AI (1: OpenAI, 2: Llama 3.2 via Ollama): ")
if aimode == "1":
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
else:
    model = OllamaLLM(model="llama3.2", temperature=0.2)

# Cargar reviews y crear retriever
retriever = load_reviews(aimode)

# Crear prompt asociando directamente la información recuperada
# No es la mejor forma, pero es la más sencilla para este ejemplo
template = """
Eres un experto en responder preguntas sobre restaurantes buenazos de Lima, Peru.
y recolectas informacion de las reviews de los influencers gastronómicos mas reconocidos de Lima.
Tienes acceso a la siguiente informacion de restaurantes:
Nombre Restaurante, Autor Review, Calificacion (del 0 al 10, donde 10 es la mejor), Resumen de Review,
Plato estrella, Distrito y Tipo de comida.

Usa la siguiente informacion para responder la {pregunta} de la mejor manera posible.
aqui la informacion que tienes:
{reviews}
"""

# Crear el chain de LLM con prompt
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while(1):
    question = input("Haz tu pregunta (X para Salir): ")
    if question.lower() == "x":
        break
    reviews = retriever.invoke(question) # Recuperar reviews relevantes usando el retriever de Chroma
    result = chain.invoke({"reviews":reviews, "pregunta": question}) # Generar respuesta usando el chain, reemplazando las variables de reviews y pregunta
    print(result)
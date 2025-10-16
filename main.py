from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector import load_reviews
from dotenv import load_dotenv

load_dotenv()

# LLM
aimode = input("Elige AI (1: OpenAI, 2: Ollama): ")
if aimode == "1":
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
else:
    model = OllamaLLM(model="llama3.2")

# Vector DB
retriever = load_reviews(aimode)

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

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while(1):
    question = input("Haz tu pregunta (X para Salir): ")
    if question.lower() == "x":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":reviews, "pregunta": question})
    print(result)
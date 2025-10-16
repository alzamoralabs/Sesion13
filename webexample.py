from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# URL del artículo web a procesar
url = "https://www.anthropic.com/engineering/building-effective-agents"

# 1. Cargar el contenido del artículo web
loader = WebBaseLoader(url)
documents = loader.load()

# 2. Dividir el texto en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Crear embeddings usando un modelo local de Ollama (por ejemplo, 'nomic-embed-text')
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 4. Persistir los embeddings en una base de datos Chroma local
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()

# 5. Ejemplo de búsqueda semántica
query = "criterios para construir agentes efectivos"
results = vectorstore.similarity_search(query, k=3)
for i, res in enumerate(results, 1):
    print(f"Resultado {i}:\n{res.page_content}\n")
from langchain_ollama import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

def load_reviews(aimode: str):
    
    embeddings = None
    db_location = ""
    
    df = pd.read_csv("data/data.csv")

    if aimode != "1":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        db_location = "./chroma_ollama_rest_reviews"
    else:
        embeddings = OpenAIEmbeddings()
        db_location = "./chroma_openai_rest_reviews"
    
    # Load data
    add_documents = not os.path.exists(db_location)

    if add_documents:
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content = "Nombre Restaurante: " + row["Nombre"] + "\n" + "Autor Review:" + row["Autor"] + "\n" +
                "Calificacion:" + row["Calificacion"] + "\n" + "Resumen de Review:" + row["Review"] + "\n" +
                "Plato estrella:"+ row["Plato_Estrella"] + "\n" + "Distrito:" + row["Distrito"] + "\n" +
                "Tipo de comida:" + row["Tipo_comida"] + "\n",
                metadata={"Clasificacion": row["Tipo_comida"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    vector_store = Chroma(
        collection_name= "restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    retriever = vector_store.as_retriever(
        search_kwargs={"k":5} # numero de documentos a recuperar
    )
    return retriever
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_vector_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

print(get_vector_embedding("Hello, world!"))
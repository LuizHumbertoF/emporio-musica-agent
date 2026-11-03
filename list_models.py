import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Inicializa o cliente direto da Groq para listar os modelos
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
modelos = client.models.list()

print("Modelos disponíveis para a sua chave:")
for m in modelos.data:
    print(f"- {m.id}")
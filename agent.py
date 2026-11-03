import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

#carrega chaves do .env
load_dotenv()

products_path = "data/desafio_tecnico_ai_eng - products.csv"
customers_path = "data/desafio_tecnico_ai_eng - customers.csv"
order_items_path = "data/desafio_tecnico_ai_eng - order_items.csv"
orders_path = "data/desafio_tecnico_ai_eng - orders.csv"
promotions_path = "data/desafio_tecnico_ai_eng - promotions.csv"

df_products = pd.read_csv(products_path)
df_customers = pd.read_csv(customers_path)
df_order_items = pd.read_csv(order_items_path)
df_orders = pd.read_csv(orders_path)
df_promotions = pd.read_csv(promotions_path)

def reading_data():
    all_paths = [products_path, customers_path, order_items_path, orders_path, promotions_path]

    for path in all_paths:
        if os.path.exists(path):
           
            df_path = pd.read_csv(path)
            print(f"'{path}' base succesfuly loaded\n")
            print(df_path.head())
        else:
            print(f"File not found in {path}")


@tool
def consult_products(product_name: str) -> str:
    """
    Use esta ferramenta para consultar informações de um produto na loja, como preço e categoria.
    Você deve passar o nome do produto (ou parte dele) como argumento.
    """
    if df_products is None:
        return "Erro: Banco de Dados de Produtos está indisponível no momento."


    #colunas: 'product_id', 'price_brl', 'name', 'category_id', 'description', 'stock_quantity', 'status', 'specs', 'created_at'

    search_result = df_products[df_products["name"].str.contains(product_name, case=False, na=False)] 

    if search_result.empty:
        return f"Nenhum produto com nome similar a: '{product_name}' encontrado."

    awnser = ""
    for _, line in search_result.head(3).iterrows(): 
        awnser += f"- {line['name']}: R$ {line['price_brl']} (Descrição: {line['description']})\n"

    return awnser

def testing_agent():
    llm = ChatGroq(
        model="qwen/qwen3.8-27b", 
        temperature=0.1
    )

    tool_functions = [consult_products]

    agent = create_agent(llm, tool_functions)
    
    system_prompt = (
        "Você é o assistente virtual da loja Empório da Música, em Campo Grande/MS. "
        "Seja amigável, prestativo e use emojis. Responda apenas com base nas ferramentas disponíveis. "
        "Se perguntarem algo fora do contexto da loja de instrumentos, recuse educadamente."
    )
    
    pergunta_cliente = "Vocês têm algum Takamine? Quanto custa?"
    print(f"\nCliente: {pergunta_cliente}\n")
    
   
    awnser = agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=pergunta_cliente)
        ]
    })
    
    print(f"\nResposta Final do Agente:\n{awnser['messages'][-1].content}")

if __name__ == "__main__":
    testing_agent()

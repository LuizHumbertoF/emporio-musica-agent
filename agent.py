import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#carrega chaves do .env
load_dotenv()

products_path = "data/desafio_tecnico_ai_eng - products.csv"
customers_path = "data/desafio_tecnico_ai_eng - customers.csv"
order_items_path = "data/desafio_tecnico_ai_eng - order_items.csv"
orders_path = "data/desafio_tecnico_ai_eng - orders.csv"
promotions_path = "data/desafio_tecnico_ai_eng - promotions.csv"
pdf_path = "data/políticas_da_loja.pdf"

df_products = pd.read_csv(products_path)
df_customers = pd.read_csv(customers_path)
df_order_items = pd.read_csv(order_items_path)
df_orders = pd.read_csv(orders_path)
df_promotions = pd.read_csv(promotions_path)

#carregando e dividindo o PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#cria os embeddings (transforma texto em números) e salva no FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

@tool
def consult_policies(query: str) -> str:
    """
    Use esta ferramenta para consultar as políticas da loja Empório da Música.
    Útil para responder dúvidas sobre trocas, devoluções, garantias, formas de pagamento, 
    horários de funcionamento, frete e prazos de entrega.
    Passe a dúvida do cliente como argumento.
    """
    if not os.path.exists(pdf_path):
        return "Erro: Documento de políticas não encontrado."
    
    resultados = retriever.invoke(query)
    
    contexto = "\n\n".join([doc.page_content for doc in resultados])
    return contexto

def testing_agent():
    llm = ChatGroq(
        model="qwen/qwen3.8-27b", 
        temperature=0.1
    )

    tool_functions = [consult_products, consult_policies]

    agent = create_agent(llm, tool_functions)
    
    system_prompt = (
        "Você é o assistente virtual da loja Empório da Música, em Campo Grande/MS. "
        "Seja amigável, prestativo e use emojis. Responda apenas com base nas ferramentas disponíveis. "
        "Se perguntarem algo fora do contexto da loja de instrumentos, recuse educadamente."
    )
    
    pergunta_cliente = "Comprei um violão mas me arrependi. Quantos dias tenho para devolver?"
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

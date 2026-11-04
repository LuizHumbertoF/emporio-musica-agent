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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langgraph.checkpoint.memory import MemorySaver


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


loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


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


    

    search_result = df_products[df_products["name"].str.contains(product_name, case=False, na=False)] 

    if search_result.empty:
        return f"Nenhum produto com nome similar a: '{product_name}' encontrado."

    awnser = ""
    for _, line in search_result.head(3).iterrows(): 
        awnser += f"- {line['name']}: R$ {line['price_brl']} (Descrição: {line['description']})\n"

    return awnser

@tool
def consult_order_status(order_id: int) -> str:
    """
    Use esta ferramenta para consultar o status de um pedido, código de rastreio e itens comprados.
    Você deve passar o ID do pedido (order_id) numérico como argumento.
    """
    if df_orders is None or df_customers is None:
        return "Erro: Bancos de dados indisponíveis."

    
    order = df_orders[df_orders["order_id"] == order_id]
    if order.empty:
        return f"Pedido #{order_id} não encontrado no sistema."

    order_data = order.iloc[0]
    
    
    customer = df_customers[df_customers["customer_id"] == order_data["customer_id"]]
    customer_name = customer.iloc[0]["name"] if not customer.empty else "Cliente Desconhecido"

    
    items = df_order_items[df_order_items["order_id"] == order_id]
    items_desc = []
    for _, item in items.iterrows():
        prod = df_products[df_products["product_id"] == item["product_id"]]
        prod_name = prod.iloc[0]["name"] if not prod.empty else "Produto Desconhecido"
        items_desc.append(f"{item['quantity']}x {prod_name}")

    itens_str = ", ".join(items_desc)

    return (
        f"Pedido #{order_id} - Cliente: {customer_name}\n"
        f"Status: {order_data['status']} (Rastreio: {order_data['tracking_code']})\n"
        f"Previsão de entrega: {order_data['estimated_delivery']}\n"
        f"Itens comprados: {itens_str}\n"
        f"Total pago: R$ {order_data['total_brl']} via {order_data['payment_method']}"
    )

@tool
def check_active_promotions() -> str:
    """
    Use esta ferramenta para listar quais produtos estão com promoção ativa hoje.
    Não requer argumentos.
    """
    if df_promotions is None:
        return "Erro: Banco de promoções indisponível."
        
    
    active_promos = df_promotions[df_promotions["is_active"] == 1]
    
    if active_promos.empty:
        return "Nenhuma promoção ativa no momento."

    result = "Promoções ativas hoje:\n"
    
    for _, promo in active_promos.iterrows():
        prod = df_products[df_products["product_id"] == promo["product_id"]]
        prod_name = prod.iloc[0]["name"] if not prod.empty else "Produto Desconhecido"
        result += f"- {prod_name}: {promo['discount_percent']}% OFF ({promo['description']})\n"

    return result

@tool
def find_order_by_email(email: str) -> str:
    """
    Use esta ferramenta para buscar os pedidos de um cliente usando o e-mail dele.
    Útil quando o cliente quer saber do pedido mas só informa o e-mail ou nome.
    Você deve passar o e-mail completo do cliente como argumento.
    """
    if df_customers is None or df_orders is None:
        return "Erro: Bancos de dados indisponíveis."
    
    
    customer = df_customers[df_customers["email"].str.lower() == email.lower()]
    
    if customer.empty:
        return f"Nenhum cadastro encontrado para o e-mail: {email}"
        
    customer_id = customer.iloc[0]["customer_id"]
    customer_name = customer.iloc[0]["name"]
    
    
    customer_orders = df_orders[df_orders["customer_id"] == customer_id]
    
    if customer_orders.empty:
        return f"O cliente {customer_name} ({email}) ainda não possui pedidos registrados."
        
    
    resposta = f"Pedidos encontrados para {customer_name}:\n"
    
    for _, order in customer_orders.iterrows():
        order_id = order['order_id']
        
        
        items = df_order_items[df_order_items["order_id"] == order_id]
        items_desc = []
        for _, item in items.iterrows():
            prod = df_products[df_products["product_id"] == item["product_id"]]
            prod_name = prod.iloc[0]["name"] if not prod.empty else "Produto Desconhecido"
            items_desc.append(f"{item['quantity']}x {prod_name}")
        itens_str = ", ".join(items_desc)
        
        resposta += (
            f"\n- Pedido #{order_id}:\n"
            f"  Status: {order['status']} (Rastreio: {order['tracking_code']})\n"
            f"  Previsão: {order['estimated_delivery']}\n"
            f"  Itens: {itens_str}\n"
            f"  Total: R$ {order['total_brl']} via {order['payment_method']}\n"
        )
        
    return resposta

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

def chat_interativo():
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.1)

    ferramentas = [consult_products, consult_policies, consult_order_status, check_active_promotions, find_order_by_email]

    memory = MemorySaver()
    agente = create_agent(llm, ferramentas, checkpointer=memory)
    
    system_prompt = (
        "Você é o assistente virtual da loja Empório da Música, em Campo Grande/MS. "
        "Seja amigável, prestativo e use emojis. Responda apenas com base nas ferramentas disponíveis. "
        "Se perguntarem algo fora do contexto da loja de instrumentos, recuse educadamente."
    )
    
    config = {"configurable": {"thread_id": "sessao_terminal_1"}}
    
    print("\n" + "="*60)
    print("🎸 BEM-VINDO AO CHAT DA EMPÓRIO DA MÚSICA 🎸")
    print("Digite sua pergunta ou 'sair' para encerrar o atendimento.")
    print("="*60 + "\n")

    primeira_interacao = True 

    while True:
        try:
            pergunta_cliente = input("👤 Você: ")
            
            if pergunta_cliente.lower() in ['sair', 'exit', 'quit', 'tchau']:
                print("\n🤖 Agente: Muito obrigado! Até logo e muita música para você! 🎶\n")
                break
            
            if not pergunta_cliente.strip():
                continue
                
            
            if primeira_interacao:
                mensagens = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=pergunta_cliente)
                ]
                primeira_interacao = False
            else:
                mensagens = [HumanMessage(content=pergunta_cliente)]

            resposta = agente.invoke(
                {"messages": mensagens},
                config=config
            )
            
            print(f"\n🤖 Agente: {resposta['messages'][-1].content}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Agente: Atendimento encerrado. Até mais! 🎸\n")
            break
        except Exception as e:
            print(f"\n❌ Ocorreu um erro na comunicação: {e}\n")

if __name__ == "__main__":
    chat_interativo()
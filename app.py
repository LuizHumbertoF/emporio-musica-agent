import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import warnings
import uuid

warnings.filterwarnings("ignore", category=DeprecationWarning)


from agent import (
    consult_products, 
    consult_policies, 
    consult_order_status, 
    check_active_promotions, 
    find_order_by_email
)

st.set_page_config(page_title="Empório da Música - IA", page_icon="🎸", layout="centered")

st.title("🎸 Assistente Empório da Música")
st.markdown("Bem-vindo! Tire suas dúvidas sobre produtos, pedidos e políticas da loja.")

@st.cache_resource
def inicializar_agente():
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.1)
    ferramentas = [consult_products, consult_policies, consult_order_status, check_active_promotions, find_order_by_email]
    memory = MemorySaver()
    agente = create_agent(llm, ferramentas, checkpointer=memory)
    return agente

agente = inicializar_agente()

system_prompt = (
    "Você é o assistente virtual da loja Empório da Música, em Campo Grande/MS. "
    "Seja amigável, prestativo e use emojis. Responda apenas com base nas ferramentas disponíveis. "
    "Se perguntarem algo fora do contexto da loja de instrumentos, recuse educadamente."
)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}


with st.sidebar:
    st.header("⚙️ Controles")
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.session_state.primeira_interacao = True
        st.session_state.thread_id = str(uuid.uuid4()) 
        st.rerun() 


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.primeira_interacao = True  

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pergunta_cliente = st.chat_input("Digite sua pergunta (ex: Onde está meu pedido?)...")

if pergunta_cliente:
    st.session_state.messages.append({"role": "user", "content": pergunta_cliente})
    with st.chat_message("user"):
        st.markdown(pergunta_cliente)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o sistema..."):
            try:
                
                if st.session_state.primeira_interacao:
                    mensagens = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=pergunta_cliente)
                    ]
                    st.session_state.primeira_interacao = False
                else:
                    mensagens = [HumanMessage(content=pergunta_cliente)]

                resposta = agente.invoke({"messages": mensagens}, config=config)
                resposta_final = resposta['messages'][-1].content
                
                st.markdown(resposta_final)
                st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
            except Exception as e:
                st.error(f"Erro na comunicação: {e}")
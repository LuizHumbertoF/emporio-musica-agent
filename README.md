# Desafio Técnico - Assistente de IA Empório da Música

Este repositório contém a solução para o desafio técnico da vaga de Engenharia de IA da Artefact. O projeto consiste em um Agente de Inteligência Artificial atuando como assistente virtual de uma loja de instrumentos musicais, capaz de cruzar dados estruturados (relacionais) e não-estruturados (documentos) de forma autônoma.

---

## 1. Instruções para Rodar o Projeto

### Pré-requisitos
* Python 3.10 ou superior.
* Uma chave de API gratuita da **Groq** (obtenha em [console.groq.com](https://console.groq.com)).

### Passo a Passo

1. **Clone o repositório e acesse a pasta:**
   ```bash
   git clone https://github.com/LuizHumbertoF/emporio-musica-agent.git
   cd emporio-musica-agent
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv venv
   # Linux/Mac:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Instale as dependências:**
   Para evitar o download pesado de bibliotecas CUDA caso não possua GPU, instale o PyTorch versão CPU primeiro, seguido das demais bibliotecas:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install streamlit langchain-groq langgraph pypdf faiss-cpu sentence-transformers langchain-community langchain-huggingface pandas
   ```

4. **Configure a variável de ambiente (Provedor do Modelo):**
   ```bash
   # Linux/Mac:
   export GROQ_API_KEY="gsk_sua_chave_aqui"
   # Windows (PowerShell):
   $env:GROQ_API_KEY="gsk_sua_chave_aqui"
   ```

5. **Execute a Interface Web:**
   ```bash
   streamlit run app.py
   ```
   O navegador abrirá automaticamente em `http://localhost:8501`. Na primeira execução, o sistema fará o download do modelo de embeddings local (aprox. 80MB).

---

## 2. Justificativa das Decisões Técnicas

* **Framework de Orquestração (LangGraph):** Optei pelo LangGraph em vez dos agentes tradicionais do LangChain. Ele oferece controle refinado sobre o fluxo cíclico (*ReAct*) e possui gerenciamento de estado persistente nativo (`MemorySaver`). Isso permite que a IA mantenha o histórico da conversa, lembrando do e-mail ou nome do usuário durante interações longas.
* **Provedor e LLM (Groq + Qwen 2.5 3.8B):** A API da Groq utiliza LPUs, garantindo latência quase zero (respostas ultrarrápidas). O modelo Qwen 3.8B demonstrou excelente capacidade de raciocínio lógico e *Function Calling* com baixo custo computacional.
* **Arquitetura de Retrieval (RAG Local):** Para ler o PDF de políticas, implementei um banco vetorial local (`FAISS`) utilizando embeddings locais da HuggingFace (`all-MiniLM-L6-v2`). Essa abordagem garante a privacidade dos documentos sensíveis da empresa e elimina custos com APIs de embeddings de terceiros.
* **Engenharia de Dados (Pandas):** Para os arquivos CSV, criei ferramentas em Python que utilizam Pandas para realizar *JOINs* dinâmicos em memória (ex: cruzar `customers` com `orders` e `products` a partir de um e-mail). Isso provê respostas determinísticas (sem alucinações matemáticas do LLM) de forma extremamente rápida.
* **Estratégia de Prompt (Guardrails):** O *System Prompt* foi desenhado para restringir o comportamento do agente estritamente ao escopo da loja, implementando um *guardrail* inicial que recusa solicitações de receitas, programação ou assuntos genéricos, protegendo a marca.
* **Frontend (Streamlit):** Escolhido pela velocidade de prototipação, permitindo entregar um produto final interativo, com histórico visual de chat e controle de sessão (`uuid`) para evitar estouro de limite de tokens da API.

---

### 3. Limitações Conhecidas e Melhorias Futuras

Com base no escopo e nos arquivos fornecidos para o desafio, aqui estão as principais limitações identificadas e o que poderia ser aprimorado com mais tempo:

1. **Limites de Tokens da API (Rate Limits):** O projeto utiliza a camada gratuita da API da Groq para rodar o modelo Qwen, que possui um limite estrito de 7.000 tokens por minuto. O `MemorySaver` do LangGraph armazena todo o histórico da conversa, o que faz com que interações longas atinjam esse limite rapidamente (gerando o erro `413 Rate Limit Exceeded`). A solução paliativa foi criar um botão para resetar a memória.
2. **Delay no Botão "Limpar Chat":** Na interface do Streamlit, ao clicar no botão de limpar o histórico, o sistema gera um novo ID de sessão e recarrega a aplicação para forçar a limpeza de memória do LangGraph. Esse processo apresenta um pequeno *delay* na tela, o que pode dar a falsa impressão de travamento ou mau funcionamento temporário antes de a tela ser efetivamente limpa.
3. **Otimização no Carregamento dos CSVs:** Atualmente, os 5 arquivos CSV são carregados diretamente na memória do script através do Pandas. Uma melhoria imediata e viável seria envelopar o carregamento dessas tabelas, evitando que os dados sejam relidos a cada nova interação e acelerando as respostas do agente.
4. **Refinamento da Leitura do PDF (RAG):** O particionamento do texto do PDF foi feito com parâmetros padrão (1000 caracteres). Com mais tempo, eu realizaria testes minuciosos de tamanho de *chunk* para garantir que o FAISS recupere trechos mais cirúrgicos, diminuindo o consumo de tokens na hora de consultar as políticas da loja.
---

## 4. Uso de Assistentes de Código (Workflow)

Durante o desenvolvimento deste desafio, utilizei o Google Gemini como suporte, adotando a metodologia de **Pair-Programming**:

* **Validação Arquitetural:** Utilizei a IA como parceiro de *brainstorming* para decidir qual seria a melhor abordagem de banco vetorial gratuito (chegando à conclusão do FAISS + HuggingFace).
* **Debugging Acelerado:** Ferramentas de IA foram essenciais para diagnosticar erros específicos de bibliotecas modernas, como o bloqueio `400 Bad Request` da Groq (que exigia um *HumanMessage* na primeira interação do LangGraph) e o timeout do Git ao lidar com arquivos grandes (`HTTP 408`).
* **Formatação:** Uso da IA para gerar estruturas repetitivas, como a formatação da interface do Streamlit e a base da documentação Markdown (exemplos, casos de uso e README), permitindo que meu tempo fosse investido 100% na lógica de negócio e nos relacionamentos de dados.
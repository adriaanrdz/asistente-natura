"""Módulo del agente RAG de Natura."""
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = Path("data")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "natura_v1_c800"


def cargar_documentos():
    """Carga los 3 PDFs con metadatos enriquecidos."""
    catalogo = {
        "01_Natura_Procesos_Internos.pdf": "procesos",
        "02_Natura_Manual_Tecnico.pdf": "manual_tecnico",
        "03_Natura_RRHH.pdf": "rrhh",
    }
    documentos = []
    for nombre, tipo in catalogo.items():
        ruta = DATA_DIR / nombre
        paginas = PyPDFLoader(str(ruta)).load()
        for p in paginas:
            p.metadata["tipo_documento"] = tipo
            p.metadata["nombre_documento"] = nombre
        documentos.extend(paginas)
    return documentos


def crear_vectorstore(documentos, embeddings):
    """Crea o recarga el vectorstore."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def construir_agente():
    """Construye y devuelve el agente completo con memoria."""
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Vectorstore: si existe lo reabre, si no lo crea
    if Path(CHROMA_DIR).exists():
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    else:
        documentos = cargar_documentos()
        vectorstore = crear_vectorstore(documentos, embeddings)
    
    # Retrievers filtrados con MMR
    def crear_retriever(tipo):
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 25,
                "lambda_mult": 0.7,
                "filter": {"tipo_documento": tipo},
            },
        )
    
    retriever_rrhh = crear_retriever("rrhh")
    retriever_tecnico = crear_retriever("manual_tecnico")
    retriever_procesos = crear_retriever("procesos")
    
    # Tools
    @tool
    def consultar_rrhh(pregunta: str) -> str:
        """Útil para preguntas sobre vacaciones, contratos, nóminas, 
        permisos, jornada laboral, formación, igualdad, equipamiento, 
        bajas médicas y accidentes laborales, teletrabajo, gastos y 
        dietas, beneficios sociales, antigüedad y código de conducta."""
        docs = retriever_rrhh.invoke(pregunta)
        if not docs:
            return "No se ha encontrado información relevante en el manual de RRHH."
        return "\n\n---\n\n".join(
            f"[RRHH, pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
    
    @tool
    def consultar_manual_tecnico(pregunta: str) -> str:
        """Útil para preguntas técnicas sobre plantación de árboles, 
        arbustos y aromáticas, instalación de césped natural y artificial, 
        sistemas de riego (goteo, aspersión, programadores), jardines 
        verticales, poda, fitosanitarios, mantenimiento estacional."""
        docs = retriever_tecnico.invoke(pregunta)
        if not docs:
            return "No se ha encontrado información relevante en el manual técnico."
        return "\n\n---\n\n".join(
            f"[Manual técnico, pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
    
    @tool
    def consultar_procesos(pregunta: str) -> str:
        """Útil para preguntas sobre procesos internos de Natura: proceso 
        comercial, presupuestos, planificación de obras, contratos de 
        mantenimiento, gestión de incidencias, garantías, facturación, 
        KPIs, estructura organizativa, política medioambiental."""
        docs = retriever_procesos.invoke(pregunta)
        if not docs:
            return "No se ha encontrado información relevante en el manual de procesos."
        return "\n\n---\n\n".join(
            f"[Procesos, pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
    
    tools = [consultar_rrhh, consultar_manual_tecnico, consultar_procesos]
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    llm_con_tools = llm.bind_tools(tools)
    
    # System prompt
    SYSTEM_PROMPT = SYSTEM_PROMPT = """Eres un asistente experto de Natura, una empresa española \
de jardinería y sistemas de riego con sede en Córdoba (Andalucía).

Tu misión es ayudar a empleados, clientes y colaboradores respondiendo preguntas \
sobre tres áreas: procesos internos de la empresa, trabajos técnicos de jardinería \
y riego, e información laboral y de RRHH.

REGLAS:
- Responde SIEMPRE basándote en la información que devuelvan las herramientas. No inventes datos.
- Si la primera búsqueda no encuentra la información, REFORMULA la pregunta usando \
sinónimos o términos relacionados, y vuelve a llamar a la herramienta. Por ejemplo, \
si buscas "días por antigüedad" y no encuentras nada, prueba con "beneficios por años \
de servicio", "premios por permanencia" o "bonificaciones para empleados antiguos".
- Solo después de 2-3 intentos con reformulaciones diferentes, concluye que la información no está.
- Cuando uses información de una herramienta, cita la fuente y la página.
- Si la pregunta no tiene relación con Natura, respóndela con tu conocimiento general.
- Responde siempre en español, con tono profesional pero cercano.
- Sé conciso y claro."""
    
    # Nodo agente
    def nodo_agente(state):
        mensajes = state["messages"]
        if not mensajes or mensajes[0].type != "system":
            mensajes = [("system", SYSTEM_PROMPT)] + list(mensajes)
        respuesta = llm_con_tools.invoke(mensajes)
        return {"messages": [respuesta]}
    
    # Grafo
    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", ToolNode(tools))
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges(
        "agente", tools_condition, {"tools": "tools", END: END}
    )
    grafo.add_edge("tools", "agente")
    
    return grafo.compile(checkpointer=InMemorySaver())


def extraer_texto(content):
    """Aplana el content de un mensaje (puede ser string o lista de bloques)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" or isinstance(b, str)
        )
    return str(content)
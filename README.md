# 🌿 Asistente Experto de Natura

Sistema **RAG con agentes inteligentes** para una empresa ficticia de jardinería 
y sistemas de riego con sede en Córdoba (Andalucía). Proyecto final del módulo 
de IA Generativa del Máster de Data Science.

El asistente responde consultas de empleados, clientes y colaboradores sobre 
tres áreas: procesos internos de la empresa, manual técnico de trabajos de 
jardinería y información de RRHH.

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────┐
│  Interfaz (Streamlit / Notebook)                │
│  ↓                                              │
│  Agente (LangGraph + Gemini)                    │
│  ↓ decide qué tool consultar                    │
│  ┌──────────────┬─────────────┬──────────────┐  │
│  │ Tool RRHH    │ Tool técn.  │ Tool proc.   │  │
│  └──────┬───────┴──────┬──────┴──────┬───────┘  │
│         ↓              ↓             ↓          │
│  Retrievers MMR filtrados por metadatos         │
│         ↓                                       │
│  ChromaDB (embeddings HuggingFace)              │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tecnologías

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.5 Flash Lite |
| Framework de agentes | LangGraph (grafo manual) |
| Orquestación | LangChain |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB persistente |
| Memoria | InMemorySaver con `thread_id` |
| Interfaz | Streamlit |
| Lectura PDF | PyPDFLoader |

---

## 📁 Estructura del proyecto

```
.
├── data/                              # PDFs del corpus
│   ├── 01_Natura_Procesos_Internos.pdf
│   ├── 02_Natura_Manual_Tecnico.pdf
│   └── 03_Natura_RRHH.pdf
├── chroma_db/                         # Base vectorial (se genera al ejecutar)
├── main.ipynb                     # Desarrollo paso a paso del proyecto
├── agente.py                          # Lógica del agente RAG
├── app.py                             # Interfaz Streamlit
├── requirements.txt                   # Dependencias
├── .env                               # API key (no se sube al repositorio)
├── .gitignore
└── README.md
```

---

## 🚀 Ejecución

### Requisitos previos

- Python 3.10 o superior
- Una API key de Google Gemini (gratuita en [aistudio.google.com](https://aistudio.google.com/apikey))

### Instalación

1. Clonar o descargar el repositorio:

```bash
   git clone https://github.com/TU_USUARIO/asistente-natura.git
   cd asistente-natura
```

2. Crear y activar un entorno virtual:

```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
```

3. Instalar dependencias:

```bash
   pip install -r requirements.txt
```

4. Crear un archivo `.env` en la raíz del proyecto con la API key:

```
   GOOGLE_API_KEY=tu_clave_aqui
```

### Ejecución del notebook

Para revisar el desarrollo paso a paso:

```bash
jupyter notebook main.ipynb
```

### Ejecución de la interfaz web

Para usar el asistente en una interfaz de chat:

```bash
streamlit run app.py
```
---

## 💬 Ejemplos de uso

El asistente responde a preguntas sobre los tres dominios cubiertos. Algunos ejemplos:

**Procesos internos:**
- *¿Qué garantía tiene una plantación?*
- *¿Cómo funciona el proceso comercial cuando llega un nuevo lead?*

**Manual técnico:**
- *¿Cómo se planta un olivo correctamente?*
- *¿Cómo se diseña un sistema de riego por goteo?*

**RRHH:**
- *¿Cuántos días de vacaciones tengo al año?*
- *¿Qué ocurre si tengo un accidente laboral?*

Gracias a la memoria conversacional, también es posible hacer preguntas de 
seguimiento sin repetir el contexto:

> Usuario: *¿Cuántos días de vacaciones tengo al año?*  
> Asistente: *Tienes derecho a 23 días laborables...*  
> Usuario: *¿Y si llevo más de 10 años?*  
> Asistente: *En ese caso, tienes 1 día extra al cumplir los 10 años...*

---

## 🧠 Decisiones de diseño destacables

- **Tres documentos especializados** en lugar de un único corpus, lo que permite 
  al agente decidir qué fuente consultar mediante tools temáticas.
- **Embeddings multilingües locales** (HuggingFace), gratuitos y sin dependencia 
  de cuotas externas para el desarrollo iterativo.
- **Grafo de LangGraph construido manualmente** (en lugar de `create_agent` 
  prefabricado) para hacer explícito el flujo del agente.
- **MMR con `lambda_mult=0.7`** ajustado empíricamente tras detectar que 
  valores más bajos descartaban chunks relevantes considerándolos demasiado 
  similares a otros ya seleccionados.
- **Memoria conversacional con `thread_id`**, que permite simular múltiples 
  conversaciones independientes.

---

## 🔮 Posibles mejoras futuras

- **Multi-Query Retrieval**: generar variantes automáticas de la pregunta 
  para mejorar la cobertura cuando la información está dispersa entre capítulos.
- **Reranking**: aplicar un modelo cross-encoder sobre los chunks recuperados 
  para reordenarlos antes de pasarlos al LLM.
- **Embeddings de Gemini** (`text-embedding-004`): podrían mejorar la calidad 
  semántica en español respecto a MiniLM.
- **Persistencia de memoria**: cambiar `InMemorySaver` por `SqliteSaver` para 
  conservar conversaciones entre sesiones.
- **Despliegue en Streamlit Cloud** para acceso público sin necesidad de 
  ejecución local.

---

## 👤 Autor

**Adrián Rodríguez Bejarano**  
Máster de Data Science — Evolve
10/05/2026

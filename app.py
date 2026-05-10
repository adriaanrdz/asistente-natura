"""Interfaz Streamlit para el agente RAG de Natura."""
import streamlit as st
import uuid
from agente import construir_agente, extraer_texto

# Configuración de la página
st.set_page_config(
    page_title="Asistente Natura",
    page_icon="🌿",
    layout="centered",
)

# Carga del agente (cacheada para que no se reconstruya en cada interacción)
@st.cache_resource(show_spinner="Cargando el asistente...")
def cargar_agente():
    return construir_agente()

agente = cargar_agente()

# Estado de la sesión: id de conversación e historial visible
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"chat_{uuid.uuid4().hex[:8]}"

if "mensajes_visibles" not in st.session_state:
    st.session_state.mensajes_visibles = []

# Cabecera
st.title("🌿 Asistente Natura")
st.caption("Pregúntame sobre procesos internos, trabajos técnicos o RRHH.")

# Sidebar con info y reset
with st.sidebar:
    st.subheader("ℹ️ Sobre el asistente")
    st.write(
        "Este asistente usa un sistema RAG con 3 manuales internos de Natura: "
        "procesos, manual técnico y RRHH. Responde basándose únicamente en "
        "la documentación oficial."
    )
    st.write(f"**Conversación actual:** `{st.session_state.thread_id}`")
    
    if st.button("🔄 Nueva conversación"):
        st.session_state.thread_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.mensajes_visibles = []
        st.rerun()
    
    st.divider()
    st.subheader("💡 Ejemplos de preguntas")
    st.write(
        "- ¿Cuántos días de vacaciones tengo?\n"
        "- ¿Cómo se planta un olivo?\n"
        "- ¿Qué garantía tienen las plantaciones?\n"
        "- ¿Qué tipos de mantenimiento ofrecen?\n"
        "- ¿Cómo configuro un riego por goteo?"
    )

# Mostrar el historial visible
for msg in st.session_state.mensajes_visibles:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

# Input del usuario
if pregunta := st.chat_input("Escribe tu pregunta..."):
    # Mostrar pregunta del usuario
    st.session_state.mensajes_visibles.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Llamar al agente
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = agente.invoke(
                {"messages": [("user", pregunta)]},
                config=config,
            )
            
            # Extraer la respuesta final
            mensaje_final = respuesta["messages"][-1]
            texto = extraer_texto(mensaje_final.content)
            
            st.markdown(texto)
            
            # Mostrar herramientas usadas (informativo)
            tools_usadas = []
            for m in respuesta["messages"]:
                if m.__class__.__name__ == "AIMessage" and m.tool_calls:
                    for tc in m.tool_calls:
                        if tc["name"] not in tools_usadas:
                            tools_usadas.append(tc["name"])
            
            if tools_usadas:
                with st.expander("🔧 Herramientas consultadas"):
                    for t in tools_usadas:
                        st.write(f"- `{t}`")
    
    # Guardar respuesta en historial
    st.session_state.mensajes_visibles.append({"rol": "assistant", "contenido": texto})
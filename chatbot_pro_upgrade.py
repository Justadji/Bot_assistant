from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

Mode = Literal["prof", "quiz", "correcteur"]

@dataclass
class SessionState:
    history: ChatMessageHistory
    summary: str
    mode: Mode

# ---- 1) Modèles ----
# Modèle principal (réponses)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Modèle “résumeur” (mémoire longue)
summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ---- 2) Prompts ----
SYSTEM_BY_MODE: Dict[Mode, str] = {
    "prof": (
        "Tu es un assistant pédagogique en statistiques/économétrie. "
        "Explique clairement, avec un exemple concret quand utile. "
        "Pose 1 question de vérification à la fin si pertinent."
    ),
    "quiz": (
        "Tu es un examinateur bienveillant en statistiques/économétrie. "
        "Tu poses des questions progressives (1 à la fois), "
        "tu attends la réponse de l'étudiant, puis tu corriges et tu notes sur 20."
    ),
    "correcteur": (
        "Tu es un correcteur rigoureux. "
        "L'étudiant te donne une solution/raisonnement : "
        "tu détectes les erreurs, tu expliques pourquoi, puis tu proposes une correction propre."
    ),
}

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "CONTEXTE RÉSUMÉ (mémoire longue) :\n{summary}\n\n"
     "MODE ACTUEL : {mode}\n"
     "{mode_instructions}"
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# ---- 3) Store sessions ----
_store: Dict[str, SessionState] = {}

def get_state(session_id: str) -> SessionState:
    if session_id not in _store:
        _store[session_id] = SessionState(history=ChatMessageHistory(), summary="", mode="prof")
    return _store[session_id]

def get_history(session_id: str) -> ChatMessageHistory:
    return get_state(session_id).history

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---- 4) Mémoire longue : résumé automatique ----
def update_summary_if_needed(session_id: str, max_messages: int = 16) -> None:
    """
    Si l'historique devient trop long, on le compresse dans un résumé,
    puis on repart avec un historique plus court.
    """
    state = get_state(session_id)
    messages = state.history.messages
    if len(messages) <= max_messages:
        return

    # On résume les messages les plus anciens et on garde les derniers
    keep_last = 8
    old = messages[:-keep_last]
    recent = messages[-keep_last:]

    old_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in old]
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu compresses une conversation en un résumé utile et fidèle.\n"
         "Objectif: garder faits, objectifs, définitions, décisions, notations, exemples clés.\n"
         "Réponse en français, style bullet points courts."),
        ("human",
         "Résumé existant (peut être vide):\n{prev}\n\n"
         "Nouveaux messages à intégrer:\n{chunk}\n\n"
         "Donne le nouveau résumé consolidé.")
    ])

    summary_chain = summary_prompt | summarizer_llm
    new_summary = summary_chain.invoke({"prev": state.summary, "chunk": old_text}).content

    state.summary = new_summary
    # On réinitialise l'historique avec seulement les derniers messages
    state.history = ChatMessageHistory()
    for m in recent:
        state.history.add_message(m)

    # IMPORTANT: mettre à jour le store avec le nouvel objet history
    _store[session_id] = state

def set_mode(session_id: str, mode: Mode) -> None:
    state = get_state(session_id)
    state.mode = mode
    _store[session_id] = state

def render_help() -> str:
    return (
        "Commandes:\n"
        "  /mode prof | quiz | correcteur\n"
        "  /summary      -> afficher le résumé mémoire longue\n"
        "  /reset        -> reset mémoire (historique + résumé)\n"
        "  /help         -> afficher l'aide\n"
        "  quit          -> quitter\n"
    )

def reset_session(session_id: str) -> None:
    _store[session_id] = SessionState(history=ChatMessageHistory(), summary="", mode="prof")

# ---- 5) Loop ----
print("🤖 Chatbot PRO+ prêt. Tape /help pour commandes. Tape 'quit' pour sortir.\n")
session_id = "njab"

while True:
    msg = input("Toi: ").strip()
    if not msg:
        continue

    if msg.lower() == "quit":
        print("👋 Bye!")
        break

    if msg.startswith("/help"):
        print(render_help())
        continue

    if msg.startswith("/mode"):
        parts = msg.split()
        if len(parts) == 2 and parts[1] in ("prof", "quiz", "correcteur"):
            set_mode(session_id, parts[1])  # type: ignore
            print(f"✅ Mode = {parts[1]}\n")
        else:
            print("❌ Utilise: /mode prof | quiz | correcteur\n")
        continue

    if msg.startswith("/summary"):
        state = get_state(session_id)
        print("🧠 Résumé (mémoire longue):\n", state.summary or "(vide)", "\n")
        continue

    if msg.startswith("/reset"):
        reset_session(session_id)
        print("✅ Session reset.\n")
        continue

    # Mise à jour résumé si nécessaire (avant d'ajouter + de messages)
    update_summary_if_needed(session_id)

    state = get_state(session_id)
    result = chat.invoke(
        {
            "input": msg,
            "summary": state.summary,
            "mode": state.mode,
            "mode_instructions": SYSTEM_BY_MODE[state.mode],
        },
        config={"configurable": {"session_id": session_id}}
    )

    print("Bot:", result.content, "\n")
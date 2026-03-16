from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# 1) Modèle
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 2) Prompt (avec un emplacement pour l'historique)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant pédagogique spécialisé en statistiques/économétrie. "
     "Tu expliques clairement, avec exemples, et tu poses 1 question de vérification quand c'est utile."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 3) LCEL chain (prompt -> modèle)
chain = prompt | llm

# 4) Stockage en mémoire (par session)
_store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

# 5) Wrapper mémoire (pro)
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("🤖 Chatbot PRO prêt. Tape 'quit' pour sortir.\n")

session_id = "njab"  # tu peux mettre ton prénom, ou un id utilisateur

while True:
    msg = input("Toi: ").strip()
    if msg.lower() == "quit":
        print("👋 Bye!")
        break

    result = chat.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )

    print("Bot:", result.content, "\n")
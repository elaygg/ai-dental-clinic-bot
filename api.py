from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a polite and professional administrator of the Apostol Dental Clinic. 
Your task is to answer the patients' questions using ONLY the context provided below.

Rules of communication:
1. Write briefly, to the point, and in a friendly manner. 
2. Never use technical terms like "CTA", "Context" or "Knowledge Base" in your answers.
3. If there are prices in the context, name them clearly.
4. At the end of each answer, BE SURE to ask one clarifying question to help the client sign up. 

Examples of good endings:
- Can I suggest free time for you to make an appointment?
- Would you like me to book a consultation time for you?
- Would you like to make an appointment for the procedure right now?

The context:
{context}

---

Patient's question: {question}
Answer:
"""


app = FastAPI(
    title="AI Business Assistant API", 
    description="API for a smart RAG bot based on client documents"
)

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_ai_assistant(request: UserRequest):

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(request.question, k=3)

    if len(results) == 0:
        return {"answer": "Sorry, I didn't find the answer in the knowledge base.", "sources": []}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request.question)

    model = ChatOpenAI(model="gpt-3.5-turbo")
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]

    return {
        "answer": response_text,
        "sources": sources
    }
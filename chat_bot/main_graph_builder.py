import os, io, re, uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import Tool
import vercelpy.blob_store as blob_store
import json

global pdf_id
def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama3-8b-8192")

def load_pdf(pdf_bytes: bytes):
    from PyPDF2 import PdfReader
    from langchain.docstore.document import Document
    global pdf_id
    pdf_id = str(uuid.uuid4())
    reader = PdfReader(pdf_bytes)
    return [Document(page_content=p.extract_text(), metadata={"page": i+1, "pdf_id": pdf_id}) for i, p in enumerate(reader.pages)]



def extract_content(msg):
    from langchain_core.messages import BaseMessage

    if isinstance(msg, BaseMessage):
        return msg.content or ""

    elif isinstance(msg, dict):
        return msg.get("content", "") or ""

    elif isinstance(msg, str):
        # Try to decode a stringified JSON list
        try:
            parsed = json.loads(msg)
            if isinstance(parsed, list):
                return "\n\n".join(
                    f"🔹 {item.get('title', '').strip()}\n{item.get('content', '').strip()}\n🔗 {item.get('url', '').strip()}"
                    for item in parsed if isinstance(item, dict)
                )
            else:
                return msg.strip()
        except json.JSONDecodeError:
            return msg.strip()

    elif isinstance(msg, list):
        return "\n\n".join(
            f"🔹 {item.get('title', '').strip()}\n{item.get('content', '').strip()}\n🔗 {item.get('url', '').strip()}"
            for item in msg if isinstance(item, dict)
        )

    return str(msg)


def upload_to_qdrant(documents, collection_name):
    from langchain_community.vectorstores import Qdrant
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from qdrant_client.models import VectorParams, Distance

    qdrant_client = get_qdrant_client()
    embeddings = get_embeddings()

    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    qdrant = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)
    qdrant.add_documents(docs)

def extract_questions(documents):
    all_questions = []
    pattern = re.compile(r'(?i)(Q\d+\.|^\d+\.)\s*(.*?)(?=(Q\d+\.|^\d+\.)|$)', re.DOTALL | re.MULTILINE)
    for doc in documents:
        for match in pattern.findall(doc.page_content):
            question_text = match[1].strip()
            marks_match = re.search(r'\[(\d+)\s*(marks?)?\]', question_text, re.IGNORECASE)
            marks = int(marks_match.group(1)) if marks_match else None
            all_questions.append({"question": question_text, "marks": marks, **doc.metadata})
    return all_questions



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic assistant."),
    ("user", """
Use retrieval_answer (~80%) and tavily_answer (~20%) to generate structured answers.
Adjust depth based on marks: short (1–2), medium (3–5), detailed (>5). Include code if relevant.
Format:
---
**Question:** {question}
**Marks:** {marks}
**Retrieval-based Answer:** {retrieval_answer}
**Tavily Answer:** {tavily_answer}
**Combined Answer:**
- Introduction
- Diagram
- Key Points
- Explanation
- Code Example
- Conclusion
---
""")
])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    marks: int

def run_llm_with_tools(state: State):
    from langchain_community.tools.tavily_search import TavilySearchResults
    embeddings = get_embeddings()
    retriever = get_retriever(embeddings)
    llm = get_groq_llm()
    tools = [
        Tool.from_function(retriever.invoke, name="RAG", description="Answer questions from PDF."),
        Tool.from_function(TavilySearchResults().invoke, name="TavilySearch", description="Web search tool.")
    ]
    llm_with_tools = llm.bind_tools(tools)

    # Use .generate() to avoid structured tool call format
    result = llm_with_tools.generate([state["messages"]])
    message = result.generations[0][0].message  # Extract the message
    return {"messages": state["messages"] + [message]}

def get_retriever(embeddings):
    from langchain_community.vectorstores import Qdrant
    return Qdrant(client=get_qdrant_client(), collection_name="pdf_documents", embeddings=embeddings).as_retriever()

def ToolExecutor(state: State):
    from langchain_community.tools.tavily_search import TavilySearchResults
    retriever = get_retriever(get_embeddings())
    rag_tool = Tool.from_function(retriever.invoke, name="RAG", description="PDF Tool")
    tavily_tool = Tool.from_function(TavilySearchResults().invoke, name="Tavily", description="Web Search")

    query = state["messages"][-2].content
    results = []
    try:
        rag = rag_tool.invoke(query)
        rag_content = "\n\n".join([doc.page_content for doc in rag])
        results.append({"role": "assistant", "content": rag_content})
    except Exception as e:
        print("RAG Error:", e)
    try:
        tavily = tavily_tool.invoke(query)
        results.append({"role": "assistant", "content": tavily})
    except Exception as e:
        print("Tavily Error:", e)
    return {"messages": state["messages"] + results}

def format_answer(state: State):
    llm = get_llm()
    chain = prompt | llm

    question = extract_content(state["messages"][0])
    retrieval_answer = extract_content(state["messages"][-2])
    tavily_answer = extract_content(state["messages"][-1])
    marks = state["marks"] if isinstance(state["marks"], int) else extract_content(state["marks"])

    input_data = {
        "question": question,
        "marks": marks,
        "retrieval_answer": retrieval_answer,
        "tavily_answer": tavily_answer
    }
    result = chain.invoke(input_data)
    return {"messages": state["messages"] + [result]}


Builder = StateGraph(State)
Builder.add_node("LLM_with_tools", run_llm_with_tools)
Builder.add_node("Tools", ToolExecutor)
Builder.add_node("format_answer", format_answer)
Builder.add_edge(START, "LLM_with_tools")
Builder.add_edge("LLM_with_tools", "Tools")
Builder.add_edge("Tools", "format_answer")
Builder.add_edge("format_answer", END)
graph = Builder.compile()

class StateGraphExecutor(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    Ans: Annotated[list[AnyMessage], add_messages]
    Referal: bytes
    QuePdf: bytes
    collection_name: str
    FinalPdf: str
    pdf_id: str

def srart_graph(state: StateGraphExecutor):
    import time
    questions = state["messages"][-1].content
    batch_inputs = [
        {"messages": [{"role": "user", "content": q["question"]}], "marks": str(q.get("marks"))}
        for q in questions
    ]

    all_answers = []
    batch_size = 5  # You can adjust based on memory

    for i in range(0, len(batch_inputs), batch_size):
        batch = batch_inputs[i:i + batch_size]
        print(f"⚙️ Batch {i // batch_size + 1}...", flush=True)
        
        for item in batch:
            try:
                result = graph.invoke(item)
                all_answers.append({"role": "assistant", "content": result["messages"][-1].content})
            except Exception as e:
                import traceback
                print("❌ Error during graph.invoke:", e)
                traceback.print_exc()


        time.sleep(0.5)  # optional cooldown to prevent worker timeout

    return {"Ans": state["Ans"] + all_answers}


def call_pdf_genrater(state):
    import markdown
    from xhtml2pdf import pisa
    questions = state["messages"][0].content
    answers = state["Ans"]
    markdown_content = "# 📜 Question-Answer Report\n\n"
    for i, a in enumerate(answers):
        q_text = questions[i].get("question")
        a_text = a.content if hasattr(a, 'content') else a
        markdown_content += f"### Q{i+1}: {q_text}\n**A{i+1}:**\n{a_text}\n\n"
    html = markdown.markdown(markdown_content)
    buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=buffer)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    resp = blob_store.put("PDF_Q&A/report_session.pdf", pdf_bytes, {"contentType": "application/pdf", "access": "public", "allowOverwrite": True})
    return {**state, "FinalPdf": resp["url"]}

def Referal_PDF_to_Qdrant(state: StateGraphExecutor):
    docs = load_pdf(state["Referal"])
    pdf_id= docs[0].metadata.get("pdf_id")
    upload_to_qdrant(docs, state["collection_name"])
    return {**state, "pdf_id": pdf_id}


def qus_loading(state: StateGraphExecutor):
    docs = load_pdf(state["QuePdf"])
    all_Ques = extract_questions(docs)
    return {"messages": state["messages"] + [{"role": "assistant", "content": all_Ques}]}

def trigger_send_email_task(state: StateGraphExecutor):
    from .email_tasks import send_email_task
    send_email_task(state["FinalPdf"])

main_builder = StateGraph(StateGraphExecutor)
main_builder.add_node("Referal_PDF_to_Qdrant", Referal_PDF_to_Qdrant)
main_builder.add_node("qus_loading", qus_loading)
main_builder.add_node("graph", srart_graph)
main_builder.add_node("pdf", call_pdf_genrater)
main_builder.add_node("send_mail", trigger_send_email_task)
main_builder.add_edge(START, "qus_loading")
main_builder.add_edge("qus_loading", "graph")
main_builder.add_edge("graph", "pdf")
main_builder.add_edge("pdf", "send_mail")
main_builder.add_edge("send_mail", END)

main_graph = main_builder.compile()

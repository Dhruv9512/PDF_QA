# ===================== Imports =====================
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance
from langchain.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os, io, re, uuid, markdown
from typing import Annotated
from typing_extensions import TypedDict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from xhtml2pdf import pisa
import vercelpy.blob_store as blob_store
from .email_tasks import send_email_task


# ===================== Environment Setup =====================
load_dotenv()

# ===================== Qdrant Initialization =====================
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "pdf_documents"


# ===================== Model Initialization =====================
# ollama_llm = OllamaLLM(model="llama3.1")
ollama_llm =ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)
# ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
Groq_llm = ChatGroq(model="llama3-8b-8192")

# ===================== Global =====================
pdf_id = None

# ===================== Utilities =====================
def load_pdf(pdf_bytes: bytes):
    global pdf_id
    pdf_id = str(uuid.uuid4())
    reader = PdfReader(pdf_bytes)
    documents = [Document(page_content=p.extract_text(), metadata={"page": i+1, "pdf_id": pdf_id}) for i, p in enumerate(reader.pages)]
    return documents

def upload_to_qdrant(documents, collection_name):
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    qdrant = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=ollama_embeddings)
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

# ===================== Prompt Template =====================
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful academic assistant. Use retrieval_answer (~80%) and tavily_answer (~20%) to generate structured answers.
Adjust depth based on marks: short (1–2), medium (3–5), detailed (>5). Include code if relevant.
Format:
---
**Question:** {question}
**Marks:** {marks}
**Retrieval-based Answer:**  {retrieval_answer}
**Tavily Answer:**  {tavily_answer}
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

# ===================== Tools Setup =====================
retriever = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=ollama_embeddings).as_retriever()
rag_tool = Tool.from_function(retriever.invoke, name="RAG", description="Answer questions from PDF.")
tavily_tool = Tool.from_function(TavilySearchResults().invoke, name="TavilySearch", description="Web search tool.")
tools = [rag_tool, tavily_tool]
llm_with_tools = Groq_llm.bind_tools(tools)
llm_chain = prompt | ollama_llm

# ===================== Graph Definitions =====================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    marks: int

def run_llm_with_tools(state: State):
    return {"messages": state["messages"] + [llm_with_tools.invoke(state["messages"]) ]}

def ToolExecutor(state: State):
    query = state["messages"][-2].content
    ans = []
    try:
        rag = rag_tool.invoke(query)
        rag_content = "\n\n".join([doc.page_content for doc in rag])
        ans.append({"role": "assistant", "content": rag_content})
    except Exception as e:
        print("RAG Error:", e)
    try:
        tavily = tavily_tool.invoke(query)
        ans.append({"role": "assistant", "content": tavily})
    except Exception as e:
        print("Tavily Error:", e)
    return {"messages": state["messages"] + ans}

def format_answer(state: State):
    input_data = {
        "question": state["messages"][0].content,
        "marks": state["marks"],
        "retrieval_answer": state["messages"][-2].content,
        "tavily_answer": state["messages"][-1].content
    }
    result = llm_chain.invoke(input_data)
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

# ===================== Main Flow =====================
class StateGraphExecutor(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    Ans: Annotated[list[AnyMessage], add_messages]
    Referal: bytes
    QuePdf: bytes
    collection_name: str
    FinalPdf: str

def srart_graph(state: StateGraphExecutor):
    questions = state["messages"][-1].content
    all_answers = []
    for q in questions:
        result = graph.invoke({"messages": [{"role": "user", "content": q["question"]}], "marks": str(q.get("marks"))})
        all_answers.append({"role": "assistant", "content": result["messages"][-1].content})
    return {"Ans": state["Ans"] + all_answers}

def call_pdf_genrater(state):
    questions = state["messages"][0].content
    answers = state["Ans"]
    markdown_content = "# 📝 Question-Answer Report\n\n"
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
    upload_to_qdrant(docs, state["collection_name"])

def qus_loading(state: StateGraphExecutor):
    docs = load_pdf(state["QuePdf"])
    all_Ques = extract_questions(docs)
    return {"messages": state["messages"] + [{"role": "assistant", "content": all_Ques}]}

def trigger_send_email_task(state: StateGraphExecutor):
    print("PDF URL to send:", state["FinalPdf"])
    # send_email_task.apply_async(args=[str(state["FinalPdf"])])
    send_email_task(state["FinalPdf"])

# ===================== Main Graph =====================
main_builder = StateGraph(StateGraphExecutor)
main_builder.add_node("Referal_PDF_to_Qdrant", Referal_PDF_to_Qdrant)
main_builder.add_node("qus_loading", qus_loading)
main_builder.add_node("graph", srart_graph)
main_builder.add_node("pdf", call_pdf_genrater)
main_builder.add_node("send_mail", trigger_send_email_task)
main_builder.add_edge(START, "qus_loading")
# main_builder.add_edge(START, "Referal_PDF_to_Qdrant")
main_builder.add_edge("qus_loading", "graph")
main_builder.add_edge("graph", "pdf")
main_builder.add_edge("pdf", "send_mail")
main_builder.add_edge("send_mail", END)
main_graph = main_builder.compile()

# ===================== Django APIView =====================
@method_decorator(csrf_exempt, name='dispatch')
class pdf(APIView):
    def post(self, request):
        # Referal = request.data.get("Referal")
        QuePdf = request.data.get("QuePdf")
        input_grapg = {
            # "Referal": io.BytesIO(Referal),
            "QuePdf": QuePdf,
            "collection_name": collection_name,
            "Ans": [],
            "messages": []
        }
        try:
            main_graph.invoke(input_grapg)
            return Response({"message": "Process started", "status": "success", "pdf_id": pdf_id}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
import html
import io
import threading
import time
import os,re, uuid,logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
import vercelpy.blob_store as blob_store
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    from langchain_qdrant import QdrantVectorStore
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
    qdrant = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
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
    ("system", "You are a clear, student‑friendly academic assistant."),
    ("user", """
---

ROLE:
You are an expert educator who tailors explanations clearly for college students, whether concept-based or programming questions. If the question asks for a comparative (“difference between X and Y”), include a comparison table.

INPUT:
Question: {question}
Marks: {marks}
Retrieval Answer (core: ~80%): {retrieval_answer}
Tavily Answer (supplement: ~20%): {tavily_answer}

INSTRUCTIONS:
1. Create a **Combined Answer** using ~80% from retrieval and ~20% from Tavily; do *not* display raw sources.
2. If the question is:
   - **Conceptual** (no code): Use this structure:
     **🟢 Introduction**  
     **🟢 Diagram** (one sentence or “No diagram needed”)  
     **🟢Key Points**: Provide **6–8 very concise bullet headings**, each 2–3 words maximum.
     **🟢 Explanation** (3–4 sentences per bullet)  
     **🟢 Conclusion**
   - **Coding**: Use this structure:
     **🟢 Introduction**  
     **🟢 Diagram** (if useful)  
     **🟢 Key Points** (3–6 bullets)  
     **🟢 Explanation** (1–2 sentences per bullet)  
     **🟢 Full Code/Program**  
     **🟢 Code Explanation**  
     **🟢 Conclusion**
   - **Difference** question (“difference between X and Y”): After the above, include:
     **🟢 Comparison Table** with columns: No./ Feature / x / y, and ~8–9 rows covering key aspects.
3. Style: friendly, clear, concise. Avoid jargon or briefly explain it.
4. Length guidance:
   - Introduction: 2–3 sentences  
   - Key Points: 3–6 bullets  
   - Explanation: 1–2 sentences per point  
   - Code Explanation: short and clear  
   - Table: ~8–9 compare rows, no prose descriptions
5. **Only output the Combined Answer**—no question, no marks, no raw source text, no prompts.

---

Provide the Combined Answer now.
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
    from langchain_qdrant import QdrantVectorStore
    return QdrantVectorStore(client=get_qdrant_client(), collection_name="pdf_documents", embedding=embeddings).as_retriever(   search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.5
        })

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

def srart_graph(state):
    try:
        logger.info("🤖 Starting answer generation graph...")
        questions = state["messages"][-1].content
        batch_inputs = [
            {"messages": [{"role": "user", "content": q["question"]}], "marks": str(q.get("marks"))}
            for q in questions if q.get("question") and str(q.get("question")).strip()
        ]
        logger.info(f"✅ Prepared {len(batch_inputs)} batch inputs.")

        all_answers = []
        batch_size = 2
        for i in range(0, len(batch_inputs), batch_size):
            logger.info(f"⚙️ Running batch {i // batch_size + 1}...")
            batch = batch_inputs[i:i + batch_size]

            for item in batch:
                user_msg = item["messages"][0].get("content", "")
                if not user_msg.strip():
                    logger.warning("⚠️ Skipping empty question.")
                    continue

                result_container = {}

                def run_graph():
                    try:
                        result_container["result"] = graph.invoke(item)
                    except Exception as e:
                        result_container["error"] = e

                t = threading.Thread(target=run_graph)
                t.start()
                t.join(timeout=60)

                if t.is_alive():
                    logger.error("❌ Timeout during graph.invoke")
                    continue
                if "error" in result_container:
                    logger.exception("❌ Error during graph.invoke")
                    continue

                result = result_container["result"]
                all_answers.append({"role": "assistant", "content": result["messages"][-1].content})

            time.sleep(0.2)

        logger.info(f"✅ Generated answers for {len(all_answers)} questions.")
        return {"Ans": state["Ans"] + all_answers}

    except Exception as e:
        logger.exception("❌ Failed in srart_graph")
        raise

def call_pdf_genrater(state):
    try:
        logger.info("📄 Generating PDF from Q&A...")
        questions = [msg["question"] for msg in state["messages"][0].content]
        answers = state["Ans"]

        buffer = io.BytesIO()
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=1*inch, leftMargin=1*inch,
                                topMargin=1*inch, bottomMargin=1*inch,
                                encoding='utf-8')

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Question", fontSize=12, leading=16, spaceAfter=6, fontName="Helvetica-Bold"))
        styles.add(ParagraphStyle(name="Answer", fontSize=11, leading=14, spaceAfter=12, fontName="Helvetica"))

        flowables = []
        flowables.append(Paragraph("<b>📜 Question-Answer Report</b>", styles["Title"]))
        flowables.append(Spacer(1, 12))

        for i, a in enumerate(answers):
            q_text = questions[i] if i < len(questions) else f"Question {i+1}"
            a_text = a.content if hasattr(a, 'content') else a
            flowables.append(Paragraph(f"<b>Q{i+1}:</b> {html.escape(q_text)}", styles["Question"]))
            for para in a_text.strip().split("\n\n"):
                flowables.append(Paragraph(html.escape(para.strip()), styles["Answer"]))

        flowables.append(Spacer(1, 24))
        flowables.append(Paragraph("<font size='9'><i>Tip: Download and open in WPS or Adobe Reader for best results.</i></font>", styles["Normal"]))

        doc.build(flowables)
        buffer.seek(0)
        pdf_bytes = buffer.read()

        logger.info("📤 Uploading PDF to Vercel Blob...")
        filename = f"report_{uuid.uuid4().hex}.pdf"
        resp = blob_store.put(filename, pdf_bytes, {
            "contentType": "application/pdf",
            "access": "public",
            "allowOverwrite": True
        })

        logger.info(f"✅ PDF uploaded to: {resp['url']}")
        return {**state, "FinalPdf": resp["url"]}
    except Exception as e:
        logger.exception("❌ Failed in call_pdf_genrater")
        raise
def Referal_PDF_to_Qdrant(state):
    try:
        logger.info("📥 Loading referral PDF and uploading to Qdrant...")
        docs = load_pdf(state["Referal"])
        pdf_id = docs[0].metadata.get("pdf_id")
        upload_to_qdrant(docs, state["collection_name"])
        logger.info(f"✅ PDF uploaded to Qdrant with pdf_id: {pdf_id}")
        return {**state, "pdf_id": pdf_id}
    except Exception as e:
        logger.exception("❌ Failed in Referal_PDF_to_Qdrant")
        raise


def qus_loading(state):
    try:
        logger.info("🔍 Loading questions from question PDF...")
        docs = load_pdf(state["QuePdf"])
        all_Ques = extract_questions(docs)
        logger.info(f"✅ Extracted {len(all_Ques)} questions.")
        return {"messages": state["messages"] + [{"role": "assistant", "content": all_Ques}]}
    except Exception as e:
        logger.exception("❌ Failed in qus_loading")
        raise

def trigger_send_email_task(state):
    try:
        logger.info("📧 Sending email task...")
        from .email_tasks import send_email_task
        send_email_task.apply_async(args=[state["FinalPdf"]])
        logger.info("✅ Email task triggered.")
    except Exception as e:
        logger.exception("❌ Failed in trigger_send_email_task")
        raise

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

import os
import base64
import sqlite3
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
llm = ChatOpenAI(model="gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INDEX_NAME = os.getenv("INDEX_NAME")

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('haqqi.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_msg TEXT, bot_msg TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_chat(user_msg, bot_msg):
    conn = sqlite3.connect('haqqi.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_msg, bot_msg) VALUES (?, ?)", (user_msg, bot_msg))
    conn.commit()
    conn.close()

# --- API Endpoints ---
@app.get("/history")
async def get_history():
    conn = sqlite3.connect('haqqi.db')
    c = conn.cursor()
    c.execute("SELECT user_msg, bot_msg FROM chat_history ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return {"history": [{"user": r[0], "bot": r[1]} for r in rows]}

def auto_fetch_from_boe(query):
    try:
        search_url = f"https://www.boe.es/buscar/ayudas/legislacion_xml.php?txt={query}"
        search_res = requests.get(search_url)
        if search_res.status_code == 200:
            root = ET.fromstring(search_res.content)
            first_id = root.find(".//id")
            if first_id is not None:
                boe_id = first_id.text
                api_url = f"https://www.boe.es/diario_boe/xml.php?id={boe_id}"
                doc_res = requests.get(api_url)
                doc_root = ET.fromstring(doc_res.content)
                text_elements = doc_root.findall(".//texto")
                return "\n".join([t.text for t in text_elements if t.text]), boe_id
    except Exception as e:
        print(f"Error: {e}")
    return None, None

@app.post("/auto_train")
async def auto_train(topic: str = Form(...)):
    text, boe_id = auto_fetch_from_boe(topic)
    if not text:
        raise HTTPException(status_code=400, detail="لم يتم العثور على تشريعات")
    doc = Document(page_content=text[:15000], metadata={"source": f"BOE:{boe_id}"})
    PineconeVectorStore.from_documents([doc], embedding=embeddings, index_name=INDEX_NAME)
    return {"status": "success", "message": f"تم دمج قانون {boe_id} بنجاح"}

@app.post("/analyze_doc")
async def analyze_doc(file: UploadFile = File(...)):
    contents = await file.read()
    encoded_img = base64.b64encode(contents).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت محامي إسباني خبير. حلل الوثيقة المرفقة بالدارجة المغربية."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}]}
        ]
    )
    answer = response.choices[0].message.content
    log_chat("[وثيقة مرفقة]", answer)
    return {"answer": answer}

@app.get("/ask")
async def ask(question: str):
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"أجب بالدارجة بناءً على هذا النص القانوني من BOE:\n{context}\n\nالسؤال: {question}"
    res = llm.invoke(prompt)
    
    log_chat(question, res.content)
    return {"answer": res.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

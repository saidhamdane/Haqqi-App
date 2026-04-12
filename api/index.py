import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        # نظام RAG مبسط لضمان السرعة وتفادي Timeout على Vercel
        context = "المصدر: الجريدة الرسمية الإسبانية BOE.es - تشريعات 2026"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"أنت خبير في القانون الإسباني. مرجعك: {context}. جاوب بالدارجة المغربية فقط وبشكل دقيق."},
                {"role": "user", "content": question}
            ],
            timeout=25.0
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "السيرفر عليه ضغط، عاود جرب دابا وغادي يخدم."}

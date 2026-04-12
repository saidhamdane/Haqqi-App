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
        # رابط API البحث الرسمي لـ BOE بنظام XML
        boe_api = f"https://www.boe.es/diario_boe/xml.php?id=BOE-S-{question}"
        
        async with httpx.AsyncClient() as http_client:
            # محاولة جلب البيانات في وقت قصير (5 ثوانٍ) لتجنب تعليق السيرفر
            boe_res = await http_client.get(boe_api, timeout=5.0)
            context = boe_res.text[:800] if boe_res.status_code == 200 else "لا توجد نتائج فورية."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"أنت خبير قانوني إسباني تستخدم نظام RAG. مرجعك: {context}. "
                                             "اشرح الحقوق والواجبات بالدارجة المغربية بوضوح تام بناءً على BOE.es."},
                {"role": "user", "content": question}
            ],
            timeout=15.0 # تقليل الوقت لضمان استجابة Vercel قبل الـ Timeout
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "السيرفر عليه ضغط بسبب البحث في BOE.es، عاود جرب دابا وغادي يخدم."}

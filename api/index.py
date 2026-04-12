# import os
import httpx
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def fetch_boe_data(query: str):
    """جلب أحدث القوانين مباشرة من BOE.es باستخدام API البحث الخاص بهم"""
    try:
        # رابط البحث الرسمي في الجريدة الرسمية الإسبانية
        search_url = f"https://www.boe.es/diario_boe/xml.php?id=BOE-S-{query}"
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(search_url, timeout=15.0)
            if response.status_code == 200:
                # تبسيط: استرجاع أول 500 حرف من السياق القانوني
                return response.text[:1000] 
        return "لا توجد نتائج مباشرة حالياً في BOE."
    except:
        return "تعذر الاتصال بـ API BOE."

@app.get("/api/index/ask")
async def ask(question: str):
    # 1. مرحلة الاسترجاع (Retrieval) من BOE
    legal_context = await fetch_boe_data(question)
    
    try:
        # 2. مرحلة التوليد (Generation) باستخدام السياق المسترجع
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"أنت مستشار قانوني إسباني مرتبط بـ BOE.es. السياق القانوني الحالي: {legal_context}. "
                                             "جاوب بالدارجة المغربية فقط وبدقة قانونية عالية."},
                {"role": "user", "content": question}
            ],
            timeout=30.0
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "السيرفر تقيل شوية حيتاش كايجيب المعلومات من BOE.es دابا، عاود جرب."}


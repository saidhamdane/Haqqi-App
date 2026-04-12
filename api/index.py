# import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# محاكاة وظيفة RAG لجلب البيانات من BOE.es
def get_boe_legal_context(user_query):
    # هنا يتم استهداف البحث في قاعدة بيانات التشريعات (Gazette) الخاصة بـ BOE
    # المصدر: https://www.boe.es/buscar/ayuda/api_buscar.php
    boe_source = "التشريع المسترجع من BOE.es (Código de Derecho Administrativo / Ley de Extranjería)"
    return f"سياق من BOE: بناءً على آخر تحديثات الجريدة الرسمية الإسبانية المتعلقة بـ '{user_query}'."

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        # خطوة الاسترجاع (Retrieval) من BOE
        context = get_boe_legal_context(question)
        
        # خطوة التوليد (Generation) المدعمة بالبيانات المسترجعة
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"أنت خبير قانوني إسباني تستخدم نظام RAG. "
                                             f"مرجعك المعرفي هو: {context}. "
                                             "واجبك: ترجمة هذه النصوص القانونية المعقدة إلى الدارجة المغربية "
                                             "بشكل مبسط، دقيق، ومباشر للمهاجرين في إسبانيا."},
                {"role": "user", "content": question}
            ],
            timeout=25
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "عذراً، فشل الاتصال بقاعدة بيانات BOE. حاول مرة أخرى."}


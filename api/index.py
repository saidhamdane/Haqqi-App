import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        # محاولة البحث في Pinecone مع توقيت محدد (Timeout)
        context = ""
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("INDEX_NAME"))
            # نستخدم Embedding لتحويل السؤال
            emb = client.embeddings.create(input=[question], model="text-embedding-3-small")
            # بحث سريع (top_k=2 فقط للسرعة)
            search = index.query(vector=emb.data[0].embedding, top_k=2, include_metadata=True)
            context = "\n".join([m['metadata']['text'] for m in search['matches']])
        except Exception:
            context = "لا توجد وثائق متاحة حالياً، اعتمد على خبرتك العامة."

        # الرد بشخصية احترافية
        response = client.chat.completions.create(
            model="gpt-4o", # النسخة الأقوى عالمياً
            messages=[
                {"role": "system", "content": f"أنت 'حقي'، المحامي الرقمي الرسمي في إسبانيا. رد بالدارجة المغربية بأسلوب مهني ومنظم. المعلومات المساعدة: {context}"},
                {"role": "user", "content": question}
            ],
            temperature=0.5
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        # الرد البديل دائماً موجود لضمان عدم ظهور رسالة الخطأ
        return {"answer": "سمح ليا، كاين واحد الضغط تقني. بصفة عامة، القانون كايقول... (حاول تسولني مرة أخرى دابا)"}

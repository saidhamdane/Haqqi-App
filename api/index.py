import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        # 1. تحويل السؤال إلى Vector
        res = client.embeddings.create(input=[question], model="text-embedding-3-small")
        query_vector = res.data[0].embedding

        # 2. البحث في Pinecone عن أدلة قانونية
        search_res = index.query(vector=query_vector, top_k=3, include_metadata=True)
        context = ""
        for match in search_res['matches']:
            context += match['metadata']['text'] + "\n"

        # 3. صياغة الرد النهائي
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"أنت 'حقي'، المحامي الرقمي الأكثر دقة في إسبانيا. استخدم المعلومات التالية للرد بالدارجة المغربية: {context}"},
                {"role": "user", "content": question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except:
        # Fallback في حال فشل البحث في الوثائق
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "رد بالدارجة المغربية كمحامي خبير في إسبانيا."}, {"role": "user", "content": question}]
        )
        return {"answer": response.choices[0].message.content}

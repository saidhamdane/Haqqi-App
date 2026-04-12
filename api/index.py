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
        # محاولة البحث في الوثائق القانونية أولاً (Pinecone)
        context = ""
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("INDEX_NAME"))
            res = client.embeddings.create(input=[question], model="text-embedding-3-small")
            search_res = index.query(vector=res.data[0].embedding, top_k=2, include_metadata=True)
            context = "\n".join([m['metadata']['text'] for m in search_res['matches']])
        except:
            context = "استخدم معلوماتك العامة عن القانون الإسباني."

        # صياغة الرد بشخصية 'حقي'
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"أنت 'حقي' خبير قانوني في إسبانيا. أجب بالدارجة المغربية. استند لهذه المعلومات: {context}"},
                {"role": "user", "content": question}
            ],
            temperature=0.6
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "سمح ليا، كاين مشكل تقني بسيط. حاول تسولني بطريقة أخرى أو عاود من بعد شوية."}

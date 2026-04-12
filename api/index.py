import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت محامي رقمي خبير في القانون الإسباني. اسمك 'حقي'. أسلوبك محترم، مهني، وتستخدم الدارجة المغربية لتبسيط المفاهيم القانونية المعقدة للمهاجرين. استخدم النقاط (Bullet points) في الإجابات الطويلة."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/index")
async def hello():
    return {"status": "Haqqi AI Engine is Online"}

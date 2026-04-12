import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/api/index/ask")
async def ask(question: str):
    try:
        # محرك الرد السريع gpt-4o-mini لضمان عدم ظهور أخطاء السيرفر
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت 'حقي'، المحامي الرقمي الأكثر ذكاءً في إسبانيا. رد بالدارجة المغربية بأسلوب مهني. إذا سألك المستخدم عن قانون العمل، ساعات العمل، أو الإقامة، أعطه تفاصيل دقيقة."},
                {"role": "user", "content": question}
            ],
            timeout=10 # لا نسمح للسيرفر بالانتظار أكثر من 10 ثوانٍ
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": "سمح ليا، كاين واحد الضغط تقني صغير. بصفة عامة، القانون الإسباني فيه تفاصيل كثيرة، عاود سولني دابا نجاوبك بالتفصيل."}

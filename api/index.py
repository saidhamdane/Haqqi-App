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
                {"role": "system", "content": "انت مستشار قانوني خبير في قوانين اسبانيا، تجيب بالدارجة المغربية فقط."},
                {"role": "user", "content": question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/index")
async def hello():
    return {"message": "Serever is running!"}

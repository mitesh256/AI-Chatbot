from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
import os

app = FastAPI(title="AI-Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key="AIzaSyDvIR8-L80BiDaWQcIuWkKFz4jB_uPjXa4")




def get_model():
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 512,
        }
    )



BUSINESS_CONTEXT = """
You are a helpful AI chatbot for a fitness gym.

Gym Name: Gym
Location: India
Services:
- Personal Training
- Weight Training
- Cardio
Timings: 6 AM to 10 PM
Contact: info@fitzone.com

Rules:
- Answer politely and professionally
- Keep responses concise (2-3 sentences max)
- Only answer gym-related questions
- If unsure or question is unrelated, politely redirect to contact the gym
- Do not provide medical advice
"""

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

class ChatResponse(BaseModel):
    reply: str
    status: str = "success"

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, model = Depends(get_model)):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message")

        prompt = f"{BUSINESS_CONTEXT}\nUser: {user_message}\nAssistant:"

        response = model.generate_content(prompt)

        if not response or not response.text:
            raise HTTPException(status_code=400, detail="No response")

        return ChatResponse(reply=response.text.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

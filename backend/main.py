import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Initialization ---
# Load environment variables from a .env file
load_dotenv()

# Create the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
# This allows your React frontend (running on a different URL) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Pydantic Models ---
# These models define the expected request and response data structures.
class ReviewRequest(BaseModel):
    hotel_name: str
    review: str

class ReplyResponse(BaseModel):
    reply: str

# --- Helper Function ---
def build_prompt(hotel_name: str, feedback: str) -> str:
    """Builds the prompt for the Gemini API call."""
    return f"""
You are a professional, empathetic, and courteous hotel manager responding to a customer review.

Hotel: {hotel_name or 'Our Hotel'}
Review: "{feedback}"

Generate a personalized, polite, and contextually appropriate reply following these rules:
1.  Start with a warm and appropriate salutation like "Dear Guest,". Do not use the customer's name.
2.  If the review is positive, thank the customer warmly and specifically mention something they enjoyed.
3.  If the review is negative, apologize sincerely for the shortcomings and assure them that the issues will be investigated and addressed. Avoid making excuses.
4.  If the review is mixed, acknowledge both the positive and negative points. Thank them for the praise and apologize for the problems.
5.  Keep the reply concise and relevant to the review's length.
6.  End on a positive and forward-looking note, such as inviting them back for a better experience.
7.  If a hotel name was provided, sign off with "Sincerely, The {hotel_name} Team" or similar. If no hotel name was given, use a generic sign-off like "Sincerely, Hotel Management".
"""

# --- API Endpoint ---
@app.post("/generate-reply", response_model=ReplyResponse)
async def generate_reply(request: ReviewRequest):
    """
    Receives review data, generates a reply using the Gemini API,
    and returns the reply.
    """
    if not model:
        return {"reply": "Error: The generative model is not configured correctly. Check the server logs."}
    
    try:
        prompt = build_prompt(request.hotel_name, request.review)
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        print(f"Error during reply generation: {e}")
        return {"reply": f"An error occurred while generating the reply: {e}"}

# To run this app:
# 1. Save it as main.py
# 2. Create a file named .env in the same directory.
# 3. In the .env file, add your API key: GEMINI_API_KEY="YOUR_API_KEY_HERE"
# 4. Install dependencies: pip install fastapi uvicorn "google-generativeai~=0.5" pydantic python-dotenv
# 5. Run the server: uvicorn main:app --reload

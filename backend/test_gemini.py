
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

if not api_key or not model_name:
    print("ERROR: Check your .env file.")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: TraceRAG Gemini connection successful!"
        )

        print(response.text)

    except Exception as error:
        print("Gemini test failed:")
        print(error)

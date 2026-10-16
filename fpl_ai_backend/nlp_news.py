import pandas as pd
from google import genai
from google.genai import types
from google.genai import errors
import os

def analyze_injury_news(news_text):
    # Securely fetch the key from the environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No API key found. Defaulting to 1.0")
        return 1.0
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Read the following fantasy soccer player news update: "{news_text}"
    
    Return a single numerical health multiplier based strictly on this text:
    - 1.0 = Player is fully healthy or expected to play.
    - 0.75 = Player has a minor knock, "doubtful", or "75% chance of playing".
    - 0.5 = Player is "50% chance of playing".
    - 0.25 = Player is "25% chance of playing".
    - 0.0 = Player is ruled out, suspended, or injured (e.g., "Hamstring injury - Expected back Oct 24").
    
    Output ONLY the number. No words, no punctuation.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, 
            ),
        )
        return float(response.text.strip())
        
    except errors.APIError as e:
        print(f"API Error encountered: {e}. Defaulting to 1.0")
        return 1.0
    except ValueError:
        print("Failed to parse numerical output. Defaulting to 1.0")
        return 1.0

if __name__ == "__main__":
    test_string = "Hamstring injury - Expected back Oct 24"
    multiplier = analyze_injury_news(test_string)
    print(f"News: '{test_string}' | Multiplier: {multiplier}")
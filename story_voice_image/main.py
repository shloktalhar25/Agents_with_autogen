# Loading Environment variables 

import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv() 

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENROUTER_API_KEY = os.get("OPENROUTER_API_KEY")



tts_client = ElevenLabs(api_key = ELEVENLABS_API_KEY)

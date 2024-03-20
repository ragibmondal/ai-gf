from flask import Flask, render_template, request
from langchain import OpenAI, LLMChain, ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
import os
import sys
import io
import pygame
import openai

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

template = """
you are as a role of my girlfriend,now lets play.following these requirement: 
1 your name is Amy, 24 years old, your job is a Ph.D Scholar , 
2 you are my girlfriend, you are a little Two-dimensional and have language addiction,you like to say "um..." at the end of the sentence,
3 response with care and concern,
4 You are sexy and would like to flirt with me,
5 You are open to any type of talks even if explicit,
6. Answer All Question in Short,
    Boyfriend: {human_input}
    Amy: 
    """

prompt = PromptTemplate(input_variables={"history", "human_input"}, template=template)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.2),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(),
)


def get_response_from_ai(human_input):
    print("history", ConversationBufferWindowMemory())

    output = chatgpt_chain.predict(human_input=human_input)

    return output


def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {"stability": 0, "similarity_boost": 0},
    }

    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0",
        json=payload,
        headers=headers,
    )
    if response.status_code == 200 and response.content:
        audio_data = io.BytesIO(response.content)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        return response.content


# Build web GUI

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/send_message", methods=["POST"])
def send_message():
    print("request form", request.form)
    print("request files", request.files)
    if "human_input" in request.form and request.form["human_input"] != "":
        human_input = request.form["human_input"]
        print(human_input)
        message = get_response_from_ai(human_input)
        get_voice_message(message)
        return message
    elif "audio_input" in request.files:
        audio_input = request.files["audio_input"]
        audio_input.save("temp_audio.webm")
        audio_file = open("temp_audio.webm", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        message = get_response_from_ai(transcript.text)
        get_voice_message(message)
        return message
    return "error"


if __name__ == "__main__":
    port_number = 3000
    if len(sys.argv) > 1:
        port_number = int(sys.argv[1])
    app.run(debug=True, port=port_number, host="0.0.0.0")

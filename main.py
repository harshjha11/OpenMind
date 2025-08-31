from openai import OpenAI
from fastapi import FastAPI, Form, Request, WebSocket, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Annotated
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv('OPENAI_API_SECRET_KEY'))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Store sessions per user
user_sessions = {}

def get_session(session_id: str):
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "chat_log": [{
                "role": "system",
                "content": "You are a chatbot designed by Harsh to provide information and assistance."
            }],
            "chat_responses": [],
            "image_history": []
        }
    return user_sessions[session_id]

# -------- Chat Routes --------
@app.get("/", response_class=HTMLResponse)
async def chat_app(request: Request, session_id: str = Cookie(default=None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    session = get_session(session_id)
    response = templates.TemplateResponse("home.html", {
        "request": request,
        "chat_responses": session["chat_responses"],
        "session_id": session_id
    })
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.websocket("/ws/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_session(session_id)

    while True:
        try:
            user_input = await websocket.receive_text()
            session["chat_log"].append({'role': 'user', 'content': user_input})
            session["chat_responses"].append(user_input)

            response = openai.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=session["chat_log"],
                temperature=0.6,
                stream=True
            )

            ai_response = ''
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    ai_response += chunk.choices[0].delta.content
                    await websocket.send_text(chunk.choices[0].delta.content)

            session["chat_log"].append({'role': 'assistant', 'content': ai_response})
            session["chat_responses"].append(ai_response)

        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break

# -------- Image Routes --------
@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request, session_id: str = Cookie(default=None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    session = get_session(session_id)
    response = templates.TemplateResponse("image.html", {
        "request": request,
        "image_history": session.get("image_history", []),
        "session_id": session_id
    })
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/image", response_class=HTMLResponse)
async def create_image(
    request: Request,
    user_input: Annotated[str, Form()],
    session_id: str = Cookie(default=None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
    session = get_session(session_id)

    response = openai.images.generate(
        prompt=user_input,
        n=1,
        size='256x256'
    )
    image_url = response.data[0].url
    session["image_history"].append(image_url)

    return templates.TemplateResponse("image.html", {
        "request": request,
        "image_url": image_url,
        "image_history": session.get("image_history", []),
        "session_id": session_id
    })

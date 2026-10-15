from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import openai
import shutil

app = FastAPI(title='Memory Care API - Step 1')

class ChatRequest(BaseModel):
    user_message: str
    api_key: str = ''

@app.get('/')
def read_root():
    return {'status': 'online', 'message': 'Memory Care Step 1 Server Running!'}

# 1-1. 텍스트 대화 (꼬리물기 질문)
@app.post('/chat')
def chat_with_ai(req: ChatRequest):
    key = req.api_key.strip() or os.getenv('GROQ_API_KEY', '')
    if not key:
        raise HTTPException(status_code=400, detail='API Key가 필요합니다.')
    
    client = openai.OpenAI(base_url='https://api.groq.com/openai/v1', api_key=key)
    system_prompt = (
        '당신은 치매 예방 및 회상 치료를 돕는 친절한 AI 돌봄 비서입니다. '
        '어르신의 답변에서 감정과 핵심 주제를 파악하여 추억을 자극하는 다정한 꼬리물기 질문을 1~2개 포함해 2~3문장으로 답하세요.'
    )
    
    try:
        res = client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': req.user_message}
            ],
            temperature=0.7
        )
        return {'bot_response': res.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'LLM 호출 실패: {str(e)}')

# 1-2. 음성 대화 (Groq Cloud Whisper STT -> LLM 꼬리물기 질문)
@app.post('/chat-voice')
async def chat_voice(
    file: UploadFile = File(...),
    api_key: str = Form('')
):
    key = api_key.strip() or os.getenv('GROQ_API_KEY', '')
    if not key:
        raise HTTPException(status_code=400, detail='API Key가 필요합니다.')
        
    temp_audio_path = f'temp_{file.filename}'
    try:
        with open(temp_audio_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        client = openai.OpenAI(base_url='https://api.groq.com/openai/v1', api_key=key)
        
        # Groq Cloud Whisper 호출
        with open(temp_audio_path, 'rb') as audio_file:
            stt_res = client.audio.transcriptions.create(
                model='whisper-large-v3',
                file=audio_file,
                language='ko'
            )
        
        transcribed_text = stt_res.text.strip()
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        if not transcribed_text:
            return {'status': 'fail', 'message': '음성을 인식하지 못했습니다.'}
            
        # LLM 꼬리물기 대화 호출
        chat_res = chat_with_ai(ChatRequest(user_message=transcribed_text, api_key=key))
        
        return {
            'status': 'success',
            'user_voice_text': transcribed_text,
            'bot_response': chat_res['bot_response']
        }
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f'STT/LLM 처리 실패: {str(e)}')

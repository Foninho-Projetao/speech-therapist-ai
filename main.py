from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import json

from gemini_classification import get_gemini_classification, get_gemini_classification_2
from mediapipe_classification import get_mediapipe_cheek_classification, get_mediapipe_vibration_classification


def extrair_resposta(texto):
    # Procura por um bloco de JSON dentro de ```json ... ```
    padrao_json = re.search(r'```json\s*(\{.*?\})\s*```', texto, re.DOTALL)
    
    if padrao_json:
        try:
            bloco_json = padrao_json.group(1)
            dados = json.loads(bloco_json)
            return dados.get("resposta")
        except json.JSONDecodeError:
            print("Erro ao decodificar JSON.")
            return None
    else:
        print("Bloco JSON não encontrado.")
        return None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#VERSÃO DO CODIGO PARA TRABALHAR EM UM SÓ PC

@app.post("/upload/", response_class=PlainTextResponse)
async def upload_video(
    nome_exercicio: str = Form(...),
    video: str = Form(...)
):
    try:
        print(f"Vídeo recebido: {video}")
        print(f"Nome do exercício: {nome_exercicio}")

        gemini = ["1", "2"]

        if nome_exercicio in gemini:
            response = get_gemini_classification(nome_exercicio, video)
            response = extrair_resposta(response)
        elif nome_exercicio == "3":
            response = get_mediapipe_cheek_classification(video)
        elif nome_exercicio == "4":
            response = get_mediapipe_vibration_classification(video)
        else:
            print ("Exercicio nao reconhecido")
            response = "Parcialmente Correto"
            
        print(f"Resposta: {response}")
        return response

    except Exception as e:
        print(f"Erro ao salvar o vídeo: {e}")
        return "Erro ao salvar o vídeo"

# PARA RODAR
# python -m uvicorn main:app --reload --port 5001






#VERSAO DO CODIGO PARA TRABALHAR COM DOIS PCS (um para o back da i.a e outro para o APP)

# @app.post("/upload/", response_class=PlainTextResponse)
# async def upload_video(
#     nome_exercicio: str = Form(...),
#     video: UploadFile = File(...)
# ):
#     try:
#         print(f"Vídeo recebido: {video.filename}")
#         print(f"Nome do exercício: {nome_exercicio}")
        
#         # Caminho da pasta onde o vídeo será salvo
#         save_dir = "videos"

#         # Cria a pasta se não existir
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             print(f"Pasta '{save_dir}' criada.")

#         save_path = os.path.join(save_dir, video.filename)

#         # Salva o vídeo
#         with open(save_path, "wb") as f:
#             while chunk := await video.read(1024 * 1024):  # lê em chunks de 1MB
#                 f.write(chunk)

#         print(f"Vídeo salvo em: {save_path}")
#         return "Correto"

#     except Exception as e:
#         print(f"Erro ao salvar o vídeo: {e}")
#         return "Erro ao salvar o vídeo"
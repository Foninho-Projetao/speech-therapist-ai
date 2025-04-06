from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# (opcional) Habilita CORS se você estiver testando com frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique os domínios permitidos
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
        
        return "Correto"

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
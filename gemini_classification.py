import os
from google import genai

from dotenv import load_dotenv
from google.genai import types

import time
from google.genai import errors

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)


lateralize_tongue_prompt = '''Veja o video da execução do paciente e avalie se o exercício de lateralização de lingua foi performado corretamente, parcialmente ou incorretamente, de acordo com a descrição do exercício fornecida. 

Descrição: O paciente deve projetar a lingua para fora da boca e depois deve mover o máximo possivel para os dois lados, um de cada vez, e retrair novamente. 
- Situações em que o paciente moveu a lingua apenas para um dos lados ou que não projetou a ligua para fora completamente são consideradas execuções parciais.
- Situações em que não houve projeção para fora ou que ele não moveu a lingua são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

lateralize_tongue_prompt_2 = '''O video anterior é de uma fonoaudióloga realizando o exercócio de lateralizar a língua. Com base nessa execução, avalie a performance de um paciente mostrada a seguir.

Descrição: O paciente deve abrir a boca, projetar a lingua para fora e depois deve mover para os dois lados, um de cada vez. 
- Situações em que não houve projeção para fora ou que ele não moveu a lingua são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

produde_and_retract_tongue_prompt = '''Veja o video da execução do paciente e avalie se o exercício de protrusão e retração de lingua foi performado corretamente, parcialmente ou incorretamente, de acordo com a descrição do exercício fornecida. 

Descrição: O paciente deve projetar a lingua para fora da boca (obrigatoriamente para frente) e depois deve retrair novamente para dentro. 
- Situações em que o paciente não projetou a lingua significativamente para fora ou não retraiu a lingua completamente são consideradas execuções parciais.
- Situações em que não houve projeção para fora ou que ele não retraiu a lingua ao final são consideradas execuções incorretas.
- Qualquer movimento adicional que não seja a projeção e retração da lingua, como a lateralização, mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

produde_and_retract_tongue_prompt_2 = '''O video anterior é de uma fonoaudióloga realizando o exercócio de protruir e retrair a língua. Com base nessa execução, avalie a performance de um paciente mostrada a seguir.

Descrição: O paciente deve abrir a boca e projetar a lingua para fora da boca (obrigatoriamente para frente) e depois deve retrair novamente para dentro, mantendo a boca aberta. 
- Situações em que não houve projeção para fora ou que ele não retraiu a lingua ao final são consideradas execuções incorretas.
- Qualquer movimento adicional que não seja a projeção e retração da lingua, como a lateralização, mordidas, caretas etc devem ser considerados incorretos.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

infalte_cheeks_prompt = '''Veja o video da execução do paciente e avalie se o exercício de inflar as bochechas foi performado corretamente, parcialmente ou incorretamente, de acordo com a descrição do exercício fornecida. 

Descrição: O paciente deve inflar as bochechas, uma de cada vez. 
- Situações em que o paciente inflou um dos lados ou são consideradas execuções parciais.
- Situações em que não houve nenhuma inflagem ou movimento são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

infalte_cheeks_prompt_2 = '''O video anterior é de uma fonoaudióloga realizando o exercócio de inflar as bochechas. Com base nessa execução, avalie a performance de um paciente mostrada a seguir.

Descrição: O paciente deve inflar as bochechas, uma de cada vez. 
- Situações em que não houve nenhuma inflagem ou movimento são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

vibrate_lips_prompt = '''Veja o video da execução do paciente e avalie se o exercício de vibrar os lábios foi performado corretamente, parcialmente ou incorretamente, de acordo com a descrição do exercício fornecida. 

Descrição: O paciente deve vibrar os lábios. 
- Situações em que o paciente vibrou os labios de forma pausada são consideradas execuções parciais.
- Situações em que não houve nenhuma vibração ou movimento são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

vibrate_lips_prompt_2 = '''O video anterior é de uma fonoaudióloga realizando o exercócio de vibrar os lábios. Com base nessa execução, avalie a performance de um paciente mostrada a seguir.

Descrição: O paciente deve vibrar os lábios. 
- Situações em que não houve nenhuma vibração ou movimento são consideradas execuções incorretas. Situações sem movimento tambem devem ser consideradas incorretas.
- Qualquer movimento adicional que não seja a projeção e lateralização da lingua, como mordidas, caretas etc devem ser considerados incorretor.

Retorne APENAS "Correto", "Parcialmente Correto" ou "Incorreto" e um breve comentário sobre o porquê da sua resposta. A estrutura da resposta deve ser a seguinte:
{
    "resposta": "Correto/Parcialmente Correto/Incorreto",
    "comentario": "Seu comentário aqui"
}
'''

def get_gemini_classification(exercise, video_file, max_retries=5, retry_delay=2):

    if exercise == "2":
        prompt = produde_and_retract_tongue_prompt
    if exercise == "1":
        prompt = lateralize_tongue_prompt
    if exercise == "infalte_cheeks":
        prompt = infalte_cheeks_prompt
    if exercise == "vibrate_lips":
        prompt = vibrate_lips_prompt

    video_bytes = open(video_file, 'rb').read()

    curr_model = 'models/gemini-1.5-pro'

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=curr_model,
                contents=types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        )
                    ]
                )
            )
            break
        except errors.APIError as e:
            if e.code == 503 or e.code == 504:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed with 503 error. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Max retries reached. Server is still unavailable.")
                    raise
            elif e.code == 429:
                print(f'{curr_model} was Exhausted...')
                curr_model = 'models/gemini-2.5-pro-exp-03-25'
                print(f'Swithing to {curr_model}...')
            else:
                print(f"An unexpected error occurred: {str(e)}")
                raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

    return response.text

def get_gemini_classification_2(exercise, video_file, max_retries=5, retry_delay=2):

    if exercise == "2":
        prompt = produde_and_retract_tongue_prompt_2
        fono_video_file_name = "experiments/fono/ex2_fono.mp4"
    if exercise == "1":
        prompt = lateralize_tongue_prompt_2
        fono_video_file_name = "experiments/fono/ex1_fono.mp4"
    if exercise == "infalte_cheeks":
        prompt = infalte_cheeks_prompt_2
        fono_video_file_name = "experiments/fono/ex3_fono.mp4"
    if exercise == "vibrate_lips":
        prompt = vibrate_lips_prompt_2
        fono_video_file_name = "experiments/fono/ex4_fono.mp4"

    fono_video_bytes = open(fono_video_file_name, 'rb').read()
    video_bytes = open(video_file, 'rb').read()

    curr_model = 'models/gemini-1.5-pro'

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=curr_model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=fono_video_bytes, mime_type='video/mp4')
                        ),
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        )
                    ]
                )
            )
            break
        except errors.APIError as e:
            if e.code == 503 or e.code == 504:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed with 503 error. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Max retries reached. Server is still unavailable.")
                    raise
            elif e.code == 429:
                print(f'{curr_model} was Exhausted...')
                curr_model = 'models/gemini-2.5-pro-exp-03-25'
                print(f'Swithing to {curr_model}...')
            else:
                print(f"An unexpected error occurred: {str(e)}")
                raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

    return response.text


if __name__ == '__main__':
    print(get_gemini_classification("produde_and_retract_tongue", "experiments/ex1_errado.mp4"))
import argparse
import os
import time
import numpy as np
# from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, ImageClip, CompositeVideoClip, CompositeAudioClip, VideoClip, TextClip
from moviepy.audio.AudioClip import AudioClip
import asyncio
import edge_tts
# import srt
# from datetime import timedelta
from ollama import Client
import re
import json
# import shutil
# from pathlib import Path
import requests
# import subprocess
from moviepy.editor import concatenate_audioclips
import math
import random
# from moviepy.video.fx.all import resize
# import moviepy.video.fx.all as vfx
import cv2
import datetime
# import traceback
import whisper
# from moviepy.video.tools.subtitles import SubtitlesClip
from typing import Optional, Dict, List, Tuple
from moviepy.config import change_settings
from glob import glob
import sys
from TTS.api import TTS
import torch
# from collections import Counter
# import nest_asyncio
# nest_asyncio.apply()
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

def parse_arguments():
    """Configura e processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Gerador automático de vídeos com narração usando LLM.')
    
    parser.add_argument('--prompt', type=str, help='Prompt inicial para o LLM gerar o roteiro do vídeo')
    parser.add_argument('--model', type=str, default='gemma2:9b', help='Modelo do Ollama a ser usado (default: gemma2:9b)')
    parser.add_argument('--sd_model', type=str, default='flux_dev', help='Modelo a ser usado do SD (default: flux_dev)')
    parser.add_argument('--output', type=str, default='video_final.mp4', help='Nome do arquivo de vídeo de saída')
    parser.add_argument('--duration', type=int, default=60, help='Duração desejada do vídeo em segundos (default: 120)')
    parser.add_argument('--sd_steps', type=int, default=40, help='Número de steps para o Stable Diffusion (default: 30)')
    parser.add_argument('--v_width', type=int, default=720, help='largura do vídeo (default: 720)')
    parser.add_argument('--v_height', type=int, default=1280, help='Altura do vídeo (default: 1280)')
    parser.add_argument('--zoom', action='store_true', help='Modo zoom (se presente, é True; se ausente, é False)')
    parser.add_argument('--subs', action='store_true', help='Habilita legendas (se presente, é True; se ausente, é False)')
    parser.add_argument('--quality', type=float, default=1, help='qualidade das imagens (1 qualidade máxima, valores maiores diminuem a qualidade)')
    parser.add_argument('--paymode', type=str, default='default', help='modo payload SD')
    
    return parser.parse_args()

#%% Configurações

args = parse_arguments()
stable_model = args.sd_model
stable_difusion_steps = args.sd_steps
v_width = args.v_width
v_height = args.v_height
quality = args.quality
enable_zoom = args.zoom
enable_sub = args.subs
ollama_client = Client(host='http://localhost:11434')
STABLE_DIFFUSION_URL = "http://127.0.0.1:7860"
  
#%% Funções

def create_run_directory():
    """Creates a timestamped run directory and necessary subdirectories."""
    # Create timestamp string
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H_%M_%S')
    run_dir = f'run_{timestamp}'
    
    # Create main run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'temp'), exist_ok=True)
    
    return run_dir

def setup_directories():
    """Creates and returns paths to all necessary directories."""
    # Create run directory with timestamp
    run_dir = create_run_directory()
    
    # Create paths dictionary
    paths = {
        'run': run_dir,
        'data': os.path.join(run_dir, 'data'),
        'temp': os.path.join(run_dir, 'temp'),
        'music': 'music'  # Music directory remains in root
    }
    
    # Ensure music directory exists
    os.makedirs(paths['music'], exist_ok=True)
    
    return paths


async def generate_image(prompt, filename, negative_prompt="", data_dir="data"):
    """Gera uma imagem usando Stable Diffusion local com parâmetros otimizados para conteúdo TikTok."""
    try:
        # Define o modelo a ser usado
        model_map = {
            "a": "realisticComicBook_v10",
            "b": "comicBabes_v2",
            "c": "juggernautXL_juggXIByRundiffusion",
            "d": "realisticVisionV60B1_v51HyperVAE",
            "e": "absolutereality_v181"
        }
        sd_model = model_map.get(stable_model.lower())

        quality_style = (
            "realistic comic book illustration of"
        )
        
        enhanced_prompt = f"{quality_style} {prompt}"
        
        quality_negative = """shirtless, nsfw, naked, duplicated, twin, no repeated characters, bad anatomy, bad proportions, disfigured, gross proportions, cloned, black and white, multiple people, twins, two persons, duplicate person, second character, repeating figures, clone characters"""
        
        enhanced_negative = f"{quality_negative}"
        if args.paymode == 'movie':
            payload = {
                "prompt": f"realistic comic book illustration of, {prompt}",
                "negative_prompt": enhanced_negative,
                "steps": stable_difusion_steps,
                "width": 640,
                "height": 640,
                "sampler_name": "DPM++ 2M Karras",
                "cfg_scale": 6,
                "karras": True,
                "refiner_checkpoint": "realisticComicBook_v10",  # Modelo do refinador
                "refiner_switch": 0.3,  # Momento de troca para o refinador
                "override_settings": {
                    "sd_model_checkpoint": "absolutereality_v181"  # Modelo principal
                },
                "override_settings_restore_afterwards": True,
                "enable_hr": True,              # Ativa o hires.fix
                "hr_upscaler": "Latent",       # Tipo de upscaler
                "hr_steps": 10,                # Passos adicionais para o hires.fix
                "denoising_strength": 0.4,     # Força do denoising
                "hr_scale": 2.0,                # Fator de upscale
                "alwayson_scripts": {
                    "ADetailer": {
                        "args": [
                            {
                                "ad_prompt": f"{prompt}",
                                "ad_model": "face_yolov8s.pt",
                                "ad_confidence": 0.3,
                                "ad_dilate_erode": 4,
                                "ad_mask_blur": 4,
                                "ad_denoising_strength": 0.6,
                                "ad_steps": 40,
                                "ad_cfg_scale": 6,
                                "ad_inpaint_only_masked": True,
                                "ad_inpaint_padding": 32,
                            }
                        ]
                    }
                }
            }
            
        else:
            # Parâmetros otimizados
            payload = {
                "prompt": f"realistic comic book illustration of, {prompt}",
                "negative_prompt": enhanced_negative,
                "steps": stable_difusion_steps,
                "width": 640,
                "height": 640,
                "sampler_name": "DPM++ SDE Karras",
                "cfg_scale": 4.5,
                "karras": True,
                "refiner_checkpoint": "realisticComicBook_v10",
                "refiner_switch_at": 0.4,
                "override_settings": {
                    "sd_model_checkpoint": "absolutereality_v181"
                },
                "override_settings_restore_afterwards": True,
                "enable_hr": True,              # Ativa o hires.fix
                "hr_upscaler": "Latent",       # Tipo de upscaler
                "hr_steps": 10,                # Passos adicionais para o hires.fix
                "denoising_strength": 0.4,     # Força do denoising
                "hr_scale": 2.0                # Fator de upscale
            }
        
        response = requests.post(
            f"{STABLE_DIFFUSION_URL}/sdapi/v1/txt2img",
            json=payload
        )
        
        if response.status_code == 200:
            r = response.json()
            image_data = r['images'][0]
            
            import base64
            from io import BytesIO
            from PIL import Image, ImageEnhance
            
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            
            # Pipeline de pós-processamento ajustado
            enhancers = [
                ('Sharpness', 1.15),
                ('Contrast', 1.0),
                ('Color', 0.55),
                ('Brightness', 1.05)
            ]
            
            for enhancer_type, factor in enhancers:
                enhancer = getattr(ImageEnhance, enhancer_type)(image)
                image = enhancer.enhance(factor)
            
            # Salva com compressão otimizada
            image.save(
                os.path.join(data_dir, filename),
                quality=95,
                optimize=True
            )
            return True
        else:
            print(f"Erro ao gerar imagem: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Erro ao gerar imagem: {str(e)}")
        return False


def create_smooth_zoom(clip, zoom_direction='in', zoom_factor=0.08):
    """
    Função que cria um zoom suave, Abordagem alternativa usando matrizes de transformação e interpolação OpenCV
    """
    w, h = clip.size
    duration = clip.duration
    
    # Criar matriz de transformação base
    # center_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    
    def get_frame(t):
        # Calcular progresso com suavização gaussiana
        progress = t / duration
        # Aplicar curva suave
        progress = np.sin(progress * np.pi / 2)
        
        if zoom_direction == 'in':
            scale = 1 + (zoom_factor * progress)
        else:
            scale = 1 + (zoom_factor * (1 - progress))
            
        # Calcular dimensões com alta precisão
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Calcular pontos de destino com base no zoom
        offset_x = (w * (scale - 1)) / 2
        offset_y = (h * (scale - 1)) / 2
        dst_points = np.float32([
            [-offset_x, -offset_y],
            [w + offset_x, -offset_y],
            [-offset_x, h + offset_y],
            [w + offset_x, h + offset_y]
        ])
        
        # Obter matriz de transformação
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar transformação com interpolação de alta qualidade
        frame = clip.get_frame(t)
        transformed = cv2.warpPerspective(
            frame,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return transformed

    return VideoClip(get_frame, duration=duration)



def image_to_video_clip(image_path, output_video, duration):
    """
    Função para converter uma imagem para vídeo mp4
    """
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Criar clip
        image_clip = ImageClip(image).set_duration(duration)
        
        # Redimensionar para HD
        image_clip = image_clip.resize(width=v_width, height=v_height)

        if enable_zoom:
            moving_clip = create_smooth_zoom(
                image_clip,
                zoom_direction='in',
                zoom_factor=0.09
            )
        else:
            moving_clip = image_clip
        
        if enable_zoom:
            # Configurações de alta qualidade
            moving_clip.write_videofile(
                output_video,
                fps=30,  
                codec='h264_nvenc',
                audio=False,
                preset='slow',
                ffmpeg_params=[
                    '-crf', '17',
                    '-tune', 'film',
                    '-profile:v', 'high',
                    '-movflags', '+faststart',
                    '-vf', 'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'
                ],
                verbose=False,
                logger=None
            )
        else:
            # Configurações de alta qualidade
            if enable_zoom:
                vid_fps = 30
            else:
                vid_fps = 1
            
            moving_clip.write_videofile(
                output_video,
                fps=vid_fps,  
                # codec='libx264',
                codec='h264_nvenc',
                audio=False,
                preset='medium',
                ffmpeg_params=[
                    '-pix_fmt', 'yuv420p',  
                    '-profile:v', 'main',    
                    '-movflags', '+faststart'
                ],
                verbose=False,
                logger=None
            )
            
        
        # Limpar
        moving_clip.close()
        image_clip.close()
        
    except Exception as e:
        print(f"Erro ao criar vídeo a partir da imagem: {str(e)}")
        exit()


async def generate_script(prompt, path, model='gemma2:9b', duration=120):
    """Gera o roteiro completo do vídeo usando Ollama."""
    
    try:
        # Verifica a existência de um roteiro na pasta temp raiz
        root_temp_path = 'temp'
        root_script_path = os.path.join(root_temp_path, 'script.json')

        if os.path.exists(root_script_path):
            with open(root_script_path, 'r', encoding='utf-8') as f:
                script = json.load(f)
            print("Roteiro existente encontrado no diretório raiz! Reutilizando...")
            return script
    except:
        print("Roteiro existente não encontrado no diretório raiz...")
        pass
    
    try:
        min_scene_duration = 8
        max_scene_duration = 15
        min_scenes_needed = math.ceil(duration / max_scene_duration)
        max_scenes_needed = math.floor(duration / min_scene_duration)
        target_scenes = (min_scenes_needed + max_scenes_needed) // 2
        
        # Exemplo de distribuição de tempo para orientar a LLM
        avg_scene_duration = duration / target_scenes
        
        system_prompt = f"""Você é um assistente especializado em criar roteiros para vídeos criativos e filosóficos. 
        Analise o prompt do usuário e gere um roteiro estruturado para um vídeo narrado em português de {duration} segundos.
        
        - São necessárias {target_scenes} cenas com duração média de {avg_scene_duration:.1f} segundos.
        - Evite descrições que gerem imagens de pessoas muito próximas ou situações detalhadas envolvendo mãos segurando objetos ou gestos específicos.
        - Prefira imagens abertas ou que mostrem cenários amplos, com pessoas aparecendo de forma distante, de costas ou como parte do cenário.
        - O image_prompt para cada cena deve sempre especificar o que está na cena. Exemplo: 'Wide shot of an elderly astronomer standing on a hill, observing the night sky with a telescope, stars filling the background.'
        - O image_prompt precisa ser em inglês e deve descrever a cena sempre especificando se é um homem, uma mulher, um local.
        - Não gere prompts para imagem do tipo "Close up shot of Pedro's hand holding the old pocket watch". Ao invés disso, gere sempre algo como "A young man holding an old pocket watch"
        - O narration (texto a para narração) precisa ser em português. Exemplo: "Enquanto ajustava o telescópio, João notou um padrão incomum de luz chamou sua atenção. As estrelas pareciam piscar em uma sequência rítmica, como se tentassem comunicar algo."
        - Retorne APENAS o JSON, sem texto adicional.
        
        Estrutura exata do JSON esperado:
        {{
            "scenes": [
                {{
                    "image_prompt": "Prompt para Stable Diffusion que representa a cena.",
                    "narration": "Texto que deve ser narrado durante esta cena, composto por 5 ou 6 frases.",
                    "duration": 10
                }},
                {{
                    "image_prompt": "Prompt em inglês para a imagem da cena 2",
                    "narration": "Narração em português 2",
                    "duration": 10
                }}
            ],
            "seo": {{
                "title": "Título SEO",
                "description": "Descrição SEO"
            }}
        }}"""
        
        full_prompt = f"{system_prompt}\n\nPrompt do usuário: {prompt}"
        
        response = ollama_client.generate(
            model=model,
            prompt=full_prompt,
            stream=False
        )
        
        def clean_json_string(json_str):
            """Limpa e corrige problemas comuns em strings JSON."""
            # Remove qualquer texto antes do primeiro {
            json_str = json_str[json_str.find('{'):]
            # Remove qualquer texto depois do último }
            json_str = json_str[:json_str.rfind('}')+1]
            # Remove vírgulas extras antes de fechamento de arrays/objetos
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json_str
        
        # Tente capturar apenas o JSON da resposta
        json_match = re.search(r'\{[\s\S]*\}', response['response'])
        if json_match:
            json_str = json_match.group()
            try:
                # Limpa e corrige o JSON antes de fazer o parse
                cleaned_json = clean_json_string(json_str)
                script = json.loads(cleaned_json)
            except json.JSONDecodeError as json_err:
                print("Erro ao decodificar JSON:", json_err)
                print("JSON original:", json_str)
                print("JSON limpo:", cleaned_json)
                return None
        else:
            print("JSON não encontrado na resposta. Resposta completa:", response['response'])
            return None
        
        # Salvar script.json na pasta temp
        with open(os.path.join(path, 'script.json'), 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        
        # Salvar arquivos de texto na pasta temp
        for i, scene in enumerate(script['scenes'], 1):
            with open(os.path.join(path, f'{i:02d}.txt'), 'w', encoding='utf-8') as f:
                f.write(scene['narration'])
        
        return script
    
    except Exception as e:
        print(f"Erro ao gerar roteiro: {str(e)}")
        return None


# função generate_video_seo para usar os elementos SEO já gerados
def generate_video_seo(script: Dict) -> Dict[str, str]:
    """
    Retorna os elementos SEO já gerados pela LLM
    """
    seo = script.get('seo', {})
    return {
        'title': seo.get('title', ''),
        'description': seo.get('description', '')
        # 'hashtags': ' '.join(f'#{tag}' for tag in seo.get('tags', []))
    }


async def generate_narration(image_path, script_data=None):
    """Gera texto de narração para uma imagem usando o script existente."""
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if script_data and 'scenes' in script_data:
            scene_index = int(base_name) - 1
            if 0 <= scene_index < len(script_data['scenes']):
                return script_data['scenes'][scene_index]['narration']
               
    except Exception as e:
        print(f"Erro ao gerar narração: {str(e)}")
        return None


async def text_to_speech_edge(text, filename):
    """Converte texto em áudio usando edge_tts."""
    VOICES = ['pt-BR-AntonioNeural', 'pt-BR-FranciscaNeural']
    VOICE = VOICES[0]
    communicate = edge_tts.Communicate(text, VOICE, pitch="-3Hz", rate="+13%")
    await communicate.save(filename)
    
async def text_to_speech_coqui(text, filename):
    """Converte texto em áudio usando Coqui TTS."""
    text = text.replace(".", ";")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, speaker="Ana Florence", language="pt", file_path=filename)
    return os.path.abspath(filename)

def text_to_speech(text, filename):
    # asyncio.run(text_to_speech_edge(text, filename))
    asyncio.run(text_to_speech_coqui(text, filename))

def safe_remove(file_path, max_attempts=5, delay=1):
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                print(f"Não foi possível remover {file_path} após {max_attempts} tentativas.")


async def process_file_async(file_path, start_time, output_dir, script_data=None):
    """Versão assíncrona do process_file com melhor sincronização de áudio"""
    file_extension = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Usar caminhos absolutos para os arquivos
    txt_path = os.path.join(output_dir, "data", f"{base_name}.txt")
    audio_path = os.path.join(output_dir, f"{base_name}.mp3")
    
    # Gerar ou carregar texto da narração
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = await generate_narration(file_path, script_data)
        if text:
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(f"Não foi possível gerar narração para {file_path}")
            text = ""
            audio_path = None

    # Gerar áudio da narração se houver texto
    if text:
        await text_to_speech_coqui(text, audio_path)

    # Processar vídeo
    if file_extension == '.mp4':
        # Carregar o vídeo sem áudio para evitar problemas de buffer
        video_clip = VideoFileClip(file_path, audio=False)
        
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            video_duration = video_clip.duration
            
            # Se o vídeo for mais curto que o áudio, estender sua duração
            if video_duration < audio_duration:
                frame_t = video_clip.duration - 1
                # Criar extensão do vídeo sem áudio
                ext_clip = video_clip.to_ImageClip(t=frame_t, duration=(audio_duration - video_duration))
                video_clip = concatenate_videoclips([video_clip, ext_clip], method="compose")
            
            # Se o vídeo for mais longo que o áudio, criar silêncio para o restante
            if video_duration > audio_duration:
                # Criar um clipe de silêncio usando uma função que retorna zero
                silence_duration = video_duration - audio_duration
                silence = AudioClip(lambda t: 0, duration=silence_duration)
                silence = silence.set_fps(audio_clip.fps)  # Usar mesmo fps do áudio original
                
                # Aplicar fade out no áudio original e fade in no silêncio
                audio_clip = audio_clip.audio_fadeout(0.1)
                silence = silence.audio_fadein(0.1)
                
                # Concatenar o áudio original com o silêncio
                final_audio = concatenate_audioclips([audio_clip, silence])
            else:
                final_audio = audio_clip
            
            # Garantir que as durações sejam exatamente iguais
            final_duration = max(video_duration, audio_duration)
            final_audio = final_audio.set_duration(final_duration)
            video_clip = video_clip.set_duration(final_duration)
            
            # Aplicar o áudio ao vídeo
            video_clip = video_clip.set_audio(final_audio)
            
            subtitle = None
                
        return video_clip, audio_path, subtitle, start_time + video_clip.duration
    return None, None, None, start_time



async def create_video_async(args, run_dir, script_data=None):
    """Versão assíncrona do create_video com transições suaves"""
    clips = []
    subtitles = []
    temp_files = []
    start_time = 0
    
    # Usar caminhos absolutos
    data_dir = os.path.join(run_dir, "data")

    
    # Garantir que os diretórios existam
    os.makedirs(data_dir, exist_ok=True)

    
    data_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
    
    if not data_files:
        print("Nenhum arquivo de mídia encontrado na pasta 'data'.")
        return

    # Ordenar arquivos numericamente
    data_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

    # Processar cada arquivo
    for file in data_files:
        try:
            file_path = os.path.join(data_dir, file)
            clip, audio_path, subtitle, new_start_time = await process_file_async(file_path, start_time, run_dir, script_data)

            if clip is not None and clip.duration > 0:
                if audio_path:
                    temp_files.append(audio_path)
                clips.append(clip)
                if subtitle:
                    subtitles.append(subtitle)
        except Exception as e:
            print(f"Erro ao processar arquivo {file}: {str(e)}")
            continue

    if not clips:
        print("Nenhum clipe válido foi criado.")
        return

    try:
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Garantir que o diretório de saída exista
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Escrever o vídeo final
        final_video.write_videofile(
            args.output,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(run_dir, "temp_audio.m4a"),
            remove_temp=True,
            write_logfile=False,
            verbose=False
        )

    except Exception as e:
        print(f"Erro ao criar vídeo final: {str(e)}")
        raise
    finally:
        # Limpar recursos
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_video.close()
        except:
            pass
        
        # Limpar arquivos temporários
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

def clean_text_for_timing(text: str) -> str:
    """
    Limpa o texto mantendo pontuação para cálculo de timing
    """
    # Remove apenas espaços extras mantendo pontuação
    text = ' '.join(text.split())
    return text

def clean_text_for_display(text: str) -> str:
    """
    Limpa o texto para exibição, removendo pontuação
    """
    # Remove pontos, vírgulas e espaços extras
    # text = text.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    text = text.replace('.', '').replace(',', '')
    text = ' '.join(text.split())
    return text

def convert_whisper_segments(segments: List[Dict], words_per_segment: int = 30, min_silence: float = 0.3) -> List[tuple]:
    """
    Converte os segmentos do Whisper para o formato (start_time, end_time, text)
    Usa pontuação para timing mas remove para exibição
    """
    processed_segments = []
    last_end_time = 0  # Mantém registro do fim da última legenda
    
    for i, segment in enumerate(segments):
        # Mantém pontuação para processamento de timing
        text_with_punct = clean_text_for_timing(segment['text']).strip()
        if not text_with_punct:
            continue
            
        # Divide o texto em frases usando pontuação
        sentences = []
        current_sentence = []
        words = text_with_punct.split()
        
        for word in words:
            current_sentence.append(word)
            if word[-1] in '.!?':
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        # Processa cada frase
        segment_duration = segment['end'] - segment['start']
        words_in_segment = len(words)
        time_per_word = segment_duration / words_in_segment
        
        current_time = max(segment['start'], last_end_time + min_silence)  # Garante espaço entre legendas
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_duration = len(sentence_words) * time_per_word
            
            # Adiciona um pequeno atraso após pontuações fortes
            if sentence[-1] in '.!?':
                sentence_duration += min_silence
            
            # Verifica se há um silêncio significativo até o próximo segmento
            if i < len(segments) - 1:
                gap_to_next = segments[i + 1]['start'] - segment['end']
                if gap_to_next > min_silence:
                    sentence_duration += min_silence
            
            # Remove pontuação apenas no texto que será exibido
            display_text = clean_text_for_display(sentence)
            
            end_time = current_time + sentence_duration
            
            # Se houver sobreposição com o próximo segmento, ajusta a duração
            if i < len(segments) - 1 and end_time > segments[i + 1]['start']:
                end_time = segments[i + 1]['start'] - min_silence
            
            processed_segments.append((
                current_time,
                end_time,
                display_text
            ))
            
            current_time = end_time + min_silence  # Adiciona espaço entre frases
            last_end_time = end_time  # Atualiza o tempo final da última legenda
    
    return processed_segments

def create_subtitle_clips(segments: List[tuple], 
                        fontsize: int = 40, 
                        font: str = 'Comic-Book-Bold', 
                        color: str = 'yellow', 
                        stroke_color: str = 'black',
                        stroke_width: int = 2) -> List[TextClip]:
    """
    Cria clips de texto para cada segmento de legenda com fade in/out
    """
    subtitle_clips = []
    # fade_duration = 0.01  # Duração do fade reduzida para 150ms
    
    for start, end, text in segments:
        duration = end - start
        
        # Configura a posição da legenda baseado na orientação do vídeo
        if v_width == 720:  # Vídeo vertical (9:16)
            position = ('center', 900)  
        else:
            position = ('center', 1000)
        
        # Cria o clip de texto
        txt_clip = TextClip(
            text,
            fontsize=fontsize,
            font=font,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            method='caption',
            size=(v_width, 0),  # Usa 90% da largura do vídeo
            transparent=True
        ).set_position(position)
        
        # Aplica a duração exata do segmento
        txt_clip = txt_clip.set_duration(duration)
               
        # Define o momento exato de início
        txt_clip = txt_clip.set_start(start)
        
        subtitle_clips.append(txt_clip)
    
    return subtitle_clips

async def add_subtitles_to_video(
    input_video_path: str,
    output_video_path: str,
    whisper_model_size: str = "medium",
    language: str = "pt",
    font_size: int = 40,
    font: str = "Comic-Book-Bold",
    text_color: str = "yellow",
    stroke_color: str = "black",
    stroke_width: int = 2
) -> Optional[str]:
    """
    Adiciona legendas ao vídeo usando Whisper para transcrição
    """
    try:
        print("Carregando modelo Whisper...")
        model = whisper.load_model(whisper_model_size)
        
        print("Transcrevendo áudio...")
        result = model.transcribe(
            input_video_path,
            language=language,
            verbose=False,
            word_timestamps=True,
            condition_on_previous_text=True,
            no_speech_threshold=0.5,
            hallucination_silence_threshold=0.5
        )
        
        print("Processando segmentos de legendas...")
        segments = convert_whisper_segments(
            result["segments"],
            words_per_segment=35,
            min_silence=0.3
        )
        
        print("Carregando vídeo...")
        video = VideoFileClip(input_video_path)
        
        print("Criando clips de legendas...")
        subtitle_clips = create_subtitle_clips(
            segments,
            fontsize=font_size,
            font=font,
            color=text_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width
        )
        
        print("Combinando vídeo com legendas...")
        final_video = CompositeVideoClip([video] + subtitle_clips)
        final_video = final_video.set_duration(video.duration)
        final_video = final_video.set_audio(video.audio)
        
        print("Salvando vídeo com legendas...")
        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            remove_temp=True,
            write_logfile=False,
            threads=8
        )
        
        # Limpar recursos
        video.close()
        final_video.close()
        for clip in subtitle_clips:
            clip.close()
        
        print("Legendas adicionadas com sucesso!")
        return output_video_path
        
    except Exception as e:
        print(f"Erro ao adicionar legendas: {str(e)}")
        return None


async def main():
    try:
        # Setup directories and get paths
        paths = setup_directories()
        print(f"Created run directory: {paths['run']}")
        
        # Process arguments
        args = parse_arguments()
        
        # Check for existing script in temp directory
        script_path = os.path.join(paths['temp'], 'script.json')
        script_data = None
        
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                print("Roteiro existente encontrado! Reutilizando...")
                
                # Check if data directory is empty
                data_files = [f for f in os.listdir(paths['data']) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
                
                if not data_files:
                    print("Pasta 'data' vazia. Gerando imagens com Stable Diffusion...")
                    for i, scene in enumerate(script_data['scenes'], 1):
                        print(f"Gerando imagem para cena {i}...")
                        image_filename = f"{i:02d}.png"
                        image_path = os.path.join(paths['data'], image_filename)
                        video_filename = f"{i:02d}.mp4"
                        video_path = os.path.join(paths['data'], video_filename)

                        # Generate image with Stable Diffusion
                        success = await generate_image(
                            scene['image_prompt'],
                            image_filename,
                            scene.get('negative_prompt', ''),
                            paths['data']  # Pass data directory path
                        )

                        if success:
                            print(f"Imagem {image_filename} gerada com sucesso!")
                            
                            # Convert image to video
                            image_to_video_clip(image_path, video_path, script_data['scenes'][i-1]['duration'])
                            print(f"Vídeo {video_filename} criado com sucesso a partir da imagem!")
                        else:
                            print(f"Falha ao gerar imagem {image_filename}")
                
            except json.JSONDecodeError:
                print("Arquivo de roteiro existente está corrompido. Gerando novo roteiro...")
                script_data = None
            except Exception as e:
                print(f"Erro ao ler roteiro existente: {str(e)}")
                script_data = None
        
        # Generate new script if needed
        if script_data is None and args.prompt:
            print("Gerando novo roteiro do vídeo...")
            script_data = await generate_script(args.prompt, paths['temp'], args.model, args.duration)
            
            if script_data:
                print("Roteiro gerado com sucesso! Verifique o arquivo script.json")
                print("Gerando imagens com Stable Diffusion...")
                
                for i, scene in enumerate(script_data['scenes'], 1):
                    print(f"Gerando imagem para cena {i}...")
                    image_filename = f"{i:02d}.png"
                    image_path = os.path.join(paths['data'], image_filename)
                    video_filename = f"{i:02d}.mp4"
                    video_path = os.path.join(paths['data'], video_filename)

                    success = await generate_image(
                        scene['image_prompt'],
                        image_filename,
                        scene.get('negative_prompt', ''),
                        paths['data']
                    )

                    if success:
                        print(f"Imagem {image_filename} gerada com sucesso!")
                        image_to_video_clip(image_path, video_path, script_data['scenes'][i-1]['duration'])
                        print(f"Vídeo {video_filename} criado com sucesso a partir da imagem!")
                    else:
                        print(f"Falha ao gerar imagem {image_filename}")
                
                print("Todas as imagens foram convertidas em vídeos!")
        
        elif script_data is None and not args.prompt:
            print("Nenhum roteiro encontrado e nenhum prompt fornecido. Use --prompt para gerar um novo roteiro.")
            return
        
        print("Iniciando a criação do vídeo final...")
        # Update output path to use run directory
        if not args.output.startswith(os.path.sep):  # If not absolute path
            args.output = os.path.join(paths['run'], args.output)
        await create_video_async(args, paths['run'], script_data)
        
        
        # script_path = os.path.join('C://Users//Andre//Desktop//Projetos//auto_tube//run_13-11-2024-01_11_47//temp', 'script.json')
        # with open(script_path, 'r', encoding='utf-8') as f:
        #               script_data = json.load(f)
        # await create_video_async(args, 'C://Users//Andre//Desktop//Projetos//auto_tube//run_13-11-2024-01_11_47', script_data)
        print("Vídeo final criado com sucesso!")
        
        if enable_sub:
            # Adicionar legendas ao vídeo final
            input_video = args.output
            output_video = os.path.splitext(input_video)[0] + '_subtitled' + os.path.splitext(input_video)[1]
            
            subtitled_video = await add_subtitles_to_video(
                input_video_path=input_video,
                output_video_path=output_video,
                whisper_model_size="large-v3-turbo",  # você pode ajustar o tamanho do modelo
                language="pt"  # ou outro idioma conforme necessário
            )
            
            if subtitled_video:
                print(f"Vídeo com legendas salvo em: {subtitled_video}")
            else:
                print("Falha ao adicionar legendas ao vídeo")
        
        # Buscar e adicionar trilha sonora aleatória se existir
        final_video = VideoFileClip(output_video)
        video_duration = final_video.duration
        
        trilhas_disponiveis = glob(os.path.join('music', 'trilha_*.mp3'))
        if trilhas_disponiveis:
            soundtrack_path = random.choice(trilhas_disponiveis)
            print(f"Trilha selecionada: {os.path.basename(soundtrack_path)}")
            try:
                soundtrack = AudioFileClip(soundtrack_path)
                soundtrack_duration = soundtrack.duration
        
                # Se a trilha for menor que o vídeo, criar um loop
                if soundtrack_duration < video_duration:
                    repeticoes = int(np.ceil(video_duration / soundtrack_duration))
                    soundtrack_final = concatenate_audioclips([soundtrack] * repeticoes)
                    soundtrack_final = soundtrack_final.subclip(0, video_duration)
                else:
                    soundtrack_final = soundtrack.subclip(0, video_duration)
        
                # Aplica a trilha sonora
                if final_video.audio is not None:
                    final_audio = CompositeAudioClip([
                        final_video.audio,
                        soundtrack_final.volumex(0.2)
                    ])
                    final_video = final_video.set_audio(final_audio)
                else:
                    final_video = final_video.set_audio(soundtrack_final)
                
                # Define o caminho do vídeo final com trilha sonora
                output_with_soundtrack = os.path.splitext(output_video)[0] + '_with_soundtrack' + os.path.splitext(output_video)[1]
                
                # Salva o vídeo final com a trilha sonora
                final_video.write_videofile(
                    output_with_soundtrack,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                print(f"Vídeo com trilha sonora salvo em: {output_with_soundtrack}")
                
                # Fecha os clips para liberar recursos
                final_video.close()
                soundtrack.close()
                soundtrack_final.close()
                
            except Exception as e:
                print(f"Erro ao adicionar trilha sonora: {str(e)}")
                # Fecha os clips mesmo em caso de erro
                if 'final_video' in locals(): final_video.close()
                if 'soundtrack' in locals(): soundtrack.close()
                if 'soundtrack_final' in locals(): soundtrack_final.close()
        else:
            print("Nenhuma trilha sonora encontrada na pasta 'music'")
        
        
        # Gerar elementos SEO para o vídeo
        print("\nGerando elementos SEO para o vídeo...")
        try:
            seo_elements = generate_video_seo(script_data)
            
            # Salvar elementos SEO em um arquivo
            seo_output_path = os.path.join(paths['run'], 'video_seo.txt')
            with open(seo_output_path, 'w', encoding='utf-8') as f:
                f.write("=== TÍTULO ===\n")
                f.write(f"{seo_elements['title']}\n\n")
                f.write("=== DESCRIÇÃO ===\n")
                f.write(f"{seo_elements['description']}\n\n")
                # f.write("=== HASHTAGS ===\n")
                # f.write(f"{seo_elements['hashtags']}\n")
                
            print(f"\nElementos SEO salvos em: {seo_output_path}")
            
            # Opcional: Salvar elementos SEO também em formato JSON
            seo_json_path = os.path.join(paths['run'], 'video_seo.json')
            with open(seo_json_path, 'w', encoding='utf-8') as f:
                json.dump(seo_elements, f, ensure_ascii=False, indent=2)
            print(f"Elementos SEO também salvos em JSON: {seo_json_path}")
            
        except Exception as e:
            print(f"Aviso: Não foi possível gerar elementos SEO: {str(e)}")
            
        # Salvar argumentos da execução
        args_output_path = os.path.join(paths['run'], 'execution_args.txt')
        with open(args_output_path, 'w', encoding='utf-8') as f:
            f.write("python " + sys.argv[0] + "\n")  # Nome do script
            
            # Converter os argumentos em string de comando
            args_dict = vars(args)  # Converte os argumentos em dicionário
            for arg, value in args_dict.items():
                if isinstance(value, bool) and value:  # Para argumentos tipo --zoom e --subs
                    f.write(f"--{arg} \n")
                elif not isinstance(value, bool):  # Para outros argumentos com valores
                    if arg == 'prompt':  # Coloca o prompt entre aspas
                        f.write(f'--{arg} "{value}"\n')
                    else:
                        f.write(f"--{arg} {value}\n")
                        
        print(f"Argumentos da execução salvos em: {args_output_path}")
        
            
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

import argparse
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, ImageClip, CompositeVideoClip, CompositeAudioClip, VideoClip, TextClip, concatenate_videoclips
import asyncio
import edge_tts
import srt
from datetime import timedelta
from ollama import Client
import re
import json
import shutil
from pathlib import Path
import requests
import subprocess
from moviepy.editor import concatenate_audioclips
import math
import random
from moviepy.video.fx.all import resize
import moviepy.video.fx.all as vfx
import cv2
import datetime
import traceback
import whisper
from moviepy.video.tools.subtitles import SubtitlesClip
from typing import Optional, Dict, List
from moviepy.config import change_settings
from glob import glob
from collections import Counter
# import nest_asyncio
# nest_asyncio.apply()
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

def parse_arguments():
    """Configura e processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Gerador automático de vídeos com narração usando LLM.')
    
    parser.add_argument('--prompt', type=str, help='Prompt inicial para o LLM gerar o roteiro do vídeo')
    parser.add_argument('--model', type=str, default='gemma2:9b', help='Modelo do Ollama a ser usado (default: gemma2:9b)')
    parser.add_argument('--output', type=str, default='video_final.mp4', help='Nome do arquivo de vídeo de saída')
    parser.add_argument('--duration', type=int, default=60, help='Duração desejada do vídeo em segundos (default: 120)')
    parser.add_argument('--sd_steps', type=int, default=40, help='Número de steps para o Stable Diffusion (default: 30)')
    parser.add_argument('--v_width', type=int, default=720, help='largura do vídeo (default: 720)')
    parser.add_argument('--v_height', type=int, default=1280, help='Altura do vídeo (default: 1280)')
    parser.add_argument('--zoom', type=bool, default=False, help='Modo zoom (default: False)')
    parser.add_argument('--subs', type=bool, default=True, help='Habilita legendas (default: True)')
    
    return parser.parse_args()

#%% Configurações

args = parse_arguments()
stable_difusion_steps = args.sd_steps
v_width = args.v_width
v_height = args.v_height
v_zoom = args.zoom
v_subs = args.subs
enable_sub = v_subs
enable_zoom = v_zoom
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
        sd_model = "realisticVisionV60B1_v51HyperVAE"
        
        # Prompt base melhorado com ênfase em conteúdo adequado para TikTok
        safe_style = (
            "suitable for social media, family friendly, PG content, "
            "modest clothing, fully clothed, appropriate attire, "
            "professional appearance, social media friendly, "
            "tiktok style, trending on tiktok"
        )
        
        quality_style = (
            "masterpiece, best quality, highly detailed, realistic anatomy, "
            "perfect hands, accurate facial features, precise body proportions, "
            "cinematic lighting, dramatic composition, high production value, "
            "viral worthy, social media ready, sharp focus, 8k uhd"
        )
        
        enhanced_prompt = f"{prompt}, {safe_style}, {quality_style}"
        
        # Negative prompt otimizado para conteúdo seguro
        base_negative = negative_prompt if negative_prompt else ""
        
        safety_negative = """
            (nsfw:1.5), (nudity:1.5), (naked:1.5), (nude:1.5), (explicit:1.5),
            (suggestive:1.4), (revealing clothes:1.4), (lingerie:1.4),
            (inappropriate:1.5), (sexual:1.5), (adult content:1.5),
            (underwear:1.4), (bikini:1.3), (swimsuit:1.3),
            (cleavage:1.4), (beach wear:1.3)
        """
        
        anatomical_negative = """
            (deformed:1.3), (distorted:1.3), (disfigured:1.3),
            (bad anatomy:1.4), (wrong anatomy:1.4), (extra limbs:1.4),
            (missing limbs:1.4), (floating limbs:1.4), (disconnected limbs:1.4),
            (malformed hands:1.4), (extra fingers:1.4), (missing fingers:1.4),
            (fused fingers:1.4), (too many fingers:1.4), (mutated hands:1.4),
            (bad hands:1.4), (poorly drawn hands:1.4),
            (malformed face:1.4), (poorly drawn face:1.4),
            (bad proportions:1.4), (gross proportions:1.4),
            (mutation:1.3), (mutated:1.3)
        """
        
        quality_negative = """
            lowres, bad quality, normal quality, jpeg artifacts,
            blurry, blur, poorly drawn, bad art, text, watermark,
            signature, out of frame, cropped, worst quality,
            low quality, normal quality, cartoon, anime, illustration
        """
        
        enhanced_negative = f"{base_negative}, {safety_negative}, {anatomical_negative}, {quality_negative}"
        
        # Parâmetros otimizados baseados no número de steps
        if stable_difusion_steps >= 120:
            payload = {
                "prompt": enhanced_prompt,
                "negative_prompt": enhanced_negative,
                "steps": stable_difusion_steps,
                "width": v_width,
                "height": v_height,
                "sampler_name": "DPM++ SDE Karras",
                "cfg_scale": 8.5,  # Aumentado para maior adesão ao prompt
                "denoising_strength": 0.65,
                "enable_hr": True,
                "hr_scale": 1.5,
                "hr_upscaler": "R-ESRGAN 4x+",
                "hr_second_pass_steps": 20,
                "override_settings": {
                    "sd_model_checkpoint": sd_model
                },
                "override_settings_restore_afterwards": True
            }
        else:
            payload = {
                "prompt": enhanced_prompt,
                "negative_prompt": enhanced_negative,
                "steps": stable_difusion_steps,
                "width": v_width,
                "height": v_height,
                "sampler_name": "DPM++ 2M Karras",
                "cfg_scale": 8.5,
                "override_settings": {
                    "sd_model_checkpoint": sd_model
                },
                "override_settings_restore_afterwards": True
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
                ('Sharpness', 1.2),
                ('Contrast', 1.1),
                ('Color', 0.95),
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
    center_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    
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
                zoom_factor=0.08
            )
        else:
            moving_clip = image_clip
        
        if enable_zoom:
            # Configurações de alta qualidade
            moving_clip.write_videofile(
                output_video,
                fps=30,  
                codec='libx264',
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
                codec='libx264',
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
            # Carregar e reutilizar o roteiro existente
            with open(root_script_path, 'r', encoding='utf-8') as f:
                script = json.load(f)
            print("Roteiro existente encontrado no diretório raiz! Reutilizando...")
            return script
    except:
        print("Roteiro existente não encontrado no diretório raiz...")
        pass
    
    
    try:
        
        # Calcular número mínimo de cenas necessário
        min_scene_duration = 7  # duração mínima por cena
        max_scene_duration = 15  # duração máxima por cena
        min_scenes_needed = math.ceil(duration / max_scene_duration)
        max_scenes_needed = math.floor(duration / min_scene_duration)
        target_scenes = (min_scenes_needed + max_scenes_needed) // 2  # média para um bom equilíbrio
        
        system_prompt = f"""Você é um assistente especializado em criar roteiros para vídeos e otimização SEO.
        Analise o prompt do usuário e gere um roteiro estruturado para um vídeo de {duration} segundos.
        
        IMPORTANTE: O vídeo deve ter aproximadamente {target_scenes} cenas para cobrir a duração total de {duration} segundos.
        Cada cena deve ter entre 7 e 15 segundos, e a soma total das durações deve ser próxima a {duration} segundos.
        
        O roteiro deve ser retornado em formato JSON com a seguinte estrutura:
        {{
            "scenes": [
                {{
                    "description": "Descrição detalhada da cena para geração de imagem com IA",
                    "image_prompt": "Prompt otimizado para Stable Diffusion gerar a imagem. Este prompt precisa ser em inglês e deve conter pelo menos 15 palavras. Deve ser coerente com a descrição da cena e o roteiro.",
                    "negative_prompt": "Prompt negativo para Stable Diffusion",
                    "narration": "Texto que deve ser narrado durante esta cena, composto por 3 ou 4 frases. O texto a ser narrado, precisa ser em português.",
                    "duration": "Duração sugerida em segundos, baseada no tamanho do texto que deve ser narrado na cena"
                }}
            ],
            "seo": {{
                "title": "Título otimizado para SEO, entre 50-60 caracteres, incluindo palavras-chave principais",
                "description": "Descrição atraente do vídeo entre 120-160 caracteres, incluindo call-to-action e palavras-chave principais",
                "tags": ["lista", "de", "10-15", "hashtags", "relevantes", "sem", "o", "símbolo", "#"],
                "keywords": ["5-7", "palavras-chave", "principais", "do", "conteúdo"]
            }}
        }}
        
        Para os prompts de imagem, seja específico e inclua detalhes como:
        - Estilo artístico
        - Composição
        - Iluminação
        - Cores
        - Perspectiva
        
        Para os elementos SEO:
        - O título deve ser atraente e otimizado para busca
        - A descrição deve informar e engajar o espectador
        - As tags devem cobrir tópicos relevantes e relacionados
        - As palavras-chave devem refletir os principais temas do conteúdo
        
        LEMBRE-SE: A soma das durações de todas as cenas deve ser igual a {duration} segundos.
        Distribua o conteúdo de forma equilibrada ao longo do vídeo.
        """
        
        full_prompt = f"{system_prompt}\n\nPrompt do usuário: {prompt}"
        
        response = ollama_client.generate(
            model=model,
            prompt=full_prompt,
            stream=False
        )
        
        json_str = re.search(r'\{[\s\S]*\}', response['response']).group()
        script = json.loads(json_str)
        
        # Salvar script.json na pasta temp
        with open(os.path.join(path, 'script.json'), 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        
        # Salvar arquivos de texto na pasta temp
        for i, scene in enumerate(script['scenes'], 1):
            with open(os.path.join(path, f'{i:02d}.txt'), 'w', encoding='utf-8') as f:
                f.write(scene['narration'])
            with open(os.path.join(path, f'{i:02d}_description.txt'), 'w', encoding='utf-8') as f:
                f.write(scene['description'])
        
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
        'description': seo.get('description', ''),
        'hashtags': ' '.join(f'#{tag}' for tag in seo.get('tags', []))
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
    communicate = edge_tts.Communicate(text, VOICE, pitch="-2Hz", rate="+15%")
    await communicate.save(filename)

def text_to_speech(text, filename):
    asyncio.run(text_to_speech_edge(text, filename))

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
        await text_to_speech_edge(text, audio_path)

    # Processar vídeo
    if file_extension == '.mp4':
        video_clip = VideoFileClip(file_path)
        
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            video_duration = video_clip.duration
            
            # Se o vídeo for mais curto que o áudio, estender sua duração
            if video_duration < audio_duration:
                #video_clip = video_clip.loop(duration=audio_duration)
                
                frame_t = video_clip.duration - 1
                ext_clip = video_clip.to_ImageClip(t=frame_t, duration=(audio_duration - video_duration))
                video_clip = concatenate_videoclips([video_clip, ext_clip])
            
            # Se o vídeo for mais longo que o áudio, manter o restante em silêncio
            silent_duration = video_duration - audio_duration if video_duration > audio_duration else 0
            
            if silent_duration > 0:
                from moviepy.audio.AudioClip import AudioClip
                silence = AudioClip(lambda t: 0, duration=silent_duration)
                final_audio = concatenate_audioclips([audio_clip, silence])
                video_clip = video_clip.set_audio(final_audio)
            else:
                video_clip = video_clip.set_audio(audio_clip)

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
        
        # Ajustar a duração do vídeo se necessário
        if args.duration and final_video.duration != args.duration:
            if final_video.duration < args.duration:
                last_clip = clips[-1]
                extension_duration = args.duration - final_video.duration
                extended_last_clip = last_clip.set_duration(last_clip.duration + extension_duration)
                clips[-1] = extended_last_clip
                final_video = concatenate_videoclips(clips, method="compose")
            else:
                final_video = final_video

        # Buscar e adicionar trilha sonora aleatória se existir
        trilhas_disponiveis = glob(os.path.join('music', 'trilha_*.mp3'))
        if trilhas_disponiveis:
            soundtrack_path = random.choice(trilhas_disponiveis)
            print(f"Trilha selecionada: {os.path.basename(soundtrack_path)}")
            try:
                soundtrack = AudioFileClip(soundtrack_path)
                if soundtrack.duration > final_video.duration:
                    soundtrack = soundtrack.subclip(0, final_video.duration)
                else:
                    soundtrack = soundtrack.loop(duration=final_video.duration)
                
                if final_video.audio is not None:
                    final_audio = CompositeAudioClip([
                        final_video.audio,
                        soundtrack.volumex(0.2)
                    ])
                    final_video = final_video.set_audio(final_audio)
                else:
                    final_video = final_video.set_audio(soundtrack)
            except Exception as e:
                print(f"Erro ao adicionar trilha sonora: {str(e)}")

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

def clean_text(text: str) -> str:
    """
    Remove pontuação indesejada e limpa o texto da legenda
    """
    # Remove pontos e vírgulas, mas mantém !?
    text = text.replace(',', '').replace('.', '')
    # Remove espaços extras
    text = ' '.join(text.split())
    return text

def convert_whisper_segments(segments: List[Dict]) -> List[tuple]:
    """
    Converte os segmentos do Whisper para o formato ((start_time, end_time), text)
    """
    return [(segment['start'], segment['end'], clean_text(segment['text'])) for segment in segments]


def create_subtitle_clips(segments: List[tuple], 
                        fontsize: int = 50, 
                        font: str = 'Agency-FB-Negrito', 
                        color: str = 'white', 
                        stroke_color: str = 'black',
                        stroke_width: int = 2) -> List[TextClip]:
    """
    Cria clips de texto para cada segmento de legenda
    """
    subtitle_clips = []
    
    for start, end, text in segments:
        duration = end - start
        
        # Se o vídeo for vertical 9:16, centraliza a legenda no centro do vídeo
        if v_width == 720:
        
            # Criar TextClip com fundo transparente (modo RGBA)
            txt_clip = TextClip(
                text,
                fontsize=fontsize,
                font=font,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                method='caption',
                size=(720, 0),  # Largura máxima do texto, altura automática
                transparent=True
                #kerning=-1,  # Ajuste fino do espaçamento entre letras
                # interline=-1  # Ajuste fino do espaçamento entre linhas
            ).set_position(('center')).set_duration(duration).set_start(start)
            
            subtitle_clips.append(txt_clip)
            
        else:
            
            # Criar TextClip com fundo transparente (modo RGBA)
            txt_clip = TextClip(
                text,
                fontsize=fontsize,
                font=font,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                method='caption',
                size=(720, None),  # Largura máxima do texto, altura automática
                transparent=True
                #kerning=-1,  # Ajuste fino do espaçamento entre letras
                # interline=-1  # Ajuste fino do espaçamento entre linhas
            ).set_position(('center', 'bottom')).set_duration(duration).set_start(start)
            
            subtitle_clips.append(txt_clip)
    
    return subtitle_clips

async def add_subtitles_to_video(
    input_video_path: str,
    output_video_path: str,
    whisper_model_size: str = "medium",
    language: str = "pt",
    font_size: int = 50,
    font: str = "impact",
    text_color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 2
) -> Optional[str]:
    """
    Adiciona legendas ao vídeo usando Whisper para transcrição, mantendo o áudio original.
    
    Args:
        input_video_path: Caminho do vídeo de entrada
        output_video_path: Caminho para salvar o vídeo com legendas
        whisper_model_size: Tamanho do modelo Whisper ("tiny", "base", "small", "medium", "large")
        language: Código do idioma para transcrição
        font_size: Tamanho da fonte das legendas
        font: Nome da fonte a ser usada
        text_color: Cor do texto das legendas
        stroke_color: Cor da borda do texto
        stroke_width: Espessura da borda do texto
    
    Returns:
        str: Caminho do vídeo com legendas ou None em caso de erro
    """
    try:
        # Carregar o modelo Whisper
        print("Carregando modelo Whisper...")
        model = whisper.load_model(whisper_model_size)
        
        # Transcrever o áudio do vídeo
        print("Transcrevendo áudio...")
        result = model.transcribe(
            input_video_path,
            language=language,
            verbose=False
        )
        
        # Converter segmentos do Whisper para o formato de legendas
        print("Processando segmentos de legendas...")
        segments = convert_whisper_segments(
            result["segments"]
        )
        
        # Carregar o vídeo original
        print("Carregando vídeo...")
        video = VideoFileClip(input_video_path)
        
        # Criar clips de legendas
        print("Criando clips de legendas...")
        subtitle_clips = create_subtitle_clips(
            segments,
            fontsize=font_size,
            font=font,
            color=text_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width
        )
        
        # Combinar vídeo com legendas
        print("Combinando vídeo com legendas...")
        final_video = CompositeVideoClip(
            [video] + subtitle_clips,
            use_bgclip=True
        ).set_duration(video.duration)  # Garantir que a duração seja a mesma do vídeo original
        
        # Copiar o áudio original
        final_video = final_video.set_audio(video.audio)
        
        # Salvar o vídeo final
        print("Salvando vídeo com legendas...")
        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            remove_temp=True,
            write_logfile=False,
            threads=4  # Adiciona threads para melhorar performance
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
        print("Vídeo final criado com sucesso!")
        
        if enable_sub:
            # Adicionar legendas ao vídeo final
            input_video = args.output
            output_video = os.path.splitext(input_video)[0] + '_subtitled' + os.path.splitext(input_video)[1]
            
            subtitled_video = await add_subtitles_to_video(
                input_video_path=input_video,
                output_video_path=output_video,
                whisper_model_size="medium",  # você pode ajustar o tamanho do modelo
                language="pt"  # ou outro idioma conforme necessário
            )
            
            if subtitled_video:
                print(f"Vídeo com legendas salvo em: {subtitled_video}")
            else:
                print("Falha ao adicionar legendas ao vídeo")
        
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
                f.write("=== HASHTAGS ===\n")
                f.write(f"{seo_elements['hashtags']}\n")
                
            print(f"\nElementos SEO salvos em: {seo_output_path}")
            
            # Opcional: Salvar elementos SEO também em formato JSON
            seo_json_path = os.path.join(paths['run'], 'video_seo.json')
            with open(seo_json_path, 'w', encoding='utf-8') as f:
                json.dump(seo_elements, f, ensure_ascii=False, indent=2)
            print(f"Elementos SEO também salvos em JSON: {seo_json_path}")
            
        except Exception as e:
            print(f"Aviso: Não foi possível gerar elementos SEO: {str(e)}")
            
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import json
from datetime import datetime
import numpy as np
import logging
import traceback
import re
import sys
import wave
import io
import base64
from typing import List, Dict, Optional
import uuid
import subprocess
import tempfile
import time

os.chdir("/home/zentek/Documents/GPT-SoVITS")
GPT_SOVITS_PATH = "/home/zentek/Documents/GPT-SoVITS/GPT_SoVITS"
AUDIO2FACE_SAMPLES_PATH = "/home/zentek/Documents/Audio2Face-3D-Samples"
AUDIO_DIR = "/home/zentek/Documents/shared"
BLENDSHAPE_DIR = "/home/zentek/Documents/blendshape"
TEMP_DIR = "/home/zentek/Documents/temp"

# 确保路径在sys.path中
for path in [GPT_SOVITS_PATH, AUDIO2FACE_SAMPLES_PATH]:
    if path not in sys.path:
        sys.path.append(path)

sys.path.append("/home/zentek/Documents/GPT-SoVITS")

# 导入模块
from chat_digital_human_lib import (
    load_gptsovits_model, 
    get_reference_audio_path, 
    get_ai_response,
    text_to_speech_optimized
)
from audio2face3d_client import Audio2Face3DClient

# 导入Riva客户端 (用于ASR)
import riva.client

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TalkingAvatarBackend')

app = FastAPI()

# 允许CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保目录存在
for directory in [AUDIO_DIR, BLENDSHAPE_DIR, TEMP_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 挂载静态文件目录
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/blendshape", StaticFiles(directory=BLENDSHAPE_DIR), name="blendshape")

# 全局变量来存储模型
gptsovits_model = None
audio2face_client = None
reference_audio_path = None
riva_asr = None

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket客户端 {client_id} 已连接")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket客户端 {client_id} 已断开连接")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for client_id, connection in self.active_connections.items():
            await connection.send_json(message)

manager = ConnectionManager()

# 数据模型
class TalkRequest(BaseModel):
    text: str
    client_id: Optional[str] = None

# 连接状态请求
class StatusRequest(BaseModel):
    client_id: str

# 音频识别请求
class SpeechRecognitionRequest(BaseModel):
    audio_data: str
    audio_format: Optional[str] = "audio/webm"
    client_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global gptsovits_model, audio2face_client, reference_audio_path, riva_asr
    
    # 加载GPT-SoVITS模型
    logger.info("开始加载GPT-SoVITS模型")
    gptsovits_model = load_gptsovits_model()
    if gptsovits_model:
        logger.info("GPT-SoVITS模型加载成功")
        # 预加载参考音频路径
        reference_audio_path = get_reference_audio_path(AUDIO_DIR)
        if reference_audio_path is not None:
            logger.info("参考音频路径加载成功")
        else:
            logger.error("参考音频路径加载失败")
    else:
        logger.error("GPT-SoVITS模型加载失败")
    
    # 初始化Audio2Face客户端
    logger.info("开始初始化Audio2Face客户端")
    audio2face_client = Audio2Face3DClient(
        server_address="172.16.10.158:52000",
        max_buffer_seconds=10.0
    )
    logger.info("Audio2Face客户端初始化完成")
    
    # 初始化Riva ASR客户端
    logger.info("开始初始化Riva ASR客户端")
    try:
        auth = riva.client.Auth(uri='172.16.10.158:50051')
        riva_asr = riva.client.ASRService(auth)
        logger.info("Riva ASR客户端初始化成功")
    except Exception as e:
        logger.error(f"Riva ASR客户端初始化失败: {e}")
        riva_asr = None

# 优化: 并行处理AI响应和音频生成
async def process_ai_response(text, client_id):
    global gptsovits_model, audio2face_client, reference_audio_path
    
    start_time = time.time()
    
    # 1. 获取AI响应
    ai_response_task = asyncio.create_task(get_ai_response(text))
    
    # 通知前端正在处理
    if client_id:
        await manager.send_message(client_id, {
            "type": "processing_status",
            "status": "getting_ai_response",
            "message": "正在获取AI响应..."
        })
    
    # 等待AI响应
    ai_response = await ai_response_task
    ai_time = time.time() - start_time
    logger.info(f"AI响应耗时: {ai_time:.2f}秒")
    
    # 通知前端AI响应已准备好
    if client_id:
        await manager.send_message(client_id, {
            "type": "ai_response",
            "text": ai_response
        })
    
    # 2. 启动TTS和Blendshape生成 - 并行处理
    # 开始语音生成
    if client_id:
        await manager.send_message(client_id, {
            "type": "processing_status",
            "status": "generating_speech",
            "message": "正在生成语音..."
        })
    
    # 2.1 启动TTS任务
    tts_start = time.time()
    tts_task = asyncio.create_task(
        text_to_speech_optimized(ai_response, gptsovits_model, reference_audio_path, AUDIO_DIR)
    )
    
    # 等待TTS完成
    audio_samples, audio_file_path = await tts_task
    tts_time = time.time() - tts_start
    
    if not audio_file_path:
        logger.error("语音合成失败")
        return None, None, ai_response
    
    logger.info(f"语音合成成功: {audio_file_path}, 耗时: {tts_time:.2f}秒")
    
    # 2.2 同时开始生成Blendshape数据 (与TTS并行)
    if client_id:
        await manager.send_message(client_id, {
            "type": "processing_status",
            "status": "generating_blendshape",
            "message": "正在生成面部动画..."
        })
    
    # 立即开始处理Blendshape (与TTS并行)
    blendshape_start = time.time()
    blendshape_task = asyncio.create_task(
        process_audio_to_blendshape(audio_file_path, audio2face_client)
    )
    
    # 通知前端音频准备好了
    if client_id:
        await manager.send_message(client_id, {
            "type": "audio_ready",
            "path": f"/audio/{os.path.basename(audio_file_path)}"
        })
    
    # 等待Blendshape处理完成
    csv_path = await blendshape_task
    blendshape_time = time.time() - blendshape_start
    
    if not csv_path:
        logger.error("Blendshape数据生成失败")
        return audio_file_path, None, ai_response
    
    logger.info(f"Blendshape数据生成成功: {csv_path}, 耗时: {blendshape_time:.2f}秒")
    
    # 转换Blendshape数据为前端所需格式
    blend_data = convert_csv_to_blend_data(csv_path)
    
    # 记录总处理时间
    total_time = time.time() - start_time
    logger.info(f"总处理时间: {total_time:.2f}秒 (AI: {ai_time:.2f}秒, TTS: {tts_time:.2f}秒, Blendshape: {blendshape_time:.2f}秒)")
    
    return audio_file_path, blend_data, ai_response

@app.post("/talk")
async def talk(request: TalkRequest):
    global gptsovits_model, audio2face_client
    
    if not gptsovits_model or not audio2face_client:
        logger.error("模型未加载，无法处理请求")
        return {"error": "模型未加载"}
    
    try:
        logger.info(f"处理文本到语音请求: {request.text[:50]}...")
        
        # 处理AI响应和音频生成
        audio_file_path, blend_data, ai_response = await process_ai_response(
            request.text, 
            request.client_id
        )
        
        if not audio_file_path:
            return {"error": "语音合成或面部动画生成失败"}
        
        # 构建响应
        audio_filename = os.path.basename(audio_file_path)
        audio_url = f"/audio/{audio_filename}"
        
        logger.info(f"返回响应，音频URL: {audio_url}, Blendshape数据帧数: {len(blend_data) if blend_data else 0}")
        
        # 如果提供了client_id，通过WebSocket发送完成通知
        if request.client_id:
            await manager.send_message(request.client_id, {
                "type": "processing_complete",
                "blendData": blend_data,
                "filename": audio_url,
                "text": ai_response
            })
        
        return {
            "blendData": blend_data,
            "filename": audio_url,
            "text": ai_response
        }
    
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        traceback.print_exc()
        if request.client_id:
            await manager.send_message(request.client_id, {
                "type": "error",
                "message": str(e)
            })
        return {"error": str(e)}

# 优化: 快速转换音频格式
async def convert_to_wav(input_data, is_file_path=True, output_path=None):
    """将任意音频转换为WAV格式 (16kHz, 16bit, mono)"""
    try:
        if is_file_path:
            input_path = input_data
            if output_path is None:
                output_path = os.path.splitext(input_path)[0] + ".wav"
                
            # 使用ffmpeg转换音频
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-i", input_path,
                "-acodec", "pcm_s16le",  # 设置音频编码为16位PCM
                "-ac", "1",              # 设置为单声道
                "-ar", "16000",          # 设置采样率为16kHz
                output_path
            ]
        else:
            # 处理内存中的音频数据
            if output_path is None:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    output_path = temp_file.name
            
            # 创建临时输入文件
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input_path = temp_input.name
                temp_input.write(input_data)
            
            # 使用ffmpeg转换音频
            cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_input_path,
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                output_path
            ]
        
        # 执行转换命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # 清理临时文件
        if not is_file_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        
        if process.returncode != 0:
            logger.error(f"转换音频格式失败: {stderr.decode()}")
            return None
        
        logger.info(f"音频转换成功: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"转换音频格式时出错: {e}")
        return None

# 优化: ASR处理
@app.post("/recognize_speech")
async def recognize_speech(
    audio: UploadFile = File(...),
    client_id: str = Form(None),
    audio_format: str = Form("audio/webm")
):
    """处理上传的音频文件，进行语音识别，并返回识别结果"""
    global riva_asr
    
    if not riva_asr:
        logger.error("Riva ASR服务未初始化")
        return {"error": "语音识别服务未初始化"}
    
    try:
        logger.info(f"接收到语音识别请求，客户端ID: {client_id}, 音频格式: {audio_format}")
        
        start_time = time.time()
        
        # 读取上传的音频内容
        content = await audio.read()
        
        # 通知前端开始处理
        if client_id:
            await manager.send_message(client_id, {
                "type": "processing_status",
                "status": "processing_speech",
                "message": "正在处理语音..."
            })
        
        # 创建临时WAV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_wav_path = os.path.join(TEMP_DIR, f"speech_input_{timestamp}.wav")
        
        # 直接从内存转换音频
        converted_wav_path = await convert_to_wav(content, is_file_path=False, output_path=temp_wav_path)
        
        if not converted_wav_path or not os.path.exists(converted_wav_path):
            logger.error("音频格式转换失败")
            if client_id:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "音频格式转换失败，请使用不同的浏览器或设备尝试"
                })
            return {"error": "音频格式转换失败", "success": False}
        
        # 使用Riva进行语音识别
        with open(converted_wav_path, "rb") as f:
            wav_content = f.read()
        
        # 通知前端正在进行语音识别
        if client_id:
            await manager.send_message(client_id, {
                "type": "processing_status",
                "status": "speech_recognition",
                "message": "正在进行语音识别..."
            })
        
        # 配置Riva识别参数
        config = riva.client.RecognitionConfig()
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM
        config.language_code = "zh-CN"
        config.max_alternatives = 1
        config.enable_automatic_punctuation = True
        config.audio_channel_count = 1
        config.sample_rate_hertz = 16000
        
        # 进行语音识别
        response = await asyncio.to_thread(
            riva_asr.offline_recognize,
            wav_content, 
            config
        )
        
        # 提取识别结果
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            logger.info(f"语音识别结果: {transcript}")
            
            # 记录处理时间
            recognition_time = time.time() - start_time
            logger.info(f"语音识别处理时间: {recognition_time:.2f}秒")
            
            # 通过WebSocket发送结果
            if client_id:
                await manager.send_message(client_id, {
                    "type": "speech_recognition_result",
                    "text": transcript
                })
                
                # 直接创建AI处理任务
                if transcript:
                    logger.info(f"自动提交识别文本到AI处理: {transcript}")
                    talk_request = TalkRequest(text=transcript, client_id=client_id)
                    asyncio.create_task(talk(talk_request))
            
            return {"text": transcript, "success": True}
        else:
            logger.warning("语音识别没有返回结果")
            
            if client_id:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "语音识别失败，没有识别结果"
                })
                
            return {"error": "语音识别失败，没有结果", "success": False}
                
    except Exception as e:
        logger.error(f"语音识别处理错误: {e}")
        traceback.print_exc()
        
        if client_id:
            await manager.send_message(client_id, {
                "type": "error",
                "message": f"语音识别错误: {str(e)}"
            })
            
        return {"error": str(e), "success": False}
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if not client_id:
        client_id = str(uuid.uuid4())
    
    await manager.connect(websocket, client_id)
    
    try:
        # 发送欢迎消息
        await manager.send_message(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "message": "已成功连接到服务器"
        })
        
        # 发送模型状态
        await manager.send_message(client_id, {
            "type": "model_status",
            "gptsovits_loaded": gptsovits_model is not None,
            "audio2face_connected": audio2face_client is not None,
            "riva_asr_available": riva_asr is not None
        })
        
        while True:
            # 持续监听来自客户端的消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                
                if msg_type == "ping":
                    await manager.send_message(client_id, {"type": "pong"})
                elif msg_type == "talk":
                    # 启动异步任务处理请求
                    text = message.get("text", "")
                    if text:
                        # 创建后台任务来处理，避免阻塞WebSocket
                        asyncio.create_task(
                            talk(TalkRequest(text=text, client_id=client_id))
                        )
                        await manager.send_message(client_id, {
                            "type": "processing_status",
                            "status": "started",
                            "message": "开始处理请求"
                        })
                    else:
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": "请求中缺少text字段"
                        })
                elif msg_type == "speech_recognition":
                    # 处理语音识别请求
                    audio_data = message.get("audio_data", "")
                    audio_format = message.get("audio_format", "audio/webm")
                    if audio_data:
                        # 处理Base64编码的音频数据
                        asyncio.create_task(
                            process_audio_data(audio_data, audio_format, client_id)
                        )
                        await manager.send_message(client_id, {
                            "type": "processing_status",
                            "status": "started",
                            "message": "开始处理语音识别"
                        })
                    else:
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": "请求中缺少audio_data字段"
                        })
            except json.JSONDecodeError:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "无效的JSON格式"
                })
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {str(e)}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"处理消息时出错: {str(e)}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket连接错误: {str(e)}")
        manager.disconnect(client_id)

# 优化: 简化音频处理流程
async def process_audio_data(base64_audio, audio_format, client_id):
    """处理Base64编码的音频数据"""
    global riva_asr
    
    if not riva_asr:
        logger.error("Riva ASR服务未初始化")
        await manager.send_message(client_id, {
            "type": "error",
            "message": "语音识别服务未初始化"
        })
        return
    
    try:
        start_time = time.time()
        
        # 从Base64解码音频数据
        audio_data = base64.b64decode(base64_audio)
        
        # 通知前端开始处理
        await manager.send_message(client_id, {
            "type": "processing_status",
            "status": "processing_speech",
            "message": "正在处理语音..."
        })
        
        # 创建临时WAV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_wav_path = os.path.join(TEMP_DIR, f"websocket_converted_{timestamp}.wav")
        
        # 直接从内存转换音频
        converted_wav_path = await convert_to_wav(audio_data, is_file_path=False, output_path=temp_wav_path)
        
        if not converted_wav_path or not os.path.exists(converted_wav_path):
            logger.error("音频格式转换失败")
            await manager.send_message(client_id, {
                "type": "error",
                "message": "音频格式转换失败，请使用不同的浏览器或设备尝试"
            })
            return
        
        # 使用Riva进行语音识别
        with open(converted_wav_path, "rb") as f:
            wav_content = f.read()
        
        # 通知前端正在进行语音识别
        await manager.send_message(client_id, {
            "type": "processing_status",
            "status": "speech_recognition",
            "message": "正在进行语音识别..."
        })
        
        # 配置Riva识别参数
        config = riva.client.RecognitionConfig()
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM
        config.language_code = "zh-CN"
        config.max_alternatives = 1
        config.enable_automatic_punctuation = True
        config.audio_channel_count = 1
        config.sample_rate_hertz = 16000
        
        # 进行语音识别
        response = await asyncio.to_thread(
            riva_asr.offline_recognize,
            wav_content, 
            config
        )
        
        # 提取识别结果
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            logger.info(f"语音识别结果: {transcript}")
            
            # 记录处理时间
            recognition_time = time.time() - start_time
            logger.info(f"WebSocket语音识别处理时间: {recognition_time:.2f}秒")
            
            # 发送识别结果
            await manager.send_message(client_id, {
                "type": "speech_recognition_result",
                "text": transcript
            })
            
            # 自动将识别文本提交给AI处理
            if transcript:
                logger.info(f"自动提交识别文本到AI处理: {transcript}")
                talk_request = TalkRequest(text=transcript, client_id=client_id)
                # 创建一个异步任务来处理请求
                asyncio.create_task(talk(talk_request))
        else:
            logger.warning("语音识别没有返回结果")
            await manager.send_message(client_id, {
                "type": "error",
                "message": "语音识别失败，没有识别结果"
            })
    
    except Exception as e:
        logger.error(f"处理音频数据时出错: {e}")
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"语音识别错误: {str(e)}"
        })
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")

@app.get("/status")
async def check_status(request: StatusRequest = None):
    """检查后端服务和模型状态"""
    status = {
        "service": "online",
        "gptsovits_model": gptsovits_model is not None,
        "audio2face_client": audio2face_client is not None,
        "reference_audio_path": reference_audio_path is not None,
        "riva_asr": riva_asr is not None
    }
    
    if request and request.client_id:
        # 如果提供了client_id，也通过WebSocket发送状态
        await manager.send_message(request.client_id, {
            "type": "status_update",
            "status": status
        })
    
    return status

# 处理音频生成Blendshape数据 - 优化版
async def process_audio_to_blendshape(audio_filepath, client):
    """处理音频文件生成Blendshape数据"""
    # 用音频文件名作为输出CSV文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(BLENDSHAPE_DIR, f"blendshape_{timestamp}.csv")
    
    try:
        # 获取音频长度
        with wave.open(audio_filepath, 'rb') as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            audio_length_seconds = n_frames / sample_rate
            
        logger.info(f"音频文件长度: {audio_length_seconds:.2f}秒")
        
        # 根据音频长度选择处理方法
        if audio_length_seconds > client.max_buffer_seconds:
            logger.info(f"音频超过最大长度限制 ({client.max_buffer_seconds}秒)，使用分段处理")
            # 使用长音频处理方式
            success = await asyncio.to_thread(
                client.process_long_audio_file_to_blendshape,
                audio_filepath, 
                output_csv_path
            )
        else:
            # 使用常规处理方式
            success = await asyncio.to_thread(
                client.process_audio_file_to_blendshape,
                audio_filepath, 
                output_csv_path
            )
        
        if success and os.path.exists(output_csv_path):
            # 检查CSV文件大小
           
            file_size = os.path.getsize(output_csv_path)
            logger.info(f"生成Blendshape数据文件: {output_csv_path}, 大小: {file_size} 字节")
            
            if file_size > 100:  # 文件大小应该大于100字节才算有效
                return output_csv_path
            else:
                logger.error(f"生成的CSV文件过小（{file_size} 字节），可能是空文件")
                return None
        else:
            logger.error("Blendshape数据生成失败")
            return None
    except Exception as e:
        logger.error(f"Audio2Face-3D处理错误: {e}")
        traceback.print_exc()
        return None

# 将CSV格式的Blendshape数据转换为前端所需的JSON格式 - 优化版
def convert_csv_to_blend_data(csv_path):
    """将CSV Blendshape数据转换为前端所需JSON格式"""
    try:
        # 直接使用numpy加载CSV以提高性能
        import numpy as np
        import pandas as pd
        
        # 使用pandas读取CSV (更快且内存效率高)
        df = pd.read_csv(csv_path)
        
        # 获取所有Blendshape列
        blendshape_cols = [col for col in df.columns if col not in ['frame', 'time', 'time_code']]
        
        # 提前构建转换映射
        conversion_map = get_blendshape_conversion_map()
        
        # 预分配数组大小来提高性能
        blend_data = []
        blend_data_capacity = len(df)
        blend_data = [None] * blend_data_capacity
        
        # 使用向量化操作处理数据
        for i, row in enumerate(df.itertuples()):
            # 获取时间
            time_value = getattr(row, 'time')
            
            # 创建blendshapes字典
            blendshapes = {}
            
            # 处理每个blendshape值
            for col in blendshape_cols:
                if hasattr(row, col) and pd.notna(getattr(row, col)):
                    value = float(getattr(row, col))
                    key = conversion_map.get(col, col)
                    blendshapes[key] = value
            
            # 创建帧数据
            blend_data[i] = {
                "time": float(time_value),
                "blendshapes": blendshapes
            }
        
        return blend_data
    except Exception as e:
        logger.error(f"转换Blendshape数据格式时出错: {e}")
        traceback.print_exc()
        return []

# 缓存blendshape名称转换映射
_blendshape_conversion_map = None

# 获取Blendshape名称转换映射
def get_blendshape_conversion_map():
    """获取ARKit Blendshape名称转换映射 (使用缓存)"""
    global _blendshape_conversion_map
    
    if _blendshape_conversion_map is None:
        # 创建转换映射
        _blendshape_conversion_map = {
            "EyeBlinkLeft": "eyeBlinkLeft",
            "EyeBlinkRight": "eyeBlinkRight",
            "JawOpen": "jawOpen",
            "MouthSmileLeft": "mouthSmileLeft",
            "MouthSmileRight": "mouthSmileRight",
            "BrowInnerUp": "browInnerUp",
            "MouthClose": "mouthClose",
            "MouthFunnel": "mouthFunnel",
            "MouthPucker": "mouthPucker",
            "MouthLeft": "mouthLeft",
            "MouthRight": "mouthRight",
            "MouthFrownLeft": "mouthFrownLeft",
            "MouthFrownRight": "mouthFrownRight",
            "MouthDimpleLeft": "mouthDimpleLeft",
            "MouthDimpleRight": "mouthDimpleRight",
            "MouthStretchLeft": "mouthStretchLeft",
            "MouthStretchRight": "mouthStretchRight",
            "MouthRollLower": "mouthRollLower",
            "MouthRollUpper": "mouthRollUpper",
            "MouthShrugLower": "mouthShrugLower",
            "MouthShrugUpper": "mouthShrugUpper",
            "MouthPressLeft": "mouthPressLeft",
            "MouthPressRight": "mouthPressRight",
            "MouthLowerDownLeft": "mouthLowerDownLeft",
            "MouthLowerDownRight": "mouthLowerDownRight",
            "MouthUpperUpLeft": "mouthUpperUpLeft",
            "MouthUpperUpRight": "mouthUpperUpRight",
        }
    
    return _blendshape_conversion_map

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

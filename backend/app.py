from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
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

# 设置路径
COSYVOICE_PATH = "/home/zentek/Documents/CosyVoice"
MATCHA_TTS_PATH = os.path.join(COSYVOICE_PATH, "third_party/Matcha-TTS")
AUDIO2FACE_SAMPLES_PATH = "/home/zentek/Documents/Audio2Face-3D-Samples"
AUDIO_DIR = "/home/zentek/Documents/shared"
BLENDSHAPE_DIR = "/home/zentek/Documents/blendshape"
TEMP_DIR = "/home/zentek/Documents/temp"

# 确保路径在sys.path中
for path in [COSYVOICE_PATH, MATCHA_TTS_PATH, AUDIO2FACE_SAMPLES_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# 导入你的模块
from chat_digital_human_lib import (
    load_cosyvoice_model, 
    get_consistent_reference_audio, 
    text_to_speech_cosyvoice,
    clean_text,
    get_ai_response
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
    allow_origins=["*"],  # 在生产环境中应该限制为你的前端域名
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
cosyvoice_model = None
audio2face_client = None
reference_audio = None
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
    global cosyvoice_model, audio2face_client, reference_audio, riva_asr
    
    # 加载CosyVoice模型
    logger.info("开始加载CosyVoice模型")
    cosyvoice_model = load_cosyvoice_model()
    if cosyvoice_model:
        logger.info("CosyVoice模型加载成功")
        # 预加载参考音频
        reference_audio = get_consistent_reference_audio(cosyvoice_model, AUDIO_DIR)
        # 修改这里：不要直接检查tensor，而是检查reference_audio是否为None
        if reference_audio is not None:
            logger.info("参考音频加载成功")
        else:
            logger.error("参考音频加载失败")
    else:
        logger.error("CosyVoice模型加载失败")
    
    # 初始化Audio2Face客户端
    logger.info("开始初始化Audio2Face客户端")
    audio2face_client = Audio2Face3DClient(
        server_address="172.16.10.158:52000",  # 使用你的Audio2Face服务器地址
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

@app.post("/talk")
async def talk(request: TalkRequest):
    global cosyvoice_model, audio2face_client, reference_audio
    
    if not cosyvoice_model or not audio2face_client:
        logger.error("模型未加载，无法处理请求")
        return {"error": "模型未加载"}
    
    try:
        logger.info(f"处理文本到语音请求: {request.text[:50]}...")
        
        # 获取AI响应
        ai_response = await get_ai_response(request.text)
        
        # 如果提供了client_id，通过WebSocket发送AI响应通知
        if request.client_id:
            await manager.send_message(request.client_id, {
                "type": "ai_response",
                "text": ai_response
            })
        
        # 生成语音
        audio_samples, audio_file_path = await text_to_speech_cosyvoice(
            ai_response, 
            cosyvoice_model, 
            reference_audio,
            AUDIO_DIR
        )
        
        if not audio_file_path:
            logger.error("语音合成失败")
            return {"error": "语音合成失败"}
        
        logger.info(f"语音合成成功: {audio_file_path}")
        
        # 如果提供了client_id，通过WebSocket发送语音生成通知
        if request.client_id:
            await manager.send_message(request.client_id, {
                "type": "audio_ready",
                "path": f"/audio/{os.path.basename(audio_file_path)}"
            })
        
        # 生成Blendshape数据
        csv_path = await process_audio_to_blendshape(audio_file_path, audio2face_client)
        
        if not csv_path:
            logger.error("Blendshape数据生成失败")
            return {"error": "Blendshape数据生成失败"}
        
        logger.info(f"Blendshape数据生成成功: {csv_path}")
        
        # 转换Blendshape数据为前端所需格式
        blend_data = convert_csv_to_blend_data(csv_path)
        
        # 构建响应
        audio_filename = os.path.basename(audio_file_path)
        audio_url = f"/audio/{audio_filename}"
        
        logger.info(f"返回响应，音频URL: {audio_url}, Blendshape数据帧数: {len(blend_data)}")
        
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

# 将任意音频转换为WAV格式 (使用ffmpeg)
async def convert_to_wav(input_path, output_path=None):
    """将任意音频转换为WAV格式 (16kHz, 16bit, mono)"""
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"
    
    try:
        # 使用ffmpeg转换音频为WAV格式 (16kHz, 16bit, mono)
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-i", input_path,
            "-acodec", "pcm_s16le",  # 设置音频编码为16位PCM
            "-ac", "1",              # 设置为单声道
            "-ar", "16000",          # 设置采样率为16kHz
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"转换音频格式失败: {stderr.decode()}")
            return None
        
        logger.info(f"音频转换成功: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"转换音频格式时出错: {e}")
        return None

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
        
        # 保存上传的原始音频文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_audio_path = os.path.join(TEMP_DIR, f"original_speech_{timestamp}")
        
        # 根据格式添加适当的扩展名
        if "webm" in audio_format:
            temp_audio_path += ".webm"
        elif "mp3" in audio_format:
            temp_audio_path += ".mp3"
        elif "wav" in audio_format:
            temp_audio_path += ".wav"
        else:
            temp_audio_path += ".audio"  # 默认扩展名
        
        # 读取上传的音频内容
        content = await audio.read()
        
        # 写入临时音频文件
        with open(temp_audio_path, "wb") as f:
            f.write(content)
        
        logger.info(f"临时原始音频文件已保存: {temp_audio_path}")
        
        # 转换为WAV格式 (16kHz, 16bit, mono) - Riva需要
        temp_wav_path = os.path.join(TEMP_DIR, f"speech_input_{timestamp}.wav")
        converted_wav_path = await convert_to_wav(temp_audio_path, temp_wav_path)
        
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
        
        # 配置Riva识别参数
        config = riva.client.RecognitionConfig()
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM  # 明确指定音频编码格式
        config.language_code = "zh-CN"
        config.max_alternatives = 1
        config.enable_automatic_punctuation = True
        config.audio_channel_count = 1
        config.sample_rate_hertz = 16000  # 明确指定采样率
        
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
            
            # 通过WebSocket发送结果（如果提供了client_id）
            if client_id:
                await manager.send_message(client_id, {
                    "type": "speech_recognition_result",
                    "text": transcript
                })
                
                # 直接创建AI处理任务
                if transcript:
                    logger.info(f"自动提交识别文本到AI处理: {transcript}")
                    talk_request = TalkRequest(text=transcript, client_id=client_id)
                    # 创建一个异步任务来处理请求
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
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
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
            "cosyvoice_loaded": cosyvoice_model is not None,
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
                            "type": "processing_started",
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
                            "type": "processing_started",
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
        # 从Base64解码音频数据
        audio_data = base64.b64decode(base64_audio)
        
        # 保存为临时音频文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_audio_path = os.path.join(TEMP_DIR, f"websocket_speech_{timestamp}")
        
        # 添加适当的扩展名
        if "webm" in audio_format:
            temp_audio_path += ".webm"
        elif "mp3" in audio_format:
            temp_audio_path += ".mp3"
        elif "wav" in audio_format:
            temp_audio_path += ".wav"
        else:
            temp_audio_path += ".audio"  # 默认扩展名
        
        # 将音频数据写入临时文件
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"临时原始音频文件已保存: {temp_audio_path}")
        
        # 转换为WAV格式
        temp_wav_path = os.path.join(TEMP_DIR, f"websocket_converted_{timestamp}.wav")
        converted_wav_path = await convert_to_wav(temp_audio_path, temp_wav_path)
        
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
        
        # 配置Riva识别参数
        config = riva.client.RecognitionConfig()
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM  # 明确指定音频编码格式
        config.language_code = "zh-CN"
        config.max_alternatives = 1
        config.enable_automatic_punctuation = True
        config.audio_channel_count = 1
        config.sample_rate_hertz = 16000  # 明确指定采样率
        
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
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")

@app.get("/status")
async def check_status(request: StatusRequest = None):
    """检查后端服务和模型状态"""
    status = {
        "service": "online",
        "cosyvoice_model": cosyvoice_model is not None,
        "audio2face_client": audio2face_client is not None,
        "reference_audio": reference_audio is not None,
        "riva_asr": riva_asr is not None
    }
    
    if request and request.client_id:
        # 如果提供了client_id，也通过WebSocket发送状态
        await manager.send_message(request.client_id, {
            "type": "status_update",
            "status": status
        })
    
    return status

# 处理音频生成Blendshape数据
async def process_audio_to_blendshape(audio_filepath, client):
    # 这个函数与chat_digital_human.py中的类似，但做了适应性修改
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(BLENDSHAPE_DIR, f"blendshape_{timestamp}.csv")
    
    try:
        # 获取音频长度
        import wave
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

# 将CSV格式的Blendshape数据转换为前端所需的JSON格式
def convert_csv_to_blend_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        blend_data = []
        
        # 获取所有Blendshape列
        blendshape_cols = [col for col in df.columns if col not in ['frame', 'time', 'time_code']]
        
        for _, row in df.iterrows():
            frame_data = {
                "time": float(row['time']),
                "blendshapes": {}
            }
            
            # 添加所有Blendshape值
            for col in blendshape_cols:
                # 转换列名为前端期望的格式
                key = convert_blendshape_name(col)
                # 避免NaN值
                if pd.notna(row[col]):
                    frame_data["blendshapes"][key] = float(row[col])
            
            blend_data.append(frame_data)
        
        return blend_data
    except Exception as e:
        logger.error(f"转换Blendshape数据格式时出错: {e}")
        traceback.print_exc()
        return []

# 将ARKit Blendshape名转换为前端使用的格式
def convert_blendshape_name(arkit_name):
    # 根据前端代码的需要进行名称转换
    conversion_map = {
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
    
    return conversion_map.get(arkit_name, arkit_name[0].lower() + arkit_name[1:])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

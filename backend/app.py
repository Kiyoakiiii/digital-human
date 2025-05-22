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
from typing import List, Dict, Optional, Union, Any
import uuid
import subprocess
import tempfile
import time
import queue
from threading import Thread

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
    text_to_speech_optimized,
    split_into_sentences
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
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"发送消息给客户端 {client_id} 失败: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息给客户端 {client_id} 失败: {e}")

manager = ConnectionManager()

# 数据模型
class TalkRequest(BaseModel):
    text: str
    client_id: Optional[str] = None

class StatusRequest(BaseModel):
    client_id: str

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

# 流式AI响应处理 - 基于参考代码的架构
def process_ai_stream(user_input: str, text_segments_list: list, full_response_list: list):
    """处理AI流式响应，实时分段"""
    try:
        # 获取AI流式响应
        stream_generator = get_ai_response(user_input, stream=True)
        
        # 标点符号用于分割句子
        biao_dian_2 = ["…", "~", "。", "？", "！", "?", "!"]
        biao_dian_3 = ["…", "~", "。", "？", "！", "?", "!", ",", "，"]
        
        full_response = ""
        temp_text = ""
        processed_text = ""
        first_segment = True
        segment_count = 0
        
        for chunk in stream_generator:
            try:
                # 提取流式内容
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    temp_text += content
                    temp_text = temp_text.replace("...", "…")
                    
                    # 简单文本清理 - 移除括号内容
                    cleaned_text = ""
                    skip = False
                    for char in temp_text:
                        if char in ["(", "（"]:
                            skip = True
                            continue
                        if char in [")", "）"]:
                            skip = False
                            continue
                        if not skip:
                            cleaned_text += char
                    
                    processed_text += cleaned_text
                    temp_text = ""
                    
                    # 分段逻辑
                    start = 0
                    for i in range(len(processed_text)):
                        if first_segment:
                            # 第一段用更严格的标点符号
                            if processed_text[i] in biao_dian_3 and i > 3:
                                segment = processed_text[start:i+1].strip()
                                if len(segment) > 1:
                                    text_segments_list.append(segment)
                                    logger.info(f"生成文本段落 {segment_count + 1}: {segment}")
                                    segment_count += 1
                                start = i + 1
                                first_segment = False
                        else:
                            # 后续段落用更宽松的标点符号
                            if processed_text[i] in biao_dian_2 and i - start > 6:
                                segment = processed_text[start:i+1].strip()
                                if len(segment) > 1:
                                    text_segments_list.append(segment)
                                    logger.info(f"生成文本段落 {segment_count + 1}: {segment}")
                                    segment_count += 1
                                start = i + 1
                    
                    # 保留未处理的部分
                    if len(processed_text) != 0:
                        processed_text = processed_text[start:]
                        
            except Exception as e:
                logger.error(f"处理AI流式响应时出错: {e}")
                continue
        
        # 处理最后剩余的文本
        if len(processed_text) > 1:
            text_segments_list.append(processed_text.strip())
            logger.info(f"生成最后文本段落: {processed_text.strip()}")
        
        # 添加结束标记
        text_segments_list.append("DONE_DONE")
        full_response_list.append(full_response)
        
        logger.info(f"AI响应处理完成，共生成 {len(text_segments_list) - 1} 个文本段落")
        
    except Exception as e:
        logger.error(f"处理AI流式响应出错: {e}")
        traceback.print_exc()
        text_segments_list.append("DONE_DONE")
        full_response_list.append("AI响应处理失败")

# TTS和Blendshape处理 - 基于参考代码的架构
def process_tts_and_blendshape(text_segment: str):
    """处理单个文本段落的TTS和Blendshape生成"""
    global gptsovits_model, audio2face_client, reference_audio_path
    
    if text_segment in ["…", "~", "。", "？", "！", "?", "!", ",", "，", ""]:
        return None
    
    try:
        start_time = time.time()
        
        # 使用同步版本的TTS处理
        audio_samples, audio_file_path = asyncio.run(text_to_speech_optimized(
            text_segment, 
            gptsovits_model, 
            reference_audio_path, 
            AUDIO_DIR
        ))
        
        if not audio_file_path:
            logger.error(f"语音合成失败: {text_segment}")
            return None
        
        tts_time = time.time() - start_time
        logger.info(f"TTS处理完成，耗时: {tts_time:.2f}秒")
        
        # 生成Blendshape数据
        blendshape_start = time.time()
        csv_path = process_audio_to_blendshape_sync(audio_file_path, audio2face_client)
        
        if csv_path:
            blend_data = convert_csv_to_blend_data(csv_path)
            blendshape_time = time.time() - blendshape_start
            logger.info(f"Blendshape生成完成，耗时: {blendshape_time:.2f}秒")
        else:
            blend_data = []
            logger.warning("Blendshape生成失败，使用空数据")
        
        # 获取音频文件名
        audio_filename = os.path.basename(audio_file_path)
        audio_url = f"/audio/{audio_filename}"
        
        total_time = time.time() - start_time
        logger.info(f"段落总处理时间: {total_time:.2f}秒")
        
        return {
            "audio_url": audio_url,
            "blend_data": blend_data,
            "text": text_segment,
            "processing_time": total_time
        }
        
    except Exception as e:
        logger.error(f"处理TTS和Blendshape时出错: {e}")
        traceback.print_exc()
        return None

# TTS和Blendshape队列处理器
def tts_blendshape_processor(text_segments_list: list, result_list: list):
    """TTS和Blendshape队列处理器"""
    i = 0
    while True:
        if i < len(text_segments_list):
            if text_segments_list[i] == "DONE_DONE":
                result_list.append("DONE_DONE")
                logger.info("TTS和Blendshape处理完成")
                break
                
            result = process_tts_and_blendshape(text_segments_list[i])
            result_list.append(result)
            i += 1
        time.sleep(0.05)

# 同步版本的Blendshape处理
def process_audio_to_blendshape_sync(audio_filepath, client):
    """同步处理音频文件生成Blendshape数据"""
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
            success = client.process_long_audio_file_to_blendshape(
                audio_filepath, 
                output_csv_path
            )
        else:
            success = client.process_audio_file_to_blendshape(
                audio_filepath, 
                output_csv_path
            )
        
        if success and os.path.exists(output_csv_path):
            file_size = os.path.getsize(output_csv_path)
            if file_size > 100:
                return output_csv_path
            else:
                logger.error(f"生成的CSV文件过小（{file_size} 字节）")
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
    """将CSV Blendshape数据转换为前端所需JSON格式"""
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        blendshape_cols = [col for col in df.columns if col not in ['frame', 'time', 'time_code']]
        conversion_map = get_blendshape_conversion_map()
        
        blend_data = []
        
        for i, row in enumerate(df.itertuples()):
            time_value = getattr(row, 'time')
            blendshapes = {}
            
            for col in blendshape_cols:
                if hasattr(row, col) and pd.notna(getattr(row, col)):
                    value = float(getattr(row, col))
                    key = conversion_map.get(col, col)
                    blendshapes[key] = value
            
            blend_data.append({
                "time": float(time_value),
                "blendshapes": blendshapes
            })
        
        return blend_data
    except Exception as e:
        logger.error(f"转换Blendshape数据格式时出错: {e}")
        return []

def get_blendshape_conversion_map():
    """获取ARKit Blendshape名称转换映射"""
    return {
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

# 流式处理主函数
async def stream_tts_processing(user_input: str, client_id: str, start_time: float):
    """流式TTS处理主函数"""
    try:
        text_segments_list = []
        result_list = []
        full_response_list = []
        
        # 启动AI流式处理线程
        ai_thread = Thread(target=process_ai_stream, args=(user_input, text_segments_list, full_response_list))
        ai_thread.daemon = True
        ai_thread.start()
        
        # 启动TTS和Blendshape处理线程
        tts_thread = Thread(target=tts_blendshape_processor, args=(text_segments_list, result_list))
        tts_thread.daemon = True
        tts_thread.start()
        
        # 流式发送结果给前端
        i = 0
        first_segment = True
        segment_count = 0
        
        while True:
            if i < len(result_list):
                if result_list[i] == "DONE_DONE":
                    # 发送完成消息
                    await manager.send_message(client_id, {
                        "type": "processing_complete_all",
                        "message": full_response_list[0] if full_response_list else "处理完成",
                        "total_segments": segment_count
                    })
                    break
                
                result = result_list[i]
                if result:
                    segment_count += 1
                    
                    # 发送流式音频片段
                    await manager.send_message(client_id, {
                        "type": "stream_audio_segment",
                        "segment_index": segment_count - 1,
                        "is_first_segment": first_segment,
                        "filename": result["audio_url"],
                        "blendData": result["blend_data"],
                        "text": result["text"],
                        "processing_time": result["processing_time"]
                    })
                    
                    if first_segment:
                        logger.info(f"首个片段处理完成，耗时: {time.time() - start_time:.2f}秒")
                        first_segment = False
                    
                    logger.info(f"发送片段 {segment_count}: {result['text']}")
                
                i += 1
            
            await asyncio.sleep(0.05)
            
    except Exception as e:
        logger.error(f"流式TTS处理出错: {e}")
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"流式处理失败: {str(e)}"
        })

@app.post("/talk")
async def talk(request: TalkRequest):
    """处理文本到语音的请求，使用流式处理"""
    global gptsovits_model, audio2face_client
    
    if not gptsovits_model or not audio2face_client:
        logger.error("模型未加载，无法处理请求")
        return {"error": "模型未加载"}
    
    logger.info(f"处理文本到语音请求: {request.text[:50]}...")
    
    # 启动流式处理
    if request.client_id:
        asyncio.create_task(stream_tts_processing(request.text, request.client_id, time.time()))
        return {"status": "streaming", "message": "流式处理已开始"}
    else:
        return {"error": "需要client_id进行流式处理"}

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
                
                # 直接启动流式处理
                if transcript:
                    logger.info(f"自动提交识别文本到AI处理: {transcript}")
                    asyncio.create_task(stream_tts_processing(transcript, client_id, time.time()))
            
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
                # 启动流式处理
                asyncio.create_task(stream_tts_processing(transcript, client_id, time.time()))
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
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                
                if msg_type == "ping":
                    await manager.send_message(client_id, {"type": "pong"})
                elif msg_type == "talk":
                    text = message.get("text", "")
                    if text:
                        # 启动流式处理
                        asyncio.create_task(stream_tts_processing(text, client_id, time.time()))
                        await manager.send_message(client_id, {
                            "type": "processing_status",
                            "status": "started",
                            "message": "开始流式处理"
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
                elif msg_type == "cancel_processing":
                    # 处理取消请求
                    logger.info(f"收到取消处理请求，客户端ID: {client_id}")
                    await manager.send_message(client_id, {
                        "type": "processing_status",
                        "status": "cancelled",
                        "message": "处理已取消"
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

@app.get("/status")
async def check_status(request: StatusRequest = None):
    """检查后端服务和模型状态"""
    status = {
        "service": "online",
        "gptsovits_model": gptsovits_model is not None,
        "audio2face_client": audio2face_client is not None,
        "reference_audio_path": reference_audio_path is not None,
        "riva_asr": riva_asr is not None,
        "streaming_enabled": True
    }
    
    if request and request.client_id:
        # 如果提供了client_id，也通过WebSocket发送状态
        await manager.send_message(request.client_id, {
            "type": "status_update",
            "status": status
        })
    
    return status

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# 获取系统信息
@app.get("/info")
async def get_system_info():
    """获取系统信息"""
    import psutil
    import platform
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "system": {
                "platform": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free": disk.free
            },
            "models": {
                "gptsovits_loaded": gptsovits_model is not None,
                "audio2face_connected": audio2face_client is not None,
                "riva_asr_available": riva_asr is not None
            },
            "connections": {
                "active_websocket_connections": len(manager.active_connections)
            }
        }
    except Exception as e:
        return {"error": f"无法获取系统信息: {str(e)}"}

# 测试端点
@app.get("/test")
async def test_endpoint():
    """测试端点"""
    return {
        "message": "服务器正常运行",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "websocket": "/ws/{client_id}",
            "talk": "/talk",
            "speech_recognition": "/recognize_speech",
            "status": "/status",
            "health": "/health",
            "info": "/info"
        }
    }

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"全局异常: {str(exc)}")
    traceback.print_exc()
    
    return {
        "error": "服务器内部错误",
        "message": str(exc),
        "timestamp": datetime.now().isoformat()
    }

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件处理"""
    global audio2face_client
    
    logger.info("正在关闭服务器...")
    
    # 关闭Audio2Face客户端
    if audio2face_client:
        try:
            audio2face_client.close()
            logger.info("Audio2Face客户端已关闭")
        except Exception as e:
            logger.error(f"关闭Audio2Face客户端时出错: {e}")
    
    # 断开所有WebSocket连接
    for client_id in list(manager.active_connections.keys()):
        manager.disconnect(client_id)
    
    logger.info("服务器已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

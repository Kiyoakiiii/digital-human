import asyncio
import numpy as np
import time
import os
import openai
from openai import OpenAI
from datetime import datetime
import re
import sys
import torch
import soundfile as sf
import logging
import concurrent.futures
from typing import List, Optional, Dict, Tuple, Any, Union

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ChatDigitalHumanLib')

# 添加GPT-SoVITS路径
GPT_SOVITS_PATH = "/home/zentek/Documents/GPT-SoVITS/GPT_SoVITS"
if GPT_SOVITS_PATH not in sys.path:
    sys.path.append(GPT_SOVITS_PATH)

sys.path.append("/home/zentek/Documents/GPT-SoVITS")

# 导入GPT-SoVITS相关模块
try:
    from TTS_infer_pack.TTS import TTS, TTS_Config
    gptsovits_available = True
    logger.info("成功加载GPT-SoVITS模块")
except ImportError as e:
    logger.error(f"无法导入GPT-SoVITS模块: {e}")
    gptsovits_available = False

# 配置服务器地址、API密钥和对话ID
address = "124.74.245.75:8091"
chat_id = "96c016e6047411f094cc0242ac120006"
api_key = "ragflow-MxODUyZmIwMDQ3NDExZjBhNDA1MDI0Mm"

# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url=f"http://{address}/api/v1/chats_openai/{chat_id}/",
    api_key=api_key
)

# 创建线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# 加载GPT-SoVITS模型
def load_gptsovits_model(config_path=None):
    if not gptsovits_available:
        logger.error("GPT-SoVITS模块不可用")
        return None
        
    try:
        # 如果未指定配置文件路径，使用默认路径
        if config_path is None:
            config_path = f"{GPT_SOVITS_PATH}/configs/tts_infer.yaml"
        
        # 初始化TTS配置和模型
        logger.info(f"加载GPT-SoVITS模型: {config_path}")
        tts_config = TTS_Config(config_path)
        
        # 设置设备
        tts_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {tts_config.device}")
        
        # 初始化TTS模型
        tts_model = TTS(tts_config)
        
        logger.info("GPT-SoVITS模型加载成功")
        return tts_model
    except Exception as e:
        logger.error(f"GPT-SoVITS模型加载失败: {e}")
        return None

# 获取一致的参考音频以确保相同的声音音调
def get_reference_audio_path(audio_dir):
    # 创建固定的参考音频路径
    ref_path = os.path.join(audio_dir, "reference_voice.wav")
    
    if not os.path.exists(ref_path):
        logger.warning(f"参考音频不存在: {ref_path}")
        return None
        
    logger.info(f"参考音频: {ref_path}")
    return ref_path

# 获取AI响应 - 优化版
async def get_ai_response(user_input, messages=None):
    try:
        if messages is None:
            messages = []
            
        messages.append({"role": "user", "content": user_input})
        
        logger.info(f"发送用户输入到AI: {user_input[:50]}...")
        
        # 使用优化的参数设置
        response = await asyncio.to_thread(client.chat.completions.create,
             model="model",
             messages=messages,
             max_tokens=200,
             temperature=0.7,
             stream=False
        )

        ai_response = response.choices[0].message.content
        logger.info(f"收到AI响应: {ai_response[:50]}...")

        # 移除思考过程 (如果有)
        pattern = r'.*?</think>'
        ai_response = re.sub(pattern, '', ai_response, flags=re.DOTALL)

        messages.append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        logger.error(f"获取AI响应错误: {e}")
        return "获取响应出错，请稍后再试"

# 将文本分割成句子以进行批处理
def split_into_sentences(text, max_length=100):
    # 简化的句子分割
    sentences = re.split(r'([。！？\.!?;；\n])', text)
    result = []
    
    # 重新组合句子和标点
    i = 0
    while i < len(sentences) - 1:
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        i += 2
    
    # 处理最后一个片段(如果没有标点)
    if i < len(sentences):
        result.append(sentences[i])
    
    # 合并短句子
    merged_sentences = []
    current = ""
    
    for s in result:
        if s.strip():  # 跳过空句子
            if len(current) + len(s) <= max_length:
                current += s
            else:
                if current:
                    merged_sentences.append(current)
                current = s
    
    if current:
        merged_sentences.append(current)
    
    return merged_sentences

# 优化的TTS处理函数 - 使用GPT-SoVITS
async def text_to_speech_optimized(text, tts_model, ref_audio_path, audio_dir):
    """使用GPT-SoVITS进行TTS生成"""
    start_time = time.time()
    try:
        if tts_model is None or ref_audio_path is None:
            logger.error("GPT-SoVITS模型或参考音频未加载")
            return None, None
            
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 创建时间戳以生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
        
        # 准备TTS请求参数
        request = {
            "text": text,
            "text_lang": "zh",  # 默认中文，可根据需要修改
            "ref_audio_path": ref_audio_path,
            "prompt_text": "",  # 如果需要提示文本可以添加
            "prompt_lang": "zh",  # 与text_lang保持一致
            "text_split_method": "cut5",  # 使用预定义的分割方法
            "speed_factor": 1.0,  # 语速因子
            "split_bucket": True,
            "return_fragment": False,
        }
        
        # 运行TTS推理
        logger.info(f"开始GPT-SoVITS语音合成: 文本长度={len(text)}")
        generator = tts_model.run(request)
        
        # 获取生成的音频
        sr, audio_data = next(generator)
        
        # 保存为WAV文件
        sf.write(wavfile_path, audio_data, sr)
        
        total_time = time.time() - start_time
        logger.info(f"GPT-SoVITS处理完成，耗时: {total_time:.2f}秒")
        
        # 将int16格式转换为float32用于进一步处理
        audio_np = audio_data.astype(np.float32) / 32768.0
        
        return audio_np, wavfile_path
            
    except Exception as e:
        logger.error(f"TTS处理错误: {e}")
        return None, None

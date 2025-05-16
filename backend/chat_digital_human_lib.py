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
import torchaudio
import logging
import soundfile as sf
import concurrent.futures

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ChatDigitalHumanLib')

# 添加GPT-SoVITS路径
GPTSOVITS_PATH = "/home/zentek/Documents/GPT-SoVITS/GPT_SoVITS"


# 确保路径在sys.path中
for path in [GPTSOVITS_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# 初始化线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# 尝试导入GPT-SoVITS模块
try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
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

# 加载GPT-SoVITS TTS模型
def load_gptsovits_model():
    if not gptsovits_available:
        logger.error("GPT-SoVITS模块不可用")
        return None
        
    try:
        # 设置配置文件路径
        config_path = os.path.join(GPTSOVITS_PATH, "configs", "tts_infer.yaml")
        
        # 若配置文件不存在，则创建一个默认配置
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            logger.info(f"配置文件不存在，将使用默认配置: {config_path}")
            
        # 加载TTS配置并初始化模型
        logger.info(f"加载GPT-SoVITS模型配置: {config_path}")
        tts_config = TTS_Config(config_path)
        
             
        if torch.cuda.is_available():
            tts_config.device = torch.device("cuda")
        else:
            logger.info("CUDA不可用，使用CPU")
            tts_config.device = torch.device("cpu")                   
        # 初始化TTS模型
        tts_model = TTS(tts_config)
        logger.info("GPT-SoVITS模型加载成功")
        return tts_model
    except Exception as e:
        logger.error(f"GPT-SoVITS模型加载失败: {e}")
        return None

# 创建一致的参考音频缓存
reference_cache = {
    "ref_audio_path": None,
    "prompt_text": None,
    "prompt_lang": None
}

# 设置默认参考音频信息
def set_reference_audio_info(ref_audio_path, prompt_text, prompt_lang):
    """设置参考音频信息用于TTS"""
    try:
        reference_cache["ref_audio_path"] = ref_audio_path
        reference_cache["prompt_text"] = prompt_text
        reference_cache["prompt_lang"] = prompt_lang
        logger.info(f"参考音频信息已设置: {ref_audio_path}")
        return True
    except Exception as e:
        logger.error(f"设置参考音频信息失败: {e}")
        return False

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
async def text_to_speech_optimized(text, tts_model, reference_info, audio_dir):
    """优化的TTS函数 - 使用GPT-SoVITS"""
    start_time = time.time()
    try:
        if tts_model is None:
            logger.error("GPT-SoVITS模型未加载")
            return None, None
            
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 使用配置进行TTS合成
        ref_audio_path = reference_info.get("ref_audio_path")
        prompt_text = reference_info.get("prompt_text", "")
        prompt_lang = reference_info.get("prompt_lang", "zh")
        
        # 如果缺少必要参数，返回错误
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            logger.error(f"参考音频路径不存在: {ref_audio_path}")
            return None, None

        # 构建GPT-SoVITS输入参数
        tts_inputs = {
            "text": text,
            "text_lang": "zh",  # 假设默认中文，可以根据需要修改
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": 5,
            "top_p": 0.6,
            "temperature": 0.6,
            "text_split_method": "cut5",  # 使用更好的分句方法
            "batch_size": 1,
            "speed_factor": 1.0
        }
        
        logger.info(f"开始GPT-SoVITS音频合成: {text[:50]}...")
        
        # 异步运行GPT-SoVITS推理
        def run_inference():
            try:
                # 运行TTS推理 (GPT-SoVITS的run方法返回生成器)
                for sr, audio_data in tts_model.run(tts_inputs):
                    # 只取第一个结果
                    return audio_data, sr
            except Exception as e:
                logger.error(f"GPT-SoVITS推理出错: {e}")
                return None, None
        
        # 在异步环境中运行TTS推理
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(executor, run_inference)
        
        if audio_data is None:
            logger.error("TTS生成失败")
            return None, None
        
        # 保存音频文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
        
        # 确保音频目录存在
        os.makedirs(audio_dir, exist_ok=True)
        
        # 保存为WAV文件
        sf.write(wavfile_path, audio_data, sample_rate)
        
        total_time = time.time() - start_time
        logger.info(f"TTS处理完成，耗时: {total_time:.2f}秒")
        return audio_data, wavfile_path
            
    except Exception as e:
        logger.error(f"TTS处理错误: {e}")
        return None, None

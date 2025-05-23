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
    """
    加载GPT-SoVITS模型，不使用缓存机制
    """
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
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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

# 修改：支持流式AI响应
def get_ai_response(user_input, messages=None, stream=False):
    """
    获取AI响应，支持流式和非流式模式
    
    参数:
        user_input: 用户输入
        messages: 对话历史
        stream: 是否使用流式模式
        
    返回:
        流式模式返回生成器，非流式模式返回字符串
    """
    try:
        if messages is None:
            messages = []
            
        messages.append({"role": "user", "content": user_input})
        
        logger.info(f"发送用户输入到AI: {user_input[:50]}...")
        
        # 流式模式
        if stream:
            response = client.chat.completions.create(
                model="model",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                stream=True
            )
            
            # 直接返回流式生成器
            return response
        
        # 非流式模式（保持原有逻辑）
        response = client.chat.completions.create(
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
        if stream:
            # 返回一个空的生成器
            return iter([])
        return "获取响应出错，请稍后再试"

# 异步版本的AI响应获取（为了兼容性保留）
async def get_ai_response_async(user_input, messages=None, stream=False):
    """异步获取AI响应"""
    return await asyncio.to_thread(get_ai_response, user_input, messages, stream)

# 将文本分割成句子以进行批处理 - 优化版，适配参考代码的分段逻辑
def split_into_sentences(text, max_length=100):
    """
    将文本智能分割成句子，采用参考代码的分段策略
    
    参数:
        text: 需要分割的文本
        max_length: 每个句子的最大长度（字符数）
    
    返回:
        分割后的句子列表
    """
    if not text:
        return []
    
    # 使用参考代码的标点符号策略
    biao_dian_2 = ["…", "~", "。", "？", "！", "?", "!"]
    biao_dian_3 = ["…", "~", "。", "？", "！", "?", "!", ",", "，"]
    
    # 基本文本清理
    text = text.replace("...", "…")
    
    # 简单的括号内容清理
    cleaned_text = ""
    skip = False
    for char in text:
        if char in ["(", "（"]:
            skip = True
            continue
        if char in [")", "）"]:
            skip = False
            continue
        if not skip:
            cleaned_text += char
    
    # 分段处理
    sentences = []
    start = 0
    first_segment = True
    
    for i in range(len(cleaned_text)):
        if first_segment:
            # 第一段使用严格标点
            if cleaned_text[i] in biao_dian_3 and i - start > 3:
                segment = cleaned_text[start:i+1].strip()
                if len(segment) > 1:
                    sentences.append(segment)
                start = i + 1
                first_segment = False
        else:
            # 后续段落使用宽松标点
            if cleaned_text[i] in biao_dian_2 and i - start > 6:
                segment = cleaned_text[start:i+1].strip()
                if len(segment) > 1:
                    sentences.append(segment)
                start = i + 1
        
        # 防止段落过长
        if i - start >= max_length:
            # 寻找最近的逗号或其他分割点
            for j in range(i, start, -1):
                if cleaned_text[j] in [",", "，", "、"]:
                    segment = cleaned_text[start:j+1].strip()
                    if len(segment) > 1:
                        sentences.append(segment)
                    start = j + 1
                    break
            else:
                # 没找到合适分割点，强制分割
                segment = cleaned_text[start:i].strip()
                if len(segment) > 1:
                    sentences.append(segment)
                start = i
    
    # 处理剩余文本
    if start < len(cleaned_text):
        remaining = cleaned_text[start:].strip()
        if len(remaining) > 1:
            sentences.append(remaining)
    
    # 后处理：合并过短的句子
    final_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) > 0:
            # 过滤掉只包含标点的句子
            if not all(c in ["…", "~", "。", "？", "！", "?", "!", ",", "，", " "] for c in sentence):
                final_sentences.append(sentence.strip())
    
    return final_sentences

# 优化的TTS处理函数 - 针对流式处理优化，保持原有逻辑
async def text_to_speech_optimized(text, tts_model, ref_audio_path, audio_dir):
    """
    使用GPT-SoVITS进行TTS生成，针对流式处理优化
    
    参数:
        text: 要合成的文本
        tts_model: GPT-SoVITS模型实例
        ref_audio_path: 参考音频路径
        audio_dir: 音频输出目录
        
    返回:
        (audio_numpy_array, audio_file_path)
    """
    start_time = time.time()
    
    try:
        if tts_model is None or ref_audio_path is None:
            logger.error("GPT-SoVITS模型或参考音频未加载")
            return None, None
            
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 创建时间戳以生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
        
        # 为空文本快速返回静音
        if not text or text.strip() == "":
            logger.warning("收到空文本，生成静音音频")
            # 创建短静音音频
            empty_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
            sf.write(wavfile_path, empty_audio, 16000)
            return empty_audio, wavfile_path
        
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
        
        # 运行GPT-SoVITS，使用超时处理
        try:
            generator = tts_model.run(request)
            sr, audio_data = await asyncio.wait_for(
                asyncio.to_thread(next, generator),
                timeout=10.0  # 最多等待10秒
            )
        except asyncio.TimeoutError:
            logger.error("GPT-SoVITS处理超时")
            # 创建空音频作为后备
            audio_data = np.zeros(16000, dtype=np.float32)  # 1秒静音
            sr = 16000
        
        # 保存为WAV文件
        sf.write(wavfile_path, audio_data, sr)
        
        total_time = time.time() - start_time
        logger.info(f"GPT-SoVITS处理完成，耗时: {total_time:.2f}秒")
        
        # 将int16格式转换为float32用于进一步处理
        if audio_data.dtype != np.float32:
            audio_np = audio_data.astype(np.float32) / 32768.0
        else:
            audio_np = audio_data
        
        return audio_np, wavfile_path
            
    except Exception as e:
        logger.error(f"TTS处理错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 其他辅助函数保持不变...
def detect_language(text):
    """检测文本的主要语言"""
    if not text:
        return 'zh'  # 默认中文
    
    # 计算各种语言的字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]', text)) - chinese_chars
    
    # 返回占比最高的语言
    if chinese_chars >= english_chars and chinese_chars >= japanese_chars:
        return 'zh'
    elif english_chars >= chinese_chars and english_chars >= japanese_chars:
        return 'en'
    else:
        return 'ja'

def preprocess_text(text):
    """对文本进行预处理"""
    if not text:
        return ""
    
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 标准化常见标点
    text = text.replace('，', '，').replace(',', '，')
    text = text.replace('。', '。').replace('.', '。')
    text = text.replace('？', '？').replace('?', '？')
    text = text.replace('！', '！').replace('!', '！')
    
    # 处理省略号
    text = re.sub(r'\.{3,}', '……', text)
    text = re.sub(r'。{2,}', '……', text)
    
    # 移除HTML和Markdown标记
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    return text

import asyncio
import io
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
import traceback
import logging
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ChatDigitalHumanLib')

# 添加路径
COSYVOICE_PATH = "/home/zentek/Documents/CosyVoice"
MATCHA_TTS_PATH = os.path.join(COSYVOICE_PATH, "third_party/Matcha-TTS")

# 确保路径在sys.path中
for path in [COSYVOICE_PATH, MATCHA_TTS_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# 尝试导入CosyVoice模块
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    cosyvoice_available = True
    logger.info("成功加载CosyVoice模块")
except ImportError as e:
    logger.error(f"无法导入CosyVoice模块: {e}")
    cosyvoice_available = False

# 配置服务器地址、API密钥和对话ID
address = "124.74.245.75:8091"
chat_id = "96c016e6047411f094cc0242ac120006"
api_key = "ragflow-MxODUyZmIwMDQ3NDExZjBhNDA1MDI0Mm"

# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url=f"http://{address}/api/v1/chats_openai/{chat_id}/",
    api_key=api_key
)

# 移除括号内的文本
def clean_text(text):
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 加载CosyVoice2模型
def load_cosyvoice_model():
    if not cosyvoice_available:
        logger.error("CosyVoice模块不可用")
        return None
        
    try:
        possible_model_paths = [
            os.path.join(COSYVOICE_PATH, 'pretrained_models/CosyVoice2-0.5B'),
            '/home/zentek/pretrained_models/CosyVoice2-0.5B',
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            logger.error("找不到CosyVoice2模型")
            return None
        
        # 加载模型
        logger.info(f"加载CosyVoice2模型: {model_path}")
        cosyvoice = CosyVoice2(model_path, 
                             load_jit=False, 
                             load_trt=False, 
                             fp16=False)
        logger.info("CosyVoice2模型加载成功")
        return cosyvoice
    except Exception as e:
        logger.error(f"CosyVoice2模型加载失败: {e}")
        traceback.print_exc()
        return None

# 创建一致的参考音频以确保相同的声音音调
def get_consistent_reference_audio(model, audio_dir):
    # 创建固定的参考音频
    prompt_path = os.path.join(audio_dir, "reference_voice.wav")
    
    try:
        # 如果文件不存在，创建一个
        if not os.path.exists(prompt_path):
            try:
                # 创建语音样本以确保语音一致性
                voice_sample_text = "这是一个测试样本，用于确保语音的一致性和连贯性。"
                
                # 首先创建一个简单的参考音频用于第一次生成
                temp_ref_path = os.path.join(audio_dir, "temp_ref.wav")
                sample_rate = 16000
                duration = 1.0
                t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
                sine_wave = np.sin(2 * np.pi * 440 * t) * 32767 * 0.3
                sine_wave = sine_wave.astype(np.int16)
                from scipy.io import wavfile
                wavfile.write(temp_ref_path, sample_rate, sine_wave)
                
                # 加载临时参考音频
                temp_prompt_speech = load_wav(temp_ref_path, 16000)
                
                # 生成真实的参考音频
                instruction = "用普通话说这句话"
                
                results = list(model.inference_instruct2(
                    voice_sample_text,
                    instruction,
                    temp_prompt_speech,
                    stream=False
                ))
                
                if results and len(results) > 0:
                    audio_tensor = results[0]['tts_speech']
                    torchaudio.save(prompt_path, audio_tensor, model.sample_rate)
                    logger.info(f"参考音频已创建: {prompt_path}")
                else:
                    logger.warning("无法生成参考音频，使用临时音频")
                    return None  # 返回None而不是可能的张量
                
                # 清理临时文件
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
            except Exception as e:
                logger.error(f"创建参考音频失败: {e}")
                traceback.print_exc()
                return None  # 确保发生错误时返回None
        
        # 加载并缓存参考音频
        try:
            reference_audio = load_wav(prompt_path, 16000)
            logger.info("参考音频加载成功")
            return reference_audio
        except Exception as e:
            logger.error(f"加载参考音频失败: {e}")
            return None  # 确保发生错误时返回None
    except Exception as e:
        logger.error(f"处理参考音频时出错: {e}")
        return None  # 确保发生任何错误时返回None

async def get_ai_response(user_input, messages=None):
    try:
        if messages is None:
            messages = []
            
        messages.append({"role": "user", "content": user_input})
        
        logger.info(f"发送用户输入到AI: {user_input[:50]}...")
        response = await asyncio.to_thread(client.chat.completions.create,
             model="model",
             messages=messages,
             stream=False
        )

        ai_response = response.choices[0].message.content
        logger.info(f"收到AI响应: {ai_response[:50]}...")

        # 移除思考过程
        pattern = r'.*?</think>'
        ai_response = re.sub(pattern, '', ai_response, flags=re.DOTALL)
        
        # 移除括号内的提示
        ai_response = clean_text(ai_response)

        messages.append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        logger.error(f"获取AI响应错误: {e}")
        traceback.print_exc()
        return "获取响应出错，请稍后再试"

# 将文本分割成句子，确保每个句子不会太长
def split_into_sentences(text, max_length=80):
    # 查找自然句子边界
    sentence_boundaries = []
    for match in re.finditer(r'[。！？\.!?;；\n]', text):
        sentence_boundaries.append(match.end())
    
    # 如果没有句子边界或第一个句子太长，按长度分割
    if not sentence_boundaries or sentence_boundaries[0] > max_length:
        result = []
        for i in range(0, len(text), max_length):
            result.append(text[i:i + max_length])
        return result
    
    # 根据句子边界分割文本
    sentences = []
    start = 0
    current_length = 0
    current_text = ""
    
    for end in sentence_boundaries:
        sentence = text[start:end]
        # 如果当前句子加上新句子的长度不超过限制，则连接
        if current_length + len(sentence) <= max_length:
            current_text += sentence
            current_length += len(sentence)
        else:
            # 否则添加当前文本并重新开始
            if current_text:
                sentences.append(current_text)
            current_text = sentence
            current_length = len(sentence)
        start = end
    
    # 添加最后部分
    if current_text:
        sentences.append(current_text)
    
    # 处理剩余部分
    if start < len(text):
        remaining = text[start:]
        if current_length + len(remaining) <= max_length:
            sentences[-1] += remaining
        else:
            sentences.append(remaining)
    
    return sentences

# 为单个句子生成音频
async def generate_audio_for_sentence(text, model, reference_audio):
    try:
        instruction = "用普通话说这句话"
        
        # 强制清理内存以避免OOM
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        logger.debug(f"生成句子音频: {text[:30]}...")
        results = list(model.inference_instruct2(
            text,
            instruction,
            reference_audio,
            stream=False
        ))
        
        if results and len(results) > 0:
            audio_tensor = results[0]['tts_speech']
            audio_np = audio_tensor.numpy()
            
            # 确保数组是一维的
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
                
            return audio_np
        return None
    except Exception as e:
        logger.error(f"句子生成错误: {e}")
        return None

# 批处理句子
async def process_sentences_in_batches(sentences, model, reference_audio, batch_size=3):
    all_audio_segments = []
    
    # 批处理句子
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # 处理当前批次
        batch_tasks = []
        for sentence in batch:
            if sentence.strip():  # 只处理非空句子
                task = generate_audio_for_sentence(sentence, model, reference_audio)
                batch_tasks.append(task)
        
        # 等待当前批次完成
        batch_results = await asyncio.gather(*batch_tasks)
        
        # 收集结果
        for result in batch_results:
            if result is not None:
                all_audio_segments.append(result)
        
        # 强制清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_audio_segments

# 使用CosyVoice进行文本到语音转换
async def text_to_speech_cosyvoice(text, model, reference_audio, audio_dir):
    try:
        if model is None:
            logger.error("CosyVoice2模型未加载")
            return None, None
        
        if reference_audio is None:  # 改为检查是否为None
            logger.error("无法获取参考音频")
            return None, None
        
        
        # 特殊处理短文本
        if len(text) <= 100:
            try:
                logger.info(f"处理短文本直接生成: {text[:50]}...")
                single_audio = await generate_audio_for_sentence(text, model, reference_audio)
                if single_audio is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
                    sample_rate = model.sample_rate
                    sf.write(wavfile_path, single_audio, sample_rate)
                    logger.info(f"短文本音频已保存: {wavfile_path}")
                    return single_audio, wavfile_path
            except Exception as e:
                logger.error(f"短文本生成失败: {e}, 尝试分段...")
        
        # 将文本分割成句子
        sentences = split_into_sentences(text)
        if not sentences:
            logger.error("文本分割后为空")
            return None, None
            
        logger.info(f"文本已分割为 {len(sentences)} 个句子")
        
        # 并发处理所有句子
        audio_segments = await process_sentences_in_batches(
            sentences, 
            model, 
            reference_audio
        )
        
        if not audio_segments:
            logger.error("没有成功生成的音频片段")
            return None, None
        
        logger.info(f"成功生成 {len(audio_segments)} 个音频片段")
        
        # 连接片段
        sample_rate = model.sample_rate
        pause_duration = 0.2  
        pause_samples = np.zeros(int(pause_duration * sample_rate))
        
        combined_segments = []
        for segment in audio_segments:
            if segment is not None and len(segment) > 0:
                max_val = np.max(np.abs(segment))
                if max_val > 0:
                    segment = segment / max_val * 0.8  
                
                combined_segments.append(segment)
                combined_segments.append(pause_samples)
        
        if combined_segments and len(combined_segments) > 1:
            combined_segments.pop()  # 移除最后一个暂停
        
        # 合并所有片段
        try:
            combined_audio = np.concatenate(combined_segments)
            
            # 最终归一化
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = combined_audio / max_val * 0.9
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
            sf.write(wavfile_path, combined_audio, sample_rate)
            logger.info(f"合并音频已保存: {wavfile_path}")
            
            # 音频数据长度信息
            audio_duration = len(combined_audio) / sample_rate
            logger.info(f"音频长度: {audio_duration:.2f}秒")
            
            return combined_audio, wavfile_path
        except Exception as e:
            logger.error(f"音频合并错误: {e}")
            traceback.print_exc()
            
            if audio_segments:
                # 如果合并失败，返回第一个片段
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
                sf.write(wavfile_path, audio_segments[0], sample_rate)
                logger.info(f"合并失败，保存第一个片段: {wavfile_path}")
                return audio_segments[0], wavfile_path
    except Exception as e:
        logger.error(f"语音合成处理错误: {e}")
        traceback.print_exc()
    
    return None, None

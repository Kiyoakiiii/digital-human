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

# 创建线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

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
        
        # 加载模型时使用优化选项
        logger.info(f"加载CosyVoice2模型: {model_path}")
        
        # 优化：启用FP16加速和JIT编译
        cosyvoice = CosyVoice2(model_path, 
                             load_jit=False,  # 使用JIT编译加速
                             load_trt=False, # 不使用TensorRT
                             fp16=True)      # 使用半精度加速
                             
        logger.info("CosyVoice2模型加载成功")
        return cosyvoice
    except Exception as e:
        logger.error(f"CosyVoice2模型加载失败: {e}")
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
                voice_sample_text = "这是一个测试样本"
                
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
                    return None
                
                # 清理临时文件
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
            except Exception as e:
                logger.error(f"创建参考音频失败: {e}")
                return None
        
        # 加载并缓存参考音频
        try:
            reference_audio = load_wav(prompt_path, 16000)
            logger.info("参考音频加载成功")
            return reference_audio
        except Exception as e:
            logger.error(f"加载参考音频失败: {e}")
            return None
    except Exception as e:
        logger.error(f"处理参考音频时出错: {e}")
        return None

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

# 批处理TTS生成 - 提高效率
async def batch_generate_audio(sentences, model, reference_audio, batch_size=3):
    """批量处理句子生成音频"""
    if not sentences:
        return []
    
    # 创建批次
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    all_results = []
    
    for batch in batches:
        # 创建并行任务
        tasks = []
        for sentence in batch:
            if sentence.strip():
                task = asyncio.create_task(generate_audio_for_sentence(sentence, model, reference_audio))
                tasks.append(task)
        
        # 执行批处理
        batch_results = await asyncio.gather(*tasks)
        all_results.extend([result for result in batch_results if result is not None])
    
    return all_results

# 单句TTS生成 - 优化版
async def generate_audio_for_sentence(text, model, reference_audio):
    try:
        # 提前准备指令
        instruction = "用普通话说这句话"
        
        # 使用线程池处理耗时的TTS操作
        def process_tts():
            results = list(model.inference_instruct2(
                text,
                instruction,
                reference_audio,
                stream=False
            ))
            
            if results and len(results) > 0:
                audio_tensor = results[0]['tts_speech']
                return audio_tensor.numpy()
            return None
        
        # 异步执行TTS
        loop = asyncio.get_event_loop()
        audio_np = await loop.run_in_executor(executor, process_tts)
        
        if audio_np is not None and len(audio_np.shape) > 1:
            audio_np = audio_np.flatten()
            
        return audio_np
    except Exception as e:
        logger.error(f"句子生成错误: {e}")
        return None

# 优化的TTS处理函数 - 支持批处理
async def text_to_speech_optimized(text, model, reference_audio, audio_dir):
    """优化的TTS函数 - 使用批处理和异步处理"""
    start_time = time.time()
    try:
        if model is None or reference_audio is None:
            logger.error("CosyVoice2模型或参考音频未加载")
            return None, None
            
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 处理短文本 - 直接生成
        if len(text) <= 50:
            logger.info(f"处理短文本直接生成: {text}")
            audio_np = await generate_audio_for_sentence(text, model, reference_audio)
            
            if audio_np is not None:
                # 保存音频
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
                sample_rate = model.sample_rate
                sf.write(wavfile_path, audio_np, sample_rate)
                
                logger.info(f"短文本音频已保存: {wavfile_path}")
                logger.info(f"TTS处理时间: {time.time() - start_time:.2f}秒")
                return audio_np, wavfile_path
            else:
                logger.error("短文本音频生成失败")
        
        # 分割文本为句子
        sentences = split_into_sentences(text, max_length=100)
        logger.info(f"文本已分割为 {len(sentences)} 个句子")
        
        # 批量处理TTS生成
        audio_segments = await batch_generate_audio(sentences, model, reference_audio, batch_size=3)
        
        if not audio_segments:
            logger.error("没有成功生成的音频片段")
            return None, None
            
        logger.info(f"成功生成 {len(audio_segments)} 个音频片段")
        
        # 连接片段 - 使用更短的暂停
        sample_rate = model.sample_rate
        pause_duration = 0.05  # 极短的暂停
        pause_samples = np.zeros(int(pause_duration * sample_rate))
        
        # 合并所有段
        combined_segments = []
        for segment in audio_segments:
            if len(segment) > 0:
                combined_segments.append(segment)
                combined_segments.append(pause_samples)
                
        if combined_segments and len(combined_segments) > 1:
            combined_segments.pop()  # 移除最后一个暂停
            
        # 合并音频
        combined_audio = np.concatenate(combined_segments)
        
        # 归一化音频
        max_val = np.max(np.abs(combined_audio))
        if max_val > 0:
            combined_audio = combined_audio / max_val * 0.9
            
        # 保存最终音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wavfile_path = os.path.join(audio_dir, f"response_audio_{timestamp}.wav")
        sf.write(wavfile_path, combined_audio, sample_rate)
        
        total_time = time.time() - start_time
        logger.info(f"TTS处理完成，耗时: {total_time:.2f}秒")
        return combined_audio, wavfile_path
            
    except Exception as e:
        logger.error(f"TTS处理错误: {e}")
        return None, None

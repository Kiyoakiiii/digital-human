import os
import time
import uuid
import grpc
import wave
import numpy as np
import csv
import traceback
import math
from typing import List, Dict, Optional, Tuple, Generator
import logging
import pandas as pd
from scipy import signal  # 用于简化版Blendshape生成

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Audio2Face3DClient')

# 导入gRPC proto生成的模块
try:
    # 尝试导入官方包
    import nvidia_ace.controller.v1_pb2 as controller_pb2
    import nvidia_ace.services.a2f_controller.v1_pb2 as a2f_controller_service_pb2
    import nvidia_ace.services.a2f_controller.v1_pb2_grpc as a2f_controller_service_pb2_grpc
    import nvidia_ace.a2f.v1_pb2 as a2f_pb2
    import nvidia_ace.audio.v1_pb2 as audio_pb2
    import nvidia_ace.animation_data.v1_pb2 as animation_data_pb2
    import nvidia_ace.status.v1_pb2 as status_pb2
    logger.info("成功导入官方Audio2Face-3D gRPC模块")
except ImportError as e:
    logger.warning(f"导入官方模块失败: {e}")
    # 尝试导入本地生成的模块
    try:
        import sys
        PROTO_DIR = os.path.join(os.path.dirname(__file__), "proto_gen")
        if PROTO_DIR not in sys.path:
            sys.path.append(PROTO_DIR)
        
        # 导入从proto文件生成的模块
        import controller_pb2
        import a2f_controller_service_pb2
        import a2f_controller_service_pb2_grpc
        import a2f_pb2
        import audio_pb2
        import animation_data_pb2
        import status_pb2
        logger.info("成功导入本地生成的Audio2Face-3D gRPC模块")
    except ImportError as e:
        logger.error(f"导入本地生成的模块也失败: {e}")
        logger.error("需要先生成gRPC模块，请运行proto_generator.py")
        raise

# 新增：简化版Blendshape生成器（省去调用Audio2Face-3D服务的复杂过程）
class SimpleBlendshapeGenerator:
    """基于音频振幅的简化版Blendshape生成器"""
    
    def __init__(self):
        """初始化简化版Blendshape生成器"""
        # ARKit Blendshape 名称列表 (精简版，只保留关键的口型相关Blendshape)
        self.blendshape_names = [
            "EyeBlinkLeft", "EyeBlinkRight", "JawOpen", "MouthClose", 
            "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight", 
            "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
            "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight"
        ]
        logger.info("初始化简化版Blendshape生成器")
    
    def generate_blendshape_from_audio(self, audio_samples, output_csv_path, sample_rate=16000, fps=30):
        """
        从音频样本生成简化的Blendshape数据
        
        参数:
            audio_samples: 音频样本数据 (numpy数组)
            output_csv_path: 输出CSV文件路径
            sample_rate: 音频采样率
            fps: 动画帧率
        
        返回:
            成功返回True，否则返回False
        """
        logger.info(f"使用简化方法生成Blendshape，采样率: {sample_rate}, fps: {fps}")
        
        try:
            # 确保音频数据是numpy数组
            if not isinstance(audio_samples, np.ndarray):
                audio_samples = np.array(audio_samples)
            
            # 正规化音频数据
            if audio_samples.dtype != np.float32:
                audio_samples = audio_samples.astype(np.float32)
                if np.max(np.abs(audio_samples)) > 1.0:
                    audio_samples = audio_samples / 32768.0  # 假设16位音频
            
            # 计算每帧的音频样本数
            samples_per_frame = int(sample_rate / fps)
            num_frames = math.ceil(len(audio_samples) / samples_per_frame)
            
            # 计算音频能量包络
            # 使用短时傅里叶变换(STFT)计算频谱
            _, _, spectrogram = signal.stft(
                audio_samples, 
                fs=sample_rate, 
                nperseg=512, 
                noverlap=256
            )
            
            # 计算能量包络 (频谱的幅度平方)
            energy = np.sum(np.abs(spectrogram)**2, axis=0)
            
            # 重采样能量包络到视频帧率
            energy_frames = np.zeros(num_frames)
            for i in range(min(num_frames, len(energy))):
                energy_frames[i] = energy[i]
            
            # 平滑能量包络
            energy_frames = signal.savgol_filter(energy_frames, 5, 2)
            
            # 正规化能量包络
            energy_max = np.max(energy_frames)
            if energy_max > 0:
                energy_frames = energy_frames / energy_max
            
            # 创建Blendshape数据
            blend_data = []
            for frame_idx in range(num_frames):
                time_point = frame_idx / fps
                energy_val = energy_frames[frame_idx]
                
                # 创建帧数据
                frame_data = {
                    "frame": frame_idx,
                    "time": time_point,
                    "time_code": frame_idx / fps
                }
                
                # 基于音频能量添加Blendshape值
                # JawOpen - 主要受音频能量影响
                frame_data["JawOpen"] = np.clip(energy_val * 1.2, 0.0, 1.0)
                
                # MouthClose - 与JawOpen相反
                frame_data["MouthClose"] = 1.0 - frame_data["JawOpen"] * 0.8
                
                # 添加微笑和其他口型的变化（基于音频能量和一些随机性）
                smile_val = energy_val * 0.5 + np.sin(time_point * 2) * 0.1
                frame_data["MouthSmileLeft"] = np.clip(smile_val, 0.0, 0.6)
                frame_data["MouthSmileRight"] = np.clip(smile_val, 0.0, 0.6)
                
                # 添加其他Blendshape（眨眼等）的基本动画
                if frame_idx % 90 == 0:  # 约3秒眨一次眼
                    frame_data["EyeBlinkLeft"] = 1.0
                    frame_data["EyeBlinkRight"] = 1.0
                elif frame_idx % 90 == 1:  # 眨眼后的下一帧
                    frame_data["EyeBlinkLeft"] = 0.5
                    frame_data["EyeBlinkRight"] = 0.5
                else:
                    frame_data["EyeBlinkLeft"] = 0.0
                    frame_data["EyeBlinkRight"] = 0.0
                
                # 添加其他口型动画
                if energy_val > 0.7:  # 高能量时的特殊口型
                    frame_data["MouthPucker"] = np.clip((energy_val - 0.7) * 2, 0, 0.5)
                else:
                    frame_data["MouthPucker"] = 0.0
                
                # 上下嘴唇运动
                frame_data["MouthUpperUpLeft"] = frame_data["JawOpen"] * 0.3
                frame_data["MouthUpperUpRight"] = frame_data["JawOpen"] * 0.3
                frame_data["MouthLowerDownLeft"] = frame_data["JawOpen"] * 0.5
                frame_data["MouthLowerDownRight"] = frame_data["JawOpen"] * 0.5
                
                # 将非零值添加到结果中
                blend_data.append(frame_data)
            
            # 保存Blendshape数据到CSV
            return self.save_blendshape_to_csv(blend_data, output_csv_path)
        
        except Exception as e:
            logger.error(f"生成简化Blendshape数据时出错: {e}")
            traceback.print_exc()
            return False
    
    def save_blendshape_to_csv(self, blendshape_data, output_csv_path):
        """将Blendshape数据保存为CSV文件"""
        if not blendshape_data:
            logger.error("没有Blendshape数据可保存")
            return False
            
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 获取所有字段名
            fieldnames = ["frame", "time", "time_code"]
            fieldnames.extend(self.blendshape_names)
            
            # 写入CSV文件
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for frame_data in blendshape_data:
                    # 创建一个只包含已知字段的行
                    row = {field: frame_data.get(field, 0.0) for field in fieldnames}
                    writer.writerow(row)
            
            logger.info(f"简化Blendshape数据已保存至: {output_csv_path}")
            return True
        except Exception as e:
            logger.error(f"保存Blendshape数据到CSV时出错: {e}")
            traceback.print_exc()
            return False

class Audio2Face3DClient:
    """与Audio2Face-3D gRPC服务交互的客户端"""

    def __init__(self, server_address: str = "localhost:52000", max_buffer_seconds: float = 10.0):
        """
        初始化Audio2Face-3D客户端
        
        参数:
            server_address: Audio2Face-3D gRPC服务器地址
            max_buffer_seconds: 最大处理音频缓冲区长度，超过此长度的音频将被分段处理
        """
        self.server_address = server_address
        self.max_buffer_seconds = max_buffer_seconds
        logger.info(f"初始化Audio2Face-3D客户端，服务器地址：{server_address}，最大缓冲区长度：{max_buffer_seconds}秒")
        
        # 创建安全选项，增加消息大小限制
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
        
        # 创建gRPC通道和存根
        self.channel = grpc.insecure_channel(server_address, options=options)
        self.stub = a2f_controller_service_pb2_grpc.A2FControllerServiceStub(self.channel)
        
        # 用于跟踪活动流会话
        self.active_streams = {}
        
        # ARKit Blendshape 名称列表
        self.blendshape_names = [
            "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
            "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
            "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
            "JawLeft", "JawRight", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft",
            "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
            "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight", "MouthRollLower",
            "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
            "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight", "BrowDownLeft",
            "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff",
            "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight", "TongueOut"
        ]
        
        # 测试连接
        try:
            # 创建简单请求检查连接
            self._test_connection()
            logger.info("成功连接到Audio2Face-3D服务")
        except Exception as e:
            logger.error(f"连接Audio2Face-3D服务失败: {e}")
    
    def _test_connection(self):
        """测试与服务器的连接"""
        try:
            # 我们使用5秒超时来测试连接
            grpc.channel_ready_future(self.channel).result(timeout=5)
            return True
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"无法连接到Audio2Face-3D服务: {self.server_address}")
    
    def close(self):
        """关闭gRPC通道并清理资源"""
        logger.info("关闭Audio2Face-3D客户端")
        for stream_id in list(self.active_streams.keys()):
            self.end_streaming_session(stream_id)
        self.channel.close()
    
    def _create_audio_stream_header(self, sample_rate: int) -> controller_pb2.AudioStreamHeader:
        """创建音频流头部"""
        logger.debug(f"创建音频流头部，采样率: {sample_rate}")
        # 创建音频头部
        audio_header = audio_pb2.AudioHeader(
            audio_format=audio_pb2.AudioHeader.AUDIO_FORMAT_PCM,
            channel_count=1,
            samples_per_second=sample_rate,
            bits_per_sample=16
        )
        
        # 创建面部参数
        face_params = a2f_pb2.FaceParameters(
            float_params={
                "lowerFaceSmoothing": 0.3,
                "upperFaceSmoothing": 0.3,
                "lowerFaceStrength": 1.0,
                "upperFaceStrength": 1.0,
                "faceMaskLevel": 0.5,
                "faceMaskSoftness": 0.3,
                "skinStrength": 1.0,
                "blinkStrength": 1.0,
                "eyelidOpenOffset": 0.0,
                "lipOpenOffset": 0.0,
                "blinkOffset": 0.0,
                "tongueStrength": 1.0,
                "tongueHeightOffset": 0.0,
                "tongueDepthOffset": 0.0
            }
        )
        
        # 创建blendshape参数
        blendshape_params = a2f_pb2.BlendShapeParameters(
            bs_weight_multipliers={
                "JawOpen": 1.3,
                "MouthSmileLeft": 1.3,
                "MouthSmileRight": 1.3,
                "BrowInnerUp": 1.2,
                "EyeBlinkLeft": 1.0,
                "EyeBlinkRight": 1.0
            },
            bs_weight_offsets={},
            enable_clamping_bs_weight=True
        )
        
        # 创建情绪参数
        emotion_params = a2f_pb2.EmotionParameters(
            live_transition_time=0.3,
            beginning_emotion={
                "neutral": 1.0
            }
        )
        
        # 创建情绪后处理参数
        emotion_post_processing_params = a2f_pb2.EmotionPostProcessingParameters(
            emotion_contrast=1.2,
            live_blend_coef=0.7,
            enable_preferred_emotion=True,
            preferred_emotion_strength=0.6,
            emotion_strength=0.7,
            max_emotions=3
        )
        
        # 创建并返回音频流头部
        return controller_pb2.AudioStreamHeader(
            audio_header=audio_header,
            face_params=face_params,
            blendshape_params=blendshape_params,
            emotion_params=emotion_params,
            emotion_post_processing_params=emotion_post_processing_params
        )
    
    def _generate_audio_stream(self, audio_data: bytes, sample_rate: int) -> Generator[controller_pb2.AudioStream, None, None]:
        """为Audio2Face-3D服务生成音频流"""
        logger.debug(f"生成音频流，数据大小: {len(audio_data)} 字节")
        # 创建并生成头部
        header = self._create_audio_stream_header(sample_rate)
        yield controller_pb2.AudioStream(audio_stream_header=header)
        
        # 创建并生成音频与情绪
        audio_with_emotion = a2f_pb2.AudioWithEmotion(
            audio_buffer=audio_data,
            emotions=[]  # 没有预定义情绪，让A2F检测
        )
        yield controller_pb2.AudioStream(audio_with_emotion=audio_with_emotion)
        
        # 标记音频流结束
        yield controller_pb2.AudioStream(end_of_audio=controller_pb2.AudioStream.EndOfAudio())
    
    def process_audio_data_segment(self, audio_data: bytes, sample_rate: int) -> List[animation_data_pb2.AnimationData]:
        """处理单个音频数据段"""
        logger.debug(f"处理音频数据段，大小: {len(audio_data)} 字节")
        audio_stream = self._generate_audio_stream(audio_data, sample_rate)
        
        try:
            # 调用服务并收集响应
            responses = self.stub.ProcessAudioStream(audio_stream)
            
            # 处理响应
            animation_data_list = []
            header_received = False
            
            for response in responses:
                if response.HasField("animation_data_stream_header"):
                    header_received = True
                    logger.debug("接收到动画数据流头部")
                elif response.HasField("animation_data") and header_received:
                    animation_data_list.append(response.animation_data)
                elif response.HasField("status"):
                    # 检查是否有错误
                    status = response.status
                    logger.debug(f"接收到状态: {status.code} - {status.message}")
                    if status.code == status_pb2.Status.ERROR:
                        raise Exception(f"Audio2Face-3D错误: {status.message}")
                elif response.HasField("event"):
                    # 记录事件
                    event = response.event
                    logger.debug(f"接收到事件: {event.event_type}")
            
            logger.debug(f"处理完成，收到 {len(animation_data_list)} 个动画数据帧")
            return animation_data_list
        except grpc.RpcError as e:
            logger.error(f"RPC错误: {e.details() if hasattr(e, 'details') else str(e)}")
            raise
        except Exception as e:
            logger.error(f"处理音频数据段时出错: {e}")
            raise
    
    def process_audio_data(self, audio_data: bytes, sample_rate: int) -> List[animation_data_pb2.AnimationData]:
        """通过Audio2Face-3D服务处理音频数据，如有必要会分段处理较长的音频"""
        logger.info(f"处理音频数据，大小: {len(audio_data)} 字节")
        
        # 计算音频长度（秒）
        bytes_per_sample = 2  # 对于16位PCM
        samples_per_second = sample_rate
        bytes_per_second = bytes_per_sample * samples_per_second
        audio_length_seconds = len(audio_data) / bytes_per_second
        
        logger.info(f"音频长度: {audio_length_seconds:.2f} 秒")
        
        # 检查是否需要分段处理
        if audio_length_seconds <= self.max_buffer_seconds:
            # 音频足够短，可以直接处理
            return self.process_audio_data_segment(audio_data, sample_rate)
        else:
            # 音频太长，需要分段处理
            logger.info(f"音频长度超过最大缓冲区长度 ({self.max_buffer_seconds} 秒)，进行分段处理")
            
            # 计算段数
            segment_bytes = int(self.max_buffer_seconds * bytes_per_second)
            num_segments = math.ceil(len(audio_data) / segment_bytes)
            
            # 存储所有段的动画数据
            all_animation_data = []
            
            # 处理每个段
            for i in range(num_segments):
                start_byte = i * segment_bytes
                end_byte = min((i + 1) * segment_bytes, len(audio_data))
                segment_data = audio_data[start_byte:end_byte]
                
                logger.info(f"处理段 {i+1}/{num_segments}，大小: {len(segment_data)} 字节")
                
                # 处理此段
                try:
                    segment_animation_data = self.process_audio_data_segment(segment_data, sample_rate)
                    all_animation_data.extend(segment_animation_data)
                    logger.info(f"段 {i+1} 处理完成，获取 {len(segment_animation_data)} 个动画数据帧")
                except Exception as e:
                    logger.error(f"处理段 {i+1} 时出错: {e}")
                    # 继续处理下一段，而不是完全失败
            
            logger.info(f"所有段处理完成，总共获取 {len(all_animation_data)} 个动画数据帧")
            return all_animation_data
    
    def process_audio_file(self, audio_file_path: str) -> List[animation_data_pb2.AnimationData]:
        """通过Audio2Face-3D服务处理音频文件"""
        logger.info(f"处理音频文件: {audio_file_path}")
        # 读取音频文件
        with wave.open(audio_file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
        
        logger.debug(f"音频文件采样率: {sample_rate}，数据大小: {len(audio_data)} 字节")
        # 处理音频数据
        return self.process_audio_data(audio_data, sample_rate)
    
    def extract_blendshape_data(self, animation_data_list: List[animation_data_pb2.AnimationData]) -> List[Dict]:
        """
        从动画数据中提取ARKit Blendshape数据
        
        参数:
            animation_data_list: AnimationData对象列表
            
        返回:
            包含blendshape数据的字典列表，每帧一个字典
        """
        all_frames = []
        
        for frame_idx, anim_data in enumerate(animation_data_list):
            if not anim_data.HasField("skel_animation"):
                continue
                
            frame_data = {
                "frame": frame_idx,
                "time": frame_idx / 30.0  # 假设30fps
            }
            
            # 提取blendshape权重
            skel_animation = anim_data.skel_animation
            for bs_data in skel_animation.blend_shape_weights:
                if not hasattr(bs_data, 'values') or not bs_data.values:
                    continue
                    
                # 获取时间码
                time_code = bs_data.time_code
                frame_data["time_code"] = time_code
                
                # 提取并存储所有blendshape值
                if hasattr(bs_data, 'values'):
                    for i, value in enumerate(bs_data.values):
                        if i < len(self.blendshape_names):
                            bs_name = self.blendshape_names[i]
                            frame_data[bs_name] = value
            
            all_frames.append(frame_data)
        
        return all_frames
    
    def save_blendshape_to_csv(self, blendshape_data: List[Dict], output_csv_path: str) -> bool:
        """
        将Blendshape数据保存为CSV文件
        
        参数:
            blendshape_data: 包含blendshape数据的字典列表
            output_csv_path: 输出CSV文件路径
            
        返回:
            成功返回True，否则返回False
        """
        if not blendshape_data:
            logger.error("没有Blendshape数据可保存")
            return False
            
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 获取所有字段名
            fieldnames = ["frame", "time", "time_code"]
            for bs_name in self.blendshape_names:
                if any(bs_name in frame for frame in blendshape_data):
                    fieldnames.append(bs_name)
            
            # 写入CSV文件
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for frame_data in blendshape_data:
                    # 创建一个只包含已知字段的行
                    row = {field: frame_data.get(field, "") for field in fieldnames}
                    writer.writerow(row)
            
            logger.info(f"Blendshape数据已保存至: {output_csv_path}")
            return True
        except Exception as e:
            logger.error(f"保存Blendshape数据到CSV时出错: {e}")
            traceback.print_exc()
            return False
    
    def process_audio_file_to_blendshape(self, audio_file_path: str, output_csv_path: str) -> bool:
        """
        处理音频文件并将Blendshape数据保存到CSV
        
        参数:
            audio_file_path: 音频文件路径（WAV）
            output_csv_path: 输出CSV文件路径
            
        返回:
            成功返回True，否则返回False
        """
        logger.info(f"处理音频文件并转换为Blendshape数据: {audio_file_path} -> {output_csv_path}")
        try:
            # 验证音频文件
            if not os.path.exists(audio_file_path):
                logger.error(f"音频文件不存在: {audio_file_path}")
                return False
                
            # 验证音频文件格式和内容
            try:
                with wave.open(audio_file_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    logger.info(f"音频文件信息: 采样率={sample_rate}Hz, 通道数={channels}, 帧数={n_frames}")
                    
                    if channels != 1:
                        logger.warning(f"警告: 音频不是单声道，Audio2Face-3D可能需要单声道音频")
                    
                    if n_frames == 0:
                        logger.error("错误: 音频文件没有帧")
                        return False
            except Exception as e:
                logger.error(f"读取音频文件时出错: {e}")
                return False
            
            # 处理音频文件获取动画数据
            animation_data_list = self.process_audio_file(audio_file_path)
            
            if not animation_data_list:
                logger.error("没有从Audio2Face-3D获取到动画数据")
                return False
                
            logger.info(f"成功获取 {len(animation_data_list)} 个动画数据帧")
            
            # 提取Blendshape数据
            blendshape_data = self.extract_blendshape_data(animation_data_list)
            
            if not blendshape_data:
                logger.error("无法从动画数据提取Blendshape数据")
                return False
                
            logger.info(f"成功提取 {len(blendshape_data)} 帧Blendshape数据")
            
            # 保存Blendshape数据到CSV
            return self.save_blendshape_to_csv(blendshape_data, output_csv_path)
        except Exception as e:
            logger.error(f"处理音频到Blendshape数据时出错: {e}")
            traceback.print_exc()
            return False
    
    def split_audio_file(self, audio_file_path: str, max_segment_seconds: float = None) -> List[str]:
        """将音频文件分割成更小的段"""
        if max_segment_seconds is None:
            max_segment_seconds = self.max_buffer_seconds
        
        logger.info(f"分割音频文件: {audio_file_path}, 最大段长: {max_segment_seconds}秒")
        
        # 读取音频文件
        try:
            with wave.open(audio_file_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
            
            audio_length_seconds = n_frames / sample_rate
            logger.info(f"音频长度: {audio_length_seconds:.2f}秒, 采样率: {sample_rate}Hz")
            
            # 检查是否需要分割
            if audio_length_seconds <= max_segment_seconds:
                logger.info("音频长度在限制范围内，无需分割")
                return [audio_file_path]
            
            # 计算每段的帧数
            frames_per_segment = int(max_segment_seconds * sample_rate)
            num_segments = math.ceil(n_frames / frames_per_segment)
            
            logger.info(f"将音频分割为 {num_segments} 段")
            
            # 创建输出目录
            output_dir = os.path.dirname(audio_file_path)
            base_name = os.path.basename(audio_file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            segment_paths = []
            
            # 分割音频
            for i in range(num_segments):
                start_frame = i * frames_per_segment
                end_frame = min((i + 1) * frames_per_segment, n_frames)
                segment_frames = end_frame - start_frame
                
                # 计算此段的字节范围
                bytes_per_frame = sample_width * channels
                start_byte = start_frame * bytes_per_frame
                end_byte = end_frame * bytes_per_frame
                
                # 提取此段的数据
                segment_data = audio_data[start_byte:end_byte]
                
                # 创建输出文件路径
                segment_path = os.path.join(output_dir, f"{name_without_ext}_segment_{i+1}.wav")
                
                # 保存段
                with wave.open(segment_path, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(segment_data)
                
                segment_paths.append(segment_path)
                logger.info(f"保存段 {i+1}/{num_segments} 到 {segment_path}")
            
            return segment_paths
        except Exception as e:
            logger.error(f"分割音频文件时出错: {e}")
            traceback.print_exc()
            return [audio_file_path]  # 失败时返回原始文件
    
    def process_long_audio_file_to_blendshape(self, audio_file_path: str, output_csv_path: str) -> bool:
        """
        处理长音频文件并保存Blendshape数据到CSV，自动处理分段
        
        参数:
            audio_file_path: 音频文件路径（WAV）
            output_csv_path: 输出CSV文件路径
            
        返回:
            成功返回True，否则返回False
        """
        logger.info(f"处理长音频文件到Blendshape数据: {audio_file_path} -> {output_csv_path}")
        
        # 分割音频文件
        segment_paths = self.split_audio_file(audio_file_path)
        
        if len(segment_paths) == 1 and segment_paths[0] == audio_file_path:
            # 无需分割，直接处理原始文件
            return self.process_audio_file_to_blendshape(audio_file_path, output_csv_path)
        
        # 处理每个段并收集动画数据
        all_animation_data = []
        
        for i, segment_path in enumerate(segment_paths):
            logger.info(f"处理段 {i+1}/{len(segment_paths)}: {segment_path}")
            
            try:
                segment_animation_data = self.process_audio_file(segment_path)
                all_animation_data.extend(segment_animation_data)
                logger.info(f"段 {i+1} 处理完成，获取 {len(segment_animation_data)} 个动画数据帧")
                
                # 删除临时段文件
                if segment_path != audio_file_path:
                    os.remove(segment_path)
            except Exception as e:
                logger.error(f"处理段 {i+1} 时出错: {e}")
                # 继续处理下一段
        
        if not all_animation_data:
            logger.error("没有从任何段获取到动画数据")
            return False
        
        logger.info(f"所有段处理完成，总共获取 {len(all_animation_data)} 个动画数据帧")
        
        # 提取Blendshape数据
        blendshape_data = self.extract_blendshape_data(all_animation_data)
        
        if not blendshape_data:
            logger.error("无法从动画数据提取Blendshape数据")
            return False
            
        logger.info(f"成功提取 {len(blendshape_data)} 帧Blendshape数据")
        
        # 保存Blendshape数据到CSV
        return self.save_blendshape_to_csv(blendshape_data, output_csv_path)

# 使用示例
if __name__ == "__main__":
    client = Audio2Face3DClient("localhost:52000", max_buffer_seconds=10.0)
    success = client.process_long_audio_file_to_blendshape(
        "example.wav", 
        "example_blendshape.csv"
    )
    print(f"处理结果: {success}")
    client.close()

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
import asyncio

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

# 缓存和性能优化
_client_instance = None
_blendshape_generator = None

# 完整的ARKit Blendshape生成器
class SimpleBlendshapeGenerator:
    """基于音频振幅的完整ARKit Blendshape生成器，针对流式处理优化"""
    
    def __init__(self):
        """初始化完整ARKit Blendshape生成器"""
        # 完整的ARKit Blendshape 名称列表 - 52个标准blendshape
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
        
        # 更完整的音位与Blendshape映射
        self.phoneme_to_blendshape = {
            # 元音映射 - 更精确的口型
            'a': {
                'JawOpen': 0.8, 'MouthUpperUpLeft': 0.4, 'MouthUpperUpRight': 0.4, 
                'MouthLowerDownLeft': 0.6, 'MouthLowerDownRight': 0.6,
                'MouthLeft': 0.1, 'MouthRight': 0.1
            },
            'e': {
                'JawOpen': 0.5, 'MouthUpperUpLeft': 0.6, 'MouthUpperUpRight': 0.6, 
                'MouthLowerDownLeft': 0.3, 'MouthLowerDownRight': 0.3,
                'MouthStretchLeft': 0.4, 'MouthStretchRight': 0.4
            },
            'i': {
                'JawOpen': 0.3, 'MouthStretchLeft': 0.7, 'MouthStretchRight': 0.7,
                'MouthSmileLeft': 0.5, 'MouthSmileRight': 0.5
            },
            'o': {
                'JawOpen': 0.6, 'MouthFunnel': 0.8, 'MouthPucker': 0.4,
                'MouthRollUpper': 0.2, 'MouthRollLower': 0.2
            },
            'u': {
                'JawOpen': 0.4, 'MouthPucker': 0.9, 'MouthFunnel': 0.6,
                'MouthRollUpper': 0.3, 'MouthRollLower': 0.3
            },
            
            # 辅音映射 - 更细致的口型变化
            'b': {
                'MouthClose': 0.9, 'MouthPressLeft': 0.8, 'MouthPressRight': 0.8,
                'JawOpen': 0.1, 'MouthRollLower': 0.3
            },
            'p': {
                'MouthClose': 0.9, 'MouthPressLeft': 0.8, 'MouthPressRight': 0.8,
                'MouthPucker': 0.3
            },
            'm': {
                'MouthClose': 0.9, 'MouthPressLeft': 0.6, 'MouthPressRight': 0.6,
                'MouthRollLower': 0.2, 'MouthRollUpper': 0.2
            },
            'f': {
                'JawOpen': 0.3, 'MouthLowerDownLeft': 0.3, 'MouthLowerDownRight': 0.3, 
                'MouthUpperUpLeft': 0.4, 'MouthUpperUpRight': 0.4,
                'MouthFunnel': 0.2
            },
            'v': {
                'JawOpen': 0.3, 'MouthLowerDownLeft': 0.4, 'MouthLowerDownRight': 0.4, 
                'MouthUpperUpLeft': 0.3, 'MouthUpperUpRight': 0.3
            },
            's': {
                'JawOpen': 0.3, 'MouthStretchLeft': 0.5, 'MouthStretchRight': 0.5,
                'MouthClose': 0.3, 'MouthFunnel': 0.2
            },
            'z': {
                'JawOpen': 0.3, 'MouthStretchLeft': 0.4, 'MouthStretchRight': 0.4,
                'MouthClose': 0.2
            },
            'th': {
                'JawOpen': 0.4, 'MouthUpperUpLeft': 0.2, 'MouthUpperUpRight': 0.2,
                'TongueOut': 0.3
            },
            'r': {
                'JawOpen': 0.5, 'MouthFunnel': 0.5, 'MouthPucker': 0.3,
                'MouthRollUpper': 0.4
            },
            'l': {
                'JawOpen': 0.4, 'MouthLeft': 0.4, 'MouthRight': 0.4,
                'TongueOut': 0.2
            },
            'n': {
                'JawOpen': 0.3, 'MouthClose': 0.4,
                'TongueOut': 0.1
            },
            'd': {
                'JawOpen': 0.4, 'MouthUpperUpLeft': 0.3, 'MouthUpperUpRight': 0.3,
                'TongueOut': 0.2
            },
            't': {
                'JawOpen': 0.3, 'MouthClose': 0.5,
                'TongueOut': 0.1
            },
            'k': {
                'JawOpen': 0.5, 'MouthLeft': 0.2, 'MouthRight': 0.2,
                'JawLeft': 0.1, 'JawRight': 0.1
            },
            'g': {
                'JawOpen': 0.5, 'MouthLeft': 0.3, 'MouthRight': 0.3,
                'JawLeft': 0.1, 'JawRight': 0.1
            }
        }
        
        # 眨眼动画模式 - 更自然的眨眼曲线
        self.blink_pattern = [0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.9, 1.0, 0.9, 0.6, 0.3, 0.1, 0.0, 0.0]
        
        # 眼部运动模式
        self.eye_movement_patterns = {
            'look_left': {'EyeLookOutLeft': 0.5, 'EyeLookInRight': 0.5},
            'look_right': {'EyeLookInLeft': 0.5, 'EyeLookOutRight': 0.5},
            'look_up': {'EyeLookUpLeft': 0.4, 'EyeLookUpRight': 0.4},
            'look_down': {'EyeLookDownLeft': 0.3, 'EyeLookDownRight': 0.3}
        }
        
        # 表情模式
        self.expression_patterns = {
            'happy': {
                'MouthSmileLeft': 0.7, 'MouthSmileRight': 0.7,
                'CheekSquintLeft': 0.3, 'CheekSquintRight': 0.3,
                'BrowOuterUpLeft': 0.2, 'BrowOuterUpRight': 0.2
            },
            'sad': {
                'MouthFrownLeft': 0.6, 'MouthFrownRight': 0.6,
                'BrowDownLeft': 0.4, 'BrowDownRight': 0.4,
                'BrowInnerUp': 0.3
            },
            'surprised': {
                'EyeWideLeft': 0.8, 'EyeWideRight': 0.8,
                'BrowOuterUpLeft': 0.7, 'BrowOuterUpRight': 0.7,
                'BrowInnerUp': 0.5, 'JawOpen': 0.6
            },
            'confused': {
                'BrowDownLeft': 0.3, 'BrowInnerUp': 0.4,
                'MouthLeft': 0.2
            }
        }
        
        logger.info("初始化完整ARKit Blendshape生成器 (52个blendshape)")
    
    def generate_blendshape_from_audio(self, audio_samples, output_csv_path, sample_rate=16000, fps=30):
        """
        从音频样本生成完整的ARKit Blendshape数据
        
        参数:
            audio_samples: 音频样本数据 (numpy数组)
            output_csv_path: 输出CSV文件路径
            sample_rate: 音频采样率
            fps: 动画帧率
        
        返回:
            成功返回True，否则返回False
        """
        logger.info(f"使用完整ARKit方法生成Blendshape，采样率: {sample_rate}, fps: {fps}")
        
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
            
            # 计算多频段的音频特征
            # 使用短时傅里叶变换(STFT)计算频谱
            _, _, spectrogram = signal.stft(
                audio_samples, 
                fs=sample_rate, 
                nperseg=1024, 
                noverlap=512
            )
            
            # 计算不同频段的能量
            freq_bins = spectrogram.shape[0]
            low_freq_energy = np.sum(np.abs(spectrogram[:freq_bins//4])**2, axis=0)      # 低频 (0-1/4)
            mid_freq_energy = np.sum(np.abs(spectrogram[freq_bins//4:freq_bins//2])**2, axis=0)  # 中频 (1/4-1/2)
            high_freq_energy = np.sum(np.abs(spectrogram[freq_bins//2:])**2, axis=0)    # 高频 (1/2-1)
            
            # 总能量
            total_energy = np.sum(np.abs(spectrogram)**2, axis=0)
            
            # 重采样能量包络到视频帧率
            energy_frames = np.zeros((4, num_frames))  # 4个频段
            for i in range(min(num_frames, total_energy.shape[0])):
                energy_frames[0, i] = low_freq_energy[i]
                energy_frames[1, i] = mid_freq_energy[i]
                energy_frames[2, i] = high_freq_energy[i]
                energy_frames[3, i] = total_energy[i]
            
            # 平滑能量包络
            for freq_idx in range(4):
                if num_frames > 5:
                    energy_frames[freq_idx] = signal.savgol_filter(energy_frames[freq_idx], 5, 2)
            
            # 正规化能量包络
            for freq_idx in range(4):
                energy_max = np.max(energy_frames[freq_idx])
                if energy_max > 0:
                    energy_frames[freq_idx] = energy_frames[freq_idx] / energy_max
            
            # 创建Blendshape数据
            blend_data = []
            
            # 生成眨眼事件 - 更自然的眨眼频率
            blink_events = []
            avg_blink_interval = 120  # 每4秒眨一次眼 (30fps * 4)
            for i in range(int(num_frames / avg_blink_interval) + 1):
                # 添加随机性
                base_frame = i * avg_blink_interval
                random_offset = np.random.randint(-20, 20)
                blink_frame = base_frame + random_offset
                if 0 <= blink_frame < num_frames:
                    blink_events.append(blink_frame)
            
            # 生成眼部运动事件
            eye_movement_events = []
            for i in range(int(num_frames / 90) + 1):  # 每3秒一次眼部运动
                movement_frame = i * 90 + np.random.randint(-10, 10)
                if 0 <= movement_frame < num_frames:
                    movement_type = np.random.choice(['look_left', 'look_right', 'look_up', 'look_down'])
                    eye_movement_events.append((movement_frame, movement_type))
            
            # 使用能量包络创建逼真的口型和表情动画
            for frame_idx in range(num_frames):
                time_point = frame_idx / fps
                
                # 获取各频段能量
                low_energy = energy_frames[0, frame_idx]
                mid_energy = energy_frames[1, frame_idx]  
                high_energy = energy_frames[2, frame_idx]
                total_energy = energy_frames[3, frame_idx]
                
                # 创建帧数据
                frame_data = {
                    "frame": frame_idx,
                    "time": time_point,
                    "time_code": frame_idx / fps
                }
                
                # 基于音频能量生成口型动画
                # 主要口型控制
                jaw_open = np.clip(total_energy * 1.2 + low_energy * 0.3, 0.0, 1.0)
                
                # 根据频段分布调整口型
                if total_energy > 0.1:
                    # 有声音时的口型
                    frame_data["JawOpen"] = jaw_open
                    
                    # 基于中频能量的嘴唇动作
                    if mid_energy > 0.3:
                        frame_data["MouthUpperUpLeft"] = mid_energy * 0.6
                        frame_data["MouthUpperUpRight"] = mid_energy * 0.6
                        frame_data["MouthLowerDownLeft"] = mid_energy * 0.8
                        frame_data["MouthLowerDownRight"] = mid_energy * 0.8
                    
                    # 基于高频能量的嘴唇收缩和伸展
                    if high_energy > 0.4:
                        frame_data["MouthStretchLeft"] = high_energy * 0.5
                        frame_data["MouthStretchRight"] = high_energy * 0.5
                    else:
                        frame_data["MouthFunnel"] = (1 - high_energy) * total_energy * 0.4
                        frame_data["MouthPucker"] = (1 - high_energy) * total_energy * 0.3
                    
                    # 基于低频能量的下颌动作  
                    if low_energy > 0.5:
                        frame_data["JawForward"] = low_energy * 0.2
                        # 随机左右颌部动作
                        if np.random.random() < 0.1:
                            if np.random.random() < 0.5:
                                frame_data["JawLeft"] = low_energy * 0.15
                            else:
                                frame_data["JawRight"] = low_energy * 0.15
                    
                    # 嘴角动作 - 基于中频能量
                    if mid_energy > 0.2:
                        mouth_corner_intensity = mid_energy * 0.4
                        # 轻微微笑
                        frame_data["MouthSmileLeft"] = mouth_corner_intensity
                        frame_data["MouthSmileRight"] = mouth_corner_intensity
                        
                        # 偶尔的嘴角拉伸
                        if np.random.random() < 0.05:
                            frame_data["MouthDimpleLeft"] = mouth_corner_intensity * 0.5
                            frame_data["MouthDimpleRight"] = mouth_corner_intensity * 0.5
                    
                    # 嘴唇卷曲和压迫
                    if total_energy > 0.6:
                        roll_intensity = (total_energy - 0.6) * 0.5
                        frame_data["MouthRollLower"] = roll_intensity
                        frame_data["MouthRollUpper"] = roll_intensity * 0.7
                        
                        # 嘴唇压迫
                        if high_energy > 0.5:
                            press_intensity = high_energy * 0.3
                            frame_data["MouthPressLeft"] = press_intensity
                            frame_data["MouthPressRight"] = press_intensity
                    
                    # 嘴唇耸肩动作
                    if total_energy > 0.4 and np.random.random() < 0.02:
                        shrug_intensity = total_energy * 0.3
                        frame_data["MouthShrugLower"] = shrug_intensity
                        frame_data["MouthShrugUpper"] = shrug_intensity * 0.8
                
                else:
                    # 静音时保持自然的嘴部位置
                    frame_data["JawOpen"] = 0.05
                    frame_data["MouthClose"] = 0.8
                
                # 眉毛动作 - 基于总能量和随机事件
                if total_energy > 0.5:
                    # 高能量时偶尔抬眉
                    if np.random.random() < 0.03:
                        brow_intensity = total_energy * 0.4
                        if np.random.random() < 0.7:
                            # 内眉抬起
                            frame_data["BrowInnerUp"] = brow_intensity
                        else:
                            # 外眉抬起
                            frame_data["BrowOuterUpLeft"] = brow_intensity
                            frame_data["BrowOuterUpRight"] = brow_intensity
                
                # 偶尔的皱眉
                if total_energy < 0.2 and np.random.random() < 0.02:
                    frown_intensity = (0.2 - total_energy) * 2
                    frame_data["BrowDownLeft"] = frown_intensity * 0.3
                    frame_data["BrowDownRight"] = frown_intensity * 0.3
                
                # 脸颊动作
                if total_energy > 0.7:
                    cheek_intensity = (total_energy - 0.7) * 0.6
                    # 脸颊鼓起
                    if np.random.random() < 0.02:
                        frame_data["CheekPuff"] = cheek_intensity
                    # 脸颊压缩
                    elif np.random.random() < 0.03:
                        frame_data["CheekSquintLeft"] = cheek_intensity * 0.5
                        frame_data["CheekSquintRight"] = cheek_intensity * 0.5
                
                # 鼻子动作
                if mid_energy > 0.6 and np.random.random() < 0.01:
                    sneer_intensity = mid_energy * 0.3
                    if np.random.random() < 0.5:
                        frame_data["NoseSneerLeft"] = sneer_intensity
                    else:
                        frame_data["NoseSneerRight"] = sneer_intensity
                
                # 舌头动作 - 很少出现
                if high_energy > 0.8 and np.random.random() < 0.005:
                    frame_data["TongueOut"] = high_energy * 0.2
                
                # 处理眨眼动画
                eye_blink_left = 0.0
                eye_blink_right = 0.0
                
                for blink_start in blink_events:
                    distance = frame_idx - blink_start
                    if 0 <= distance < len(self.blink_pattern):
                        blink_value = self.blink_pattern[distance]
                        eye_blink_left = max(eye_blink_left, blink_value)
                        eye_blink_right = max(eye_blink_right, blink_value)
                
                frame_data["EyeBlinkLeft"] = eye_blink_left
                frame_data["EyeBlinkRight"] = eye_blink_right
                
                # 处理眼部运动
                for movement_frame, movement_type in eye_movement_events:
                    # 眼部运动持续约30帧 (1秒)
                    distance = abs(frame_idx - movement_frame)
                    if distance <= 15:
                        # 使用高斯函数创建平滑的眼部运动
                        intensity = np.exp(-(distance**2) / (2 * 5**2)) * 0.8
                        
                        movement_pattern = self.eye_movement_patterns.get(movement_type, {})
                        for blendshape, value in movement_pattern.items():
                            frame_data[blendshape] = intensity * value
                
                # 眼部表情细节
                if total_energy > 0.3:
                    # 说话时轻微的眼部变化
                    eye_intensity = total_energy * 0.1
                    
                    # 随机的眯眼
                    if np.random.random() < 0.01:
                        frame_data["EyeSquintLeft"] = eye_intensity
                        frame_data["EyeSquintRight"] = eye_intensity
                    
                    # 随机的瞪眼
                    if total_energy > 0.8 and np.random.random() < 0.005:
                        frame_data["EyeWideLeft"] = eye_intensity * 2
                        frame_data["EyeWideRight"] = eye_intensity * 2
                
                # 将所有非零值保留，零值也保留以保持完整性
                blend_data.append(frame_data)
            
            # 保存Blendshape数据到CSV
            return self.save_blendshape_to_csv(blend_data, output_csv_path)
        
        except Exception as e:
            logger.error(f"生成完整Blendshape数据时出错: {e}")
            traceback.print_exc()
            return False
    
    def generate_blendshape_fast(self, audio_file_path, output_csv_path, fps=30):
        """
        从音频文件快速生成完整ARKit Blendshape数据，针对流式处理优化
        
        参数:
            audio_file_path: 音频文件路径
            output_csv_path: 输出CSV文件路径
            fps: 动画帧率
            
        返回:
            成功返回True，否则返回False
        """
        try:
            # 读取音频文件
            with wave.open(audio_file_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            
            # 生成完整ARKit Blendshape数据
            return self.generate_blendshape_from_audio(
                audio_samples, 
                output_csv_path,
                sample_rate,
                fps
            )
        except Exception as e:
            logger.error(f"快速生成完整Blendshape数据时出错: {e}")
            traceback.print_exc()
            return False
    
    def save_blendshape_to_csv(self, blendshape_data, output_csv_path):
        """将完整的ARKit Blendshape数据保存为CSV文件"""
        if not blendshape_data:
            logger.error("没有Blendshape数据可保存")
            return False
            
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 获取所有字段名 - 完整的ARKit列表
            fieldnames = ["frame", "time", "time_code"]
            fieldnames.extend(self.blendshape_names)
            
            # 写入CSV文件
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for frame_data in blendshape_data:
                    # 创建一个包含所有ARKit blendshape的行
                    row = {}
                    
                    # 基础字段
                    row["frame"] = frame_data.get("frame", 0)
                    row["time"] = frame_data.get("time", 0.0)
                    row["time_code"] = frame_data.get("time_code", 0.0)
                    
                    # 所有ARKit blendshape字段
                    for bs_name in self.blendshape_names:
                        row[bs_name] = frame_data.get(bs_name, 0.0)
                    
                    writer.writerow(row)
            
           
            logger.info(f"完整ARKit Blendshape数据已保存至: {output_csv_path}")
            logger.info(f"包含 {len(self.blendshape_names)} 个ARKit blendshape和 {len(blendshape_data)} 帧数据")
            return True
        except Exception as e:
            logger.error(f"保存完整Blendshape数据到CSV时出错: {e}")
            traceback.print_exc()
            return False

class Audio2Face3DClient:
   """与Audio2Face-3D gRPC服务交互的客户端，针对流式处理优化"""

   def __init__(self, server_address: str = "localhost:52000", max_buffer_seconds: float = 10.0):
       """
       初始化Audio2Face-3D客户端
       
       参数:
           server_address: Audio2Face-3D gRPC服务器地址
           max_buffer_seconds: 最大处理音频缓冲区长度，超过此长度的音频将被分段处理
       """
       global _blendshape_generator
       
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
       
       # 完整的ARKit Blendshape 名称列表
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
       
       # 初始化完整ARKit Blendshape生成器
       if _blendshape_generator is None:
           _blendshape_generator = SimpleBlendshapeGenerator()
       self.blendshape_generator = _blendshape_generator
       
       # 连接状态
       self.is_connected = False
       
       # 测试连接
       try:
           # 创建简单请求检查连接
           self._test_connection()
           self.is_connected = True
           logger.info("成功连接到Audio2Face-3D服务")
       except Exception as e:
           logger.error(f"连接Audio2Face-3D服务失败: {e}")
           self.is_connected = False
   
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
       
       # 创建面部参数 - 优化参数以获得更自然的表情
       face_params = a2f_pb2.FaceParameters(
           float_params={
               "lowerFaceSmoothing": 0.2,  # 降低平滑度以获得更敏锐的反应
               "upperFaceSmoothing": 0.3,
               "lowerFaceStrength": 1.2,   # 增强下半脸的强度
               "upperFaceStrength": 1.0,
               "faceMaskLevel": 0.5,
               "faceMaskSoftness": 0.3,
               "skinStrength": 1.0,
               "blinkStrength": 1.0,
               "eyelidOpenOffset": 0.0,
               "lipOpenOffset": 0.1,       # 稍微增加嘴唇张开偏移
               "blinkOffset": 0.0,
               "tongueStrength": 1.0,
               "tongueHeightOffset": 0.0,
               "tongueDepthOffset": 0.0
           }
       )
       
       # 创建blendshape参数
       blendshape_params = a2f_pb2.BlendShapeParameters(
           bs_weight_multipliers={
               "JawOpen": 1.3,              # 增加下巴开启幅度
               "MouthSmileLeft": 1.2,
               "MouthSmileRight": 1.2,
               "BrowInnerUp": 1.2,
               "EyeBlinkLeft": 1.0,
               "EyeBlinkRight": 1.0,
               "MouthUpperUpLeft": 1.2,     # 增加上嘴唇运动
               "MouthUpperUpRight": 1.2,
               "MouthLowerDownLeft": 1.3,   # 增加下嘴唇运动
               "MouthLowerDownRight": 1.3
           },
           bs_weight_offsets={},
           enable_clamping_bs_weight=True
       )
       
       # 创建情绪参数
       emotion_params = a2f_pb2.EmotionParameters(
           live_transition_time=0.2,  # 缩短过渡时间以获得更快的响应
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
           emotion_strength=0.8,        # 增加情绪强度
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
       从动画数据中提取完整ARKit Blendshape数据
       
       参数:
           animation_data_list: AnimationData对象列表
           
       返回:
           包含完整blendshape数据的字典列表,每帧一个字典
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
       将完整ARKit Blendshape数据保存为CSV文件
       
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
               
           # 获取所有字段名 - 完整的ARKit列表
           fieldnames = ["frame", "time", "time_code"]
           fieldnames.extend(self.blendshape_names)
           
           # 写入CSV文件
           with open(output_csv_path, 'w', newline='') as csvfile:
               writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
               writer.writeheader()
               
               for frame_data in blendshape_data:
                   # 创建一个包含所有字段的行
                   row = {}
                   for field in fieldnames:
                       row[field] = frame_data.get(field, 0.0)
                   writer.writerow(row)
           
           logger.info(f"完整ARKit Blendshape数据已保存至: {output_csv_path}")
           logger.info(f"包含 {len(self.blendshape_names)} 个blendshape和 {len(blendshape_data)} 帧数据")
           return True
       except Exception as e:
           logger.error(f"保存完整Blendshape数据到CSV时出错: {e}")
           traceback.print_exc()
           return False
   
   # 智能决定使用快速处理还是精确处理
   def smart_process_audio_file_to_blendshape(self, audio_file_path: str, output_csv_path: str, fast_mode=False) -> bool:
       """
       智能处理音频文件并将完整ARKit Blendshape数据保存到CSV，自动选择处理模式
       
       参数:
           audio_file_path: 音频文件路径（WAV）
           output_csv_path: 输出CSV文件路径
           fast_mode: 是否强制使用快速模式
           
       返回:
           成功返回True，否则返回False
       """
       logger.info(f"智能处理音频文件到完整Blendshape: {audio_file_path} -> {output_csv_path}")
       
       # 检测文件长度和内容
       try:
           with wave.open(audio_file_path, 'rb') as wf:
               n_frames = wf.getnframes()
               sample_rate = wf.getframerate()
               duration = n_frames / sample_rate
               
               # 自动决定是否使用快速模式
               # 如果已指定快速模式或服务不可用，使用完整版生成器
               auto_fast_mode = fast_mode or not self.is_connected
               
               # 对于短音频（<0.5秒）或时间紧迫的场景，使用快速模式
               if duration < 0.5 or auto_fast_mode:
                   logger.info(f"使用完整ARKit快速处理模式生成Blendshape数据，音频长度: {duration:.2f}秒")
                   return self.blendshape_generator.generate_blendshape_fast(
                       audio_file_path, 
                       output_csv_path
                   )
               else:
                   # 使用完整A2F处理获得更准确的结果
                   logger.info(f"使用A2F精确处理模式生成完整Blendshape数据，音频长度: {duration:.2f}秒")
                   return self.process_audio_file_to_blendshape(
                       audio_file_path, 
                       output_csv_path
                   )
       except Exception as e:
           logger.error(f"智能处理音频文件时出错: {e}")
           
           # 出错时尝试使用完整版处理
           try:
               logger.info("使用完整ARKit Blendshape生成器作为后备")
               return self.blendshape_generator.generate_blendshape_fast(
                   audio_file_path, 
                   output_csv_path
               )
           except Exception as e2:
               logger.error(f"后备处理也失败: {e2}")
               return False
   
   def process_audio_file_to_blendshape(self, audio_file_path: str, output_csv_path: str) -> bool:
       """
       处理音频文件并将完整ARKit Blendshape数据保存到CSV
       
       参数:
           audio_file_path: 音频文件路径（WAV）
           output_csv_path: 输出CSV文件路径
           
       返回:
           成功返回True，否则返回False
       """
       logger.info(f"处理音频文件并转换为完整ARKit Blendshape数据: {audio_file_path} -> {output_csv_path}")
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
               
               # 尝试使用备选方案
               logger.info("尝试使用完整ARKit Blendshape生成器")
               return self.blendshape_generator.generate_blendshape_fast(
                   audio_file_path, 
                   output_csv_path
               )
               
           logger.info(f"成功获取 {len(animation_data_list)} 个动画数据帧")
           
           # 提取完整ARKit Blendshape数据
           blendshape_data = self.extract_blendshape_data(animation_data_list)
           
           if not blendshape_data:
               logger.error("无法从动画数据提取Blendshape数据")
               
               # 尝试备选方案
               logger.info("尝试使用完整ARKit Blendshape生成器")
               return self.blendshape_generator.generate_blendshape_fast(
                   audio_file_path, 
                   output_csv_path
               )
               
           logger.info(f"成功提取 {len(blendshape_data)} 帧完整ARKit Blendshape数据")
           
           # 保存完整ARKit Blendshape数据到CSV
           return self.save_blendshape_to_csv(blendshape_data, output_csv_path)
       except Exception as e:
           logger.error(f"处理音频到完整Blendshape数据时出错: {e}")
           traceback.print_exc()
           
           # 出错时尝试使用完整版处理
           try:
               logger.info("尝试使用完整ARKit Blendshape生成器作为后备")
               return self.blendshape_generator.generate_blendshape_fast(
                   audio_file_path, 
                   output_csv_path
               )
           except Exception as e2:
               logger.error(f"后备处理也失败: {e2}")
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
       处理长音频文件并保存完整ARKit Blendshape数据到CSV，自动处理分段
       
       参数:
           audio_file_path: 音频文件路径（WAV）
           output_csv_path: 输出CSV文件路径
           
       返回:
           成功返回True，否则返回False
       """
       logger.info(f"处理长音频文件到完整ARKit Blendshape数据: {audio_file_path} -> {output_csv_path}")
       
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
           
           # 尝试使用备选方案
           logger.info("尝试使用完整ARKit Blendshape生成器")
           return self.blendshape_generator.generate_blendshape_fast(
               audio_file_path, 
               output_csv_path
           )
       
       logger.info(f"所有段处理完成，总共获取 {len(all_animation_data)} 个动画数据帧")
       
       # 提取完整ARKit Blendshape数据
       blendshape_data = self.extract_blendshape_data(all_animation_data)
       
       if not blendshape_data:
           logger.error("无法从动画数据提取Blendshape数据")
           
           # 尝试备选方案
           logger.info("尝试使用完整ARKit Blendshape生成器")
           return self.blendshape_generator.generate_blendshape_fast(
               audio_file_path, 
               output_csv_path
           )
           
       logger.info(f"成功提取 {len(blendshape_data)} 帧完整ARKit Blendshape数据")
       
       # 保存完整ARKit Blendshape数据到CSV
       return self.save_blendshape_to_csv(blendshape_data, output_csv_path)

# 获取单例客户端实例，优化初始化性能
def get_audio2face_client(server_address="localhost:52000", max_buffer_seconds=10.0):
   """获取Audio2Face客户端单例，避免重复初始化"""
   global _client_instance
   
   if _client_instance is None:
       logger.info("创建新的Audio2Face客户端实例")
       _client_instance = Audio2Face3DClient(server_address, max_buffer_seconds)
   
   return _client_instance

# 异步处理音频文件为完整ARKit Blendshape数据，支持超时和回退策略
async def async_process_audio_to_blendshape(audio_file_path, output_csv_path, server_address="localhost:52000", timeout=5.0):
   """
   异步处理音频文件为完整ARKit Blendshape数据，带有超时和回退策略
   
   参数:
       audio_file_path: 音频文件路径
       output_csv_path: 输出CSV文件路径
       server_address: A2F服务器地址
       timeout: 处理超时时间（秒）
       
   返回:
       成功返回True，否则返回False
   """
   global _blendshape_generator  # 先声明全局变量
   
   try:
       # 获取客户端实例
       client = get_audio2face_client(server_address)
       
       # 首先尝试使用智能处理（会自动选择最佳方式）
       processing_task = asyncio.create_task(
           asyncio.to_thread(
               client.smart_process_audio_file_to_blendshape,
               audio_file_path,
               output_csv_path
           )
       )
       
       # 使用超时控制
       try:
           success = await asyncio.wait_for(processing_task, timeout=timeout)
           return success
       except asyncio.TimeoutError:
           logger.warning(f"处理音频超时（{timeout}秒），使用完整ARKit Blendshape生成器")
           
           # 超时时使用完整版生成器
           return await asyncio.to_thread(
               client.blendshape_generator.generate_blendshape_fast,
               audio_file_path,
               output_csv_path
           )
   except Exception as e:
       logger.error(f"异步处理音频到完整ARKit Blendshape时出错: {e}")
       traceback.print_exc()
       
       # 尝试使用完整版生成器作为最后的备选方案
       try:
           logger.info("使用完整ARKit Blendshape生成器作为后备")
           if _blendshape_generator is None:
               _blendshape_generator = SimpleBlendshapeGenerator()
               
           return await asyncio.to_thread(
               _blendshape_generator.generate_blendshape_fast,
               audio_file_path,
               output_csv_path
           )
       except Exception as e2:
           logger.error(f"后备处理也失败: {e2}")
           return False

# 使用示例
if __name__ == "__main__":
   client = Audio2Face3DClient("localhost:52000", max_buffer_seconds=10.0)
   success = client.smart_process_audio_file_to_blendshape(
       "example.wav", 
       "example_blendshape.csv"
   )
   print(f"处理结果: {success}")
   client.close()

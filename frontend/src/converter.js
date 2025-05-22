import {
	AnimationClip,
	NumberKeyframeTrack,
	InterpolateLinear,
	NormalAnimationBlendMode
} from 'three';

// 设置FPS以确保与音频同步
var fps = 30;

// 缓存blendshape键名称映射
const keyNameCache = {};

// 跟踪上一个动画片段的最后一帧状态，用于平滑过渡
let lastFrameValues = {};

function modifiedKey(key) {
  // 使用缓存避免重复字符串处理
  if (keyNameCache[key] !== undefined) {
    return keyNameCache[key];
  }
  
  let result;
  if (["eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight"].includes(key)) {
    result = key;
  } else if (key.endsWith("Right")) {
    result = key.replace("Right", "_R");
  } else if (key.endsWith("Left")) {
    result = key.replace("Left", "_L");
  } else {
    result = key;
  }
  
  // 缓存结果避免重复计算
  keyNameCache[key] = result;
  return result;
}

// 创建动画，优化以支持流式处理和平滑过渡
function createAnimation(recordedData, morphTargetDictionary, bodyPart, isStreamingMode = true) {
  if (!recordedData || recordedData.length === 0 || !morphTargetDictionary) {
    return null;
  }

  try {
    // 创建blendshapes到索引的映射
    const morphTargetMapping = new Map();
    for (const [key, index] of Object.entries(morphTargetDictionary)) {
      morphTargetMapping.set(key, index);
    }
    
    // 建立通道数据映射
    const trackDataMap = new Map();
    const time = [];
    
    // 第一遍处理数据: 收集时间点和所有blendshape数据
    for (let frameIndex = 0; frameIndex < recordedData.length; frameIndex++) {
      const frame = recordedData[frameIndex];
      time.push(frame.time);
      
      // 处理该帧的所有blendshapes
      for (const [key, value] of Object.entries(frame.blendshapes)) {
        const modKey = modifiedKey(key);
        
        if (!morphTargetMapping.has(modKey)) {
          continue; // 跳过未知的blendshape
        }
        
        const index = morphTargetMapping.get(modKey);
        
        if (!trackDataMap.has(index)) {
          // 为该blendshape创建一个新数组，长度等于总帧数
          trackDataMap.set(index, new Array(recordedData.length).fill(0));
        }
        
        // 保存当前帧的值
        trackDataMap.get(index)[frameIndex] = value;
      }
    }
    
    // 支持流式处理的平滑过渡逻辑
    if (isStreamingMode && Object.keys(lastFrameValues).length > 0) {
      // 获取当前片段的第一帧时间
      const firstFrameTime = recordedData[0].time;
      
      // 添加过渡帧
      if (firstFrameTime > 0 && time[0] !== 0) {
        // 在开头添加过渡帧
        time.unshift(0);
        
        // 为所有轨道添加过渡值
        for (const [index, values] of trackDataMap.entries()) {
          const trackName = `${bodyPart}.morphTargetInfluences[${index}]`;
          const lastValue = lastFrameValues[trackName] || 0;
          
          // 向数组开头添加最后一帧的值用于平滑过渡
          trackDataMap.get(index).unshift(lastValue);
        }
      }
    }
    
    // 创建动画轨道
    const tracks = [];
    
    // 根据收集的数据创建轨道
    for (const [index, values] of trackDataMap.entries()) {
      // 过滤掉全部为0的轨道以减小动画大小
      if (values.every(v => v === 0)) {
        continue;
      }
      
      const trackName = `${bodyPart}.morphTargetInfluences[${index}]`;
      const track = new NumberKeyframeTrack(trackName, time, values);
      
      // 添加插值和混合模式设置以改善过渡
      track.interpolation = InterpolateLinear;
      tracks.push(track);
      
      // 更新最后一帧值用于下一个片段
      if (isStreamingMode) {
        lastFrameValues[trackName] = values[values.length - 1];
      }
    }
    
    // 如果没有轨道，返回null
    if (tracks.length === 0) {
      return null;
    }
    
    // 获取动画总时长
    const lastTimePoint = recordedData[recordedData.length - 1].time;
    
    // 创建动画剪辑
    const clip = new AnimationClip(`animation_${Date.now()}`, lastTimePoint, tracks);
    
    // 设置混合模式以改善多个片段之间的过渡
    clip.blendMode = NormalAnimationBlendMode;
    
    return clip;
  } catch (error) {
    console.error("创建动画时出错:", error);
    return null;
  }
}

// 重置流式动画状态
function resetAnimationState() {
  // 清空最后一帧的值
  lastFrameValues = {};
  console.log("重置动画状态");
}

// 获取动画片段的持续时间
function getAnimationDuration(recordedData) {
  if (!recordedData || recordedData.length === 0) {
    return 0;
  }
  return recordedData[recordedData.length - 1].time;
}

// 计算以确保动画和音频同步
function calculateAnimationTimingInfo(recordedData) {
  if (!recordedData || recordedData.length === 0) {
    return { duration: 0, frameCount: 0, fps: 30 };
  }
  
  const duration = recordedData[recordedData.length - 1].time;
  const frameCount = recordedData.length;
  const calculatedFps = frameCount / duration;
  
  return {
    duration,
    frameCount,
    fps: calculatedFps
  };
}

// 创建用于平滑过渡的空白动画
function createEmptyTransitionAnimation(morphTargetDictionary, bodyPart, duration = 0.2) {
  try {
    // 创建blendshapes到索引的映射
    const morphTargetMapping = new Map();
    for (const [key, index] of Object.entries(morphTargetDictionary)) {
      morphTargetMapping.set(key, index);
    }
    
    // 创建轨道
    const tracks = [];
    const time = [0, duration];
    
    // 为每个重要的blendshape创建轨道
    for (const [trackName, lastValue] of Object.entries(lastFrameValues)) {
      // 从trackName中提取索引 (bodyPart.morphTargetInfluences[index])
      const indexMatch = trackName.match(/\[(\d+)\]$/);
      if (!indexMatch) continue;
      
      const index = parseInt(indexMatch[1]);
      const values = [lastValue, 0]; // 从上一个值过渡到0
      
      const track = new NumberKeyframeTrack(trackName, time, values);
      tracks.push(track);
    }
    
    // 如果没有轨道，返回null
    if (tracks.length === 0) {
      return null;
    }
    
    // 创建过渡动画剪辑
    const clip = new AnimationClip(`transition_${Date.now()}`, duration, tracks);
    return clip;
  } catch (error) {
    console.error("创建过渡动画时出错:", error);
    return null;
  }
}

export { 
  createAnimation, 
  resetAnimationState, 
  getAnimationDuration,
  calculateAnimationTimingInfo,
  createEmptyTransitionAnimation
};

export default createAnimation;

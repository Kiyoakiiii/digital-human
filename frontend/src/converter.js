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
    console.warn("createAnimation: 输入数据无效", {
      recordedDataLength: recordedData ? recordedData.length : 0,
      hasMorphTargetDictionary: !!morphTargetDictionary,
      bodyPart
    });
    return null;
  }

  try {
    console.log(`创建动画: ${bodyPart}, 数据帧数: ${recordedData.length}, 流式模式: ${isStreamingMode}`);
    
    // 创建blendshapes到索引的映射
    const morphTargetMapping = new Map();
    for (const [key, index] of Object.entries(morphTargetDictionary)) {
      morphTargetMapping.set(key, index);
    }
    
    console.log(`形态目标映射创建完成，共 ${morphTargetMapping.size} 个目标`);
    
    // 建立通道数据映射
    const trackDataMap = new Map();
    const time = [];
    
    // 第一遍处理数据: 收集时间点和所有blendshape数据
    for (let frameIndex = 0; frameIndex < recordedData.length; frameIndex++) {
      const frame = recordedData[frameIndex];
      
      // 获取时间点
      let timePoint = frame.time;
      if (typeof timePoint !== 'number') {
        timePoint = frameIndex / fps; // 如果时间点无效，使用帧索引计算
      }
      time.push(timePoint);
      
      // 处理该帧的所有blendshapes
      if (frame.blendshapes && typeof frame.blendshapes === 'object') {
        for (const [key, value] of Object.entries(frame.blendshapes)) {
          // 跳过非数字值
          if (typeof value !== 'number' || isNaN(value)) {
            continue;
          }
          
          const modKey = modifiedKey(key);
          
          if (!morphTargetMapping.has(modKey)) {
            continue; // 跳过未知的blendshape
          }
          
          const index = morphTargetMapping.get(modKey);
          
          if (!trackDataMap.has(index)) {
            // 为该blendshape创建一个新数组，长度等于总帧数
            trackDataMap.set(index, new Array(recordedData.length).fill(0));
          }
          
          // 保存当前帧的值，限制在合理范围内
          const clampedValue = Math.max(0, Math.min(1, value));
          trackDataMap.get(index)[frameIndex] = clampedValue;
        }
      }
    }
    
    console.log(`数据处理完成，共创建 ${trackDataMap.size} 个轨道`);
    
    // 支持流式处理的平滑过渡逻辑
    if (isStreamingMode && Object.keys(lastFrameValues).length > 0) {
      console.log("应用流式处理平滑过渡");
      
      // 获取当前片段的第一帧时间
      const firstFrameTime = time[0];
      
      // 添加过渡帧
      if (firstFrameTime > 0.1) { // 如果第一帧时间大于0.1秒，添加过渡
        // 在开头添加过渡帧
        time.unshift(0);
        
        // 为所有轨道添加过渡值
        for (const [index, values] of trackDataMap.entries()) {
          const trackName = `${bodyPart}.morphTargetInfluences[${index}]`;
          const lastValue = lastFrameValues[trackName] || 0;
          
          // 向数组开头添加最后一帧的值用于平滑过渡
          values.unshift(lastValue);
        }
        
        console.log("添加过渡帧完成");
      }
    }
    
    // 创建动画轨道
    const tracks = [];
    let activeTrackCount = 0;
    
    // 根据收集的数据创建轨道
    for (const [index, values] of trackDataMap.entries()) {
      // 检查是否有有效数据
      const hasValidData = values.some(v => v > 0.001); // 设置较小的阈值
      
      if (!hasValidData) {
        continue; // 跳过没有明显变化的轨道
      }
      
      const trackName = `${bodyPart}.morphTargetInfluences[${index}]`;
      
      // 创建轨道
      const track = new NumberKeyframeTrack(trackName, time, values);
      
      // 添加插值和混合模式设置以改善过渡
      track.interpolation = InterpolateLinear;
      tracks.push(track);
      activeTrackCount++;
      
      // 更新最后一帧值用于下一个片段
      if (isStreamingMode) {
        lastFrameValues[trackName] = values[values.length - 1];
      }
    }
    
    console.log(`创建了 ${activeTrackCount} 个活动轨道`);
    
    // 如果没有轨道，返回null
    if (tracks.length === 0) {
      console.warn("没有创建任何动画轨道");
      return null;
    }
    
    // 获取动画总时长
    const lastTimePoint = time[time.length - 1];
    
    // 创建动画剪辑
    const clip = new AnimationClip(`animation_${bodyPart}_${Date.now()}`, lastTimePoint, tracks);
    
    // 设置混合模式以改善多个片段之间的过渡
    clip.blendMode = NormalAnimationBlendMode;
    
    console.log(`动画剪辑创建成功: ${clip.name}, 持续时间: ${lastTimePoint.toFixed(2)}秒, 轨道数: ${tracks.length}`);
    
    return clip;
  } catch (error) {
    console.error("创建动画时出错:", error);
    console.error("错误堆栈:", error.stack);
    return null;
  }
}

// 重置流式动画状态
function resetAnimationState() {
  // 清空最后一帧的值
  const previousCount = Object.keys(lastFrameValues).length;
  lastFrameValues = {};
  console.log(`重置动画状态，清理了 ${previousCount} 个缓存值`);
}

// 获取动画片段的持续时间
function getAnimationDuration(recordedData) {
  if (!recordedData || recordedData.length === 0) {
    return 0;
  }
  
  const lastFrame = recordedData[recordedData.length - 1];
  return lastFrame.time || (recordedData.length - 1) / fps;
}

// 计算以确保动画和音频同步
function calculateAnimationTimingInfo(recordedData) {
  if (!recordedData || recordedData.length === 0) {
    return { duration: 0, frameCount: 0, fps: 30 };
  }
  
  const duration = getAnimationDuration(recordedData);
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
    console.log(`创建过渡动画: ${bodyPart}, 持续时间: ${duration}秒`);
    
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
      track.interpolation = InterpolateLinear;
      tracks.push(track);
    }
    
    // 如果没有轨道，返回null
    if (tracks.length === 0) {
      console.log("没有需要过渡的轨道");
      return null;
    }
    
    // 创建过渡动画剪辑
    const clip = new AnimationClip(`transition_${bodyPart}_${Date.now()}`, duration, tracks);
    
    console.log(`过渡动画创建成功: ${tracks.length} 个轨道`);
    return clip;
  } catch (error) {
    console.error("创建过渡动画时出错:", error);
    return null;
  }
}

// 验证动画数据的完整性
function validateAnimationData(recordedData) {
  if (!recordedData || !Array.isArray(recordedData)) {
    return { valid: false, error: "数据不是有效数组" };
  }
  
  if (recordedData.length === 0) {
    return { valid: false, error: "数据数组为空" };
  }
  
  let validFrames = 0;
  let totalBlendshapes = 0;
  
  for (let i = 0; i < recordedData.length; i++) {
    const frame = recordedData[i];
    
    if (!frame || typeof frame !== 'object') {
      continue;
    }
    
    if (frame.blendshapes && typeof frame.blendshapes === 'object') {
      const blendshapeCount = Object.keys(frame.blendshapes).length;
      if (blendshapeCount > 0) {
        validFrames++;
        totalBlendshapes += blendshapeCount;
      }
    }
  }
  
  return {
    valid: validFrames > 0,
    validFrames,
    totalFrames: recordedData.length,
    averageBlendshapes: validFrames > 0 ? Math.round(totalBlendshapes / validFrames) : 0,
    error: validFrames === 0 ? "没有有效的blendshape数据" : null
  };
}

export { 
  createAnimation, 
  resetAnimationState, 
  getAnimationDuration,
  calculateAnimationTimingInfo,
  createEmptyTransitionAnimation,
  validateAnimationData
};

export default createAnimation;

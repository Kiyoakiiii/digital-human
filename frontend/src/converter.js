import {
	AnimationClip,
	NumberKeyframeTrack
} from 'three';

// 设置FPS以确保与音频同步
var fps = 30;

// 缓存blendshape键名称映射
const keyNameCache = {};

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

// 创建动画 - 保持高质量
function createAnimation(recordedData, morphTargetDictionary, bodyPart) {
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
      tracks.push(track);
    }
    
    // 如果没有轨道，返回null
    if (tracks.length === 0) {
      return null;
    }
    
    // 获取动画总时长
    const lastTimePoint = recordedData[recordedData.length - 1].time;
    
    // 创建并返回动画剪辑
    const clip = new AnimationClip('animation', lastTimePoint, tracks);
    return clip;
  } catch (error) {
    console.error("创建动画时出错:", error);
    return null;
  }
}

export default createAnimation;

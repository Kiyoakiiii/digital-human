import {
	AnimationClip,
	NumberKeyframeTrack
} from 'three';

// 修改FPS以确保与音频同步
var fps = 30;

// 优化: 缓存已转换的key名称
const keyNameCache = {};

function modifiedKey(key) {
  // 使用缓存避免重复处理
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
  
  // 缓存结果
  keyNameCache[key] = result;
  return result;
}

// 优化: 加速动画创建
function createAnimation(recordedData, morphTargetDictionary, bodyPart) {
  console.log(`生成动画，使用FPS: ${fps}`);

  if (!recordedData || recordedData.length === 0) {
    console.warn("没有动画数据可处理");
    return null;
  }

  if (!morphTargetDictionary) {
    console.warn("缺少morphTargetDictionary");
    return null;
  }

  try {
    // 预处理: 创建blendshapes到索引的映射
    const morphTargetMapping = new Map();
    for (const [key, index] of Object.entries(morphTargetDictionary)) {
      morphTargetMapping.set(key, index);
    }
    
    // 跳帧处理: 如果帧数太多，进行抽样
    const maxFrames = 300;
    let frameStep = 1;
    let processedData = recordedData;
    
    if (recordedData.length > maxFrames) {
      frameStep = Math.ceil(recordedData.length / maxFrames);
      processedData = recordedData.filter((_, i) => i % frameStep === 0);
      console.log(`优化: 从 ${recordedData.length} 帧减少到 ${processedData.length} 帧`);
    }

    // 建立通道数据映射
    const trackDataMap = new Map();
    const time = [];
    
    // 计算音频总时长
    const lastTimePoint = processedData[processedData.length - 1].time;
    console.log(`动画总时长: ${lastTimePoint}秒，帧数: ${processedData.length}`);
    
    // 第一遍处理数据: 收集时间点和所有blendshape数据
    for (let frameIndex = 0; frameIndex < processedData.length; frameIndex++) {
      const frame = processedData[frameIndex];
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
          trackDataMap.set(index, new Array(processedData.length).fill(0));
        }
        
        // 增强某些表情
        let enhancedValue = value;
        if (key === 'mouthShrugUpper') {
          enhancedValue += 0.4;
        }
        
        // 保存当前帧的值
        trackDataMap.get(index)[frameIndex] = enhancedValue;
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
      console.warn("没有生成动画轨道");
      return null;
    }
    
    // 创建并返回动画剪辑
    const clip = new AnimationClip('animation', lastTimePoint, tracks);
    console.log(`创建动画剪辑成功，持续时间: ${lastTimePoint}秒，轨道数: ${tracks.length}`);
    return clip;
  } catch (error) {
    console.error("创建动画时出错:", error);
    return null;
  }
}

export default createAnimation;

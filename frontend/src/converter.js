import {
	AnimationClip,
	BooleanKeyframeTrack,
	ColorKeyframeTrack,
	NumberKeyframeTrack,
	Vector3,
	VectorKeyframeTrack
} from 'three';

// 修改FPS以确保与音频同步 - 这需要与后端生成的帧率匹配
var fps = 30;  // 修改为与后端实际生成的帧率相匹配

function modifiedKey(key) {
  if (["eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight"].includes(key)) {
    return key;
  }

  if (key.endsWith("Right")) {
    return key.replace("Right", "_R");
  }
  if (key.endsWith("Left")) {
    return key.replace("Left", "_L");
  }
  return key;
}

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
    // 初始化动画数组
    let animation = [];
    for (let i = 0; i < Object.keys(morphTargetDictionary).length; i++) {
      animation.push([]);
    }
    
    let time = [];
    let finishedFrames = 0;
    
    // 计算音频总时长（用于确保动画长度与音频匹配）
    const lastTimePoint = recordedData[recordedData.length - 1].time;
    console.log(`动画总时长: ${lastTimePoint}秒，帧数: ${recordedData.length}`);
    
    // 处理所有帧
    recordedData.forEach((d, i) => {
      // 使用实际的时间戳而不是假设的帧率
      time.push(d.time);
      
      // 处理每一帧的blendshapes
      Object.entries(d.blendshapes).forEach(([key, value]) => {
        if (!(modifiedKey(key) in morphTargetDictionary)) {
          return;
        }
        
        // 稍微增强某些表情
        if (key === 'mouthShrugUpper') {
          value += 0.4;
        }
        
        // 将值添加到对应的动画通道
        animation[morphTargetDictionary[modifiedKey(key)]].push(value);
      });
      
      finishedFrames++;
    });
    
    // 创建动画轨道
    let tracks = [];
    
    Object.entries(morphTargetDictionary).forEach(([key, index]) => {
      // 确保此blendshape在数据中存在
      const blendshapeKey = key;
      const originalKey = key.endsWith("_R") 
        ? key.replace("_R", "Right") 
        : key.endsWith("_L") 
          ? key.replace("_L", "Left") 
          : key;
          
      // 检查是否至少有一帧包含这个blendshape
      let hasData = false;
      for (let frame of recordedData) {
        if (frame.blendshapes[originalKey] !== undefined) {
          hasData = true;
          break;
        }
      }
      
      if (hasData) {
        // 创建此blendshape的动画轨道
        const trackName = `${bodyPart}.morphTargetInfluences[${index}]`;
        const track = new NumberKeyframeTrack(trackName, time, animation[index]);
        tracks.push(track);
      }
    });
    
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

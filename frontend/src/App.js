import React, { Suspense, useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, useTexture, Loader, Environment, useFBX, useAnimations, OrthographicCamera } from '@react-three/drei';
import { MeshStandardMaterial } from 'three/src/materials/MeshStandardMaterial';

import { LinearEncoding, sRGBEncoding } from 'three/src/constants';
import { LineBasicMaterial, MeshPhysicalMaterial, Vector2 } from 'three';

import createAnimation, { resetAnimationState } from './converter';
import blinkData from './blendDataBlink.json';

import * as THREE from 'three';
import axios from 'axios';
const _ = require('lodash');

// 硬编码服务器地址 - 这里使用你服务器的实际IP
const SERVER_IP = "172.16.10.158";
const host = `http://${SERVER_IP}:5000`;
console.log("后端API地址:", host);

// WebSocket配置
const wsUrl = `ws://${SERVER_IP}:5000/ws`;
let websocket = null;
let clientId = 'client_' + Math.random().toString(36).substr(2, 9);

// 录音相关变量
let mediaRecorder = null;
let audioChunks = [];

// 修改Avatar组件以支持流式动画处理
function Avatar({ avatar_url, speak, setSpeak, text, playing, setPlaying, setResponse, setAnimReady, animationData, setAudioElement, isAudioPlaying, currentSegmentIndex }) {
  let gltf = useGLTF(avatar_url);
  let morphTargetDictionaryBody = null;
  let morphTargetDictionaryLowerTeeth = null;
  const mixerRef = useRef(null);
  
  // 存储当前活动的动画clips
  const activeClipsRef = useRef([]);
  const segmentClipsRef = useRef(new Map()); // 存储每个片段的clips

  const [
    bodyTexture,
    eyesTexture,
    teethTexture,
    bodySpecularTexture,
    bodyRoughnessTexture,
    bodyNormalTexture,
    teethNormalTexture,
    hairTexture,
    tshirtDiffuseTexture,
    tshirtNormalTexture,
    tshirtRoughnessTexture,
    hairAlphaTexture,
    hairNormalTexture,
    hairRoughnessTexture,
    ] = useTexture([
    "/images/body.webp",
    "/images/eyes.webp",
    "/images/teeth_diffuse.webp",
    "/images/body_specular.webp",
    "/images/body_roughness.webp",
    "/images/body_normal.webp",
    "/images/teeth_normal.webp",
    "/images/h_color.webp",
    "/images/tshirt_diffuse.webp",
    "/images/tshirt_normal.webp",
    "/images/tshirt_roughness.webp",
    "/images/h_alpha.webp",
    "/images/h_normal.webp",
    "/images/h_roughness.webp",
  ]);

  _.each([
    bodyTexture,
    eyesTexture,
    teethTexture,
    teethNormalTexture,
    bodySpecularTexture,
    bodyRoughnessTexture,
    bodyNormalTexture,
    tshirtDiffuseTexture,
    tshirtNormalTexture,
    tshirtRoughnessTexture,
    hairAlphaTexture,
    hairNormalTexture,
    hairRoughnessTexture
  ], t => {
    t.encoding = sRGBEncoding;
    t.flipY = false;
  });

  bodyNormalTexture.encoding = LinearEncoding;
  tshirtNormalTexture.encoding = LinearEncoding;
  teethNormalTexture.encoding = LinearEncoding;
  hairNormalTexture.encoding = LinearEncoding;

  gltf.scene.traverse(node => {
    if(node.type === 'Mesh' || node.type === 'LineSegments' || node.type === 'SkinnedMesh') {
      node.castShadow = true;
      node.receiveShadow = true;
      node.frustumCulled = false;

      if (node.name.includes("Body")) {
        node.castShadow = true;
        node.receiveShadow = true;
        node.material = new MeshPhysicalMaterial();
        node.material.map = bodyTexture;
        node.material.roughness = 1.7;
        node.material.roughnessMap = bodyRoughnessTexture;
        node.material.normalMap = bodyNormalTexture;
        node.material.normalScale = new Vector2(0.6, 0.6);
        morphTargetDictionaryBody = node.morphTargetDictionary;
        node.material.envMapIntensity = 0.8;
      }

      if (node.name.includes("Eyes")) {
        node.material = new MeshStandardMaterial();
        node.material.map = eyesTexture;
        node.material.roughness = 0.1;
        node.material.envMapIntensity = 0.5;
      }

      if (node.name.includes("Brows")) {
        node.material = new LineBasicMaterial({color: 0x000000});
        node.material.linewidth = 1;
        node.material.opacity = 0.5;
        node.material.transparent = true;
        node.visible = false;
      }

      if (node.name.includes("Teeth")) {
        node.receiveShadow = true;
        node.castShadow = true;
        node.material = new MeshStandardMaterial();
        node.material.roughness = 0.1;
        node.material.map = teethTexture;
        node.material.normalMap = teethNormalTexture;
        node.material.envMapIntensity = 0.7;
      }

      if (node.name.includes("Hair")) {
        node.material = new MeshStandardMaterial();
        node.material.map = hairTexture;
        node.material.alphaMap = hairAlphaTexture;
        node.material.normalMap = hairNormalTexture;
        node.material.roughnessMap = hairRoughnessTexture;
        node.material.transparent = true;
        node.material.depthWrite = false;
        node.material.side = 2;
        node.material.color.setHex(0x000000);
        node.material.envMapIntensity = 0.3;
      }

      if (node.name.includes("TeethLower")) {
        morphTargetDictionaryLowerTeeth = node.morphTargetDictionary;
      }

      if (node.name.includes("TSHIRT")) {
        node.material = new MeshStandardMaterial();
        node.material.map = tshirtDiffuseTexture;
        node.material.roughnessMap = tshirtRoughnessTexture;
        node.material.normalMap = tshirtNormalTexture;
        node.material.color.setHex(0xffffff);
        node.material.envMapIntensity = 0.5;
      }
    }
  });

  const mixer = useMemo(() => {
    const newMixer = new THREE.AnimationMixer(gltf.scene);
    mixerRef.current = newMixer;
    return newMixer;
  }, []);

  // 设置基础动画（眨眼和闲置动画）
  let idleFbx = useFBX('/idle.fbx');
  let { clips: idleClips } = useAnimations(idleFbx.animations);

  idleClips[0].tracks = _.filter(idleClips[0].tracks, track => {
    return track.name.includes("Head") || track.name.includes("Neck") || track.name.includes("Spine2");
  });

  idleClips[0].tracks = _.map(idleClips[0].tracks, track => {
    if (track.name.includes("Head")) {
      track.name = "head.quaternion";
    }

    if (track.name.includes("Neck")) {
      track.name = "neck.quaternion";
    }

    if (track.name.includes("Spine")) {
      track.name = "spine2.quaternion";
    }

    return track;
  });

  // 初始化基础动画
  useEffect(() => {
    if (!mixer || !morphTargetDictionaryBody) return;

    let idleClipAction = mixer.clipAction(idleClips[0]);
    idleClipAction.play();

    let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body', false);
    if (blinkClip) {
      let blinkAction = mixer.clipAction(blinkClip);
      blinkAction.play();
    }

    setAnimReady(true);
  }, [mixer, morphTargetDictionaryBody]);

  // 处理新的动画数据 - 关键修改点
  useEffect(() => {
    if (!animationData || !animationData.blendData || !morphTargetDictionaryBody || !morphTargetDictionaryLowerTeeth) {
      return;
    }

    console.log("处理新动画数据片段，帧数:", animationData.blendData.length);
    console.log("动画数据示例:", animationData.blendData.slice(0, 2));

    try {
      // 停止之前的表情动画（保留基础动画）
      activeClipsRef.current.forEach(clip => {
        const action = mixer.existingAction(clip);
        if (action) {
          action.stop();
          action.reset();
        }
      });

      // 创建新的动画剪辑
      const bodyClip = createAnimation(animationData.blendData, morphTargetDictionaryBody, 'HG_Body', true);
      const teethClip = createAnimation(animationData.blendData, morphTargetDictionaryLowerTeeth, 'HG_TeethLower', true);

      const newClips = [bodyClip, teethClip].filter(clip => clip !== null);

      if (newClips.length > 0) {
        console.log("动画剪辑已创建:", newClips.map(c => `${c.tracks.length}个轨道`));
        
        // 存储新的活动clips
        activeClipsRef.current = newClips;

        // 播放新的动画
        newClips.forEach(clip => {
          if (clip && clip.tracks && clip.tracks.length > 0) {
            const clipAction = mixer.clipAction(clip);
            clipAction.setLoop(THREE.LoopOnce);
            clipAction.clampWhenFinished = true;
            clipAction.reset();
            clipAction.play();
            
            console.log(`播放动画剪辑: ${clip.tracks.length}个轨道, 持续时间: ${clip.duration.toFixed(2)}秒`);
          }
        });
      } else {
        console.warn("无法创建动画剪辑");
      }
    } catch (error) {
      console.error("处理动画数据时出错:", error);
    }
  }, [animationData, mixer, morphTargetDictionaryBody, morphTargetDictionaryLowerTeeth]);

  // 重置动画状态
  const resetAllAnimations = useCallback(() => {
    console.log("重置所有动画");
    
    // 停止所有表情动画
    activeClipsRef.current.forEach(clip => {
      const action = mixer.existingAction(clip);
      if (action) {
        action.stop();
        action.reset();
      }
    });

    // 清空活动动画列表
    activeClipsRef.current = [];
    segmentClipsRef.current.clear();

    // 重置动画状态
    resetAnimationState();

    // 确保基础动画继续播放
    if (idleClips[0]) {
      let idleClipAction = mixer.clipAction(idleClips[0]);
      idleClipAction.play();
    }

    if (morphTargetDictionaryBody) {
      let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body', false);
      if (blinkClip) {
        let blinkAction = mixer.clipAction(blinkClip);
        blinkAction.play();
      }
    }
  }, [mixer, idleClips, morphTargetDictionaryBody]);

  // 监听播放状态变化
  useEffect(() => {
    if (!isAudioPlaying && activeClipsRef.current.length > 0) {
      console.log("音频播放结束，重置所有动画");
      resetAllAnimations();
    }
  }, [isAudioPlaying, resetAllAnimations]);

  useFrame((state, delta) => {
    if (mixer) {
      mixer.update(delta);
    }
  });

  return (
    <group name="avatar">
      <primitive object={gltf.scene} dispose={null} />
    </group>
  );
}

// WebSocket连接优化
function setupWebSocket(setBackendStatus, setWsReady) {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    console.log("WebSocket已连接，不需要重新连接");
    return websocket;
  }

  // 完整的WebSocket URL
  const fullWsUrl = `${wsUrl}/${clientId}`;
  console.log("尝试连接WebSocket:", fullWsUrl);

  try {
    const ws = new WebSocket(fullWsUrl);

    ws.onopen = () => {
      console.log("WebSocket连接成功");
      setBackendStatus("已连接 (WebSocket)");
      websocket = ws;
      if (setWsReady) setWsReady(true);

      // 发送ping保持连接
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        } else {
          clearInterval(pingInterval);
        }
      }, 30000);
    };

    ws.onerror = (error) => {
      console.error("WebSocket错误:", error);
      setBackendStatus("WebSocket连接失败 - 使用HTTP API");
      websocket = null;
      if (setWsReady) setWsReady(false);
    };

    ws.onclose = () => {
      console.log("WebSocket连接已关闭");
      setBackendStatus("使用HTTP API");
      websocket = null;
      if (setWsReady) setWsReady(false);

      // 尝试重新连接
      setTimeout(() => {
        setupWebSocket(setBackendStatus, setWsReady);
      }, 5000);
    };

    return ws;
  } catch (error) {
    console.error("创建WebSocket时出错:", error);
    setBackendStatus("WebSocket失败 - 使用HTTP API");
    if (setWsReady) setWsReady(false);
    return null;
  }
}

// 麦克风初始化
async function setupMicrophone(setMicStatus) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,
        channelCount: 1
      }
    });
    setMicStatus("麦克风已准备就绪");
    return stream;
  } catch (error) {
    console.error("获取麦克风权限失败:", error);
    setMicStatus(`麦克风错误: ${error.message}`);
    return null;
  }
}

// 录音配置
function startRecording(stream, setIsRecording, setRecordingStatus) {
  if (!stream) return;

  audioChunks = [];

  // 配置MediaRecorder
  const options = {
    mimeType: 'audio/webm',
    audioBitsPerSecond: 16000
  };

  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    console.warn("WebM格式不支持，尝试使用替代格式");
    try {
      mediaRecorder = new MediaRecorder(stream);
    } catch (e2) {
      console.error("无法创建MediaRecorder:", e2);
      setRecordingStatus("浏览器不支持录音功能");
      return;
    }
  }

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstart = () => {
    console.log("录音开始");
    setIsRecording(true);
    setRecordingStatus("正在录音...");
  };

  mediaRecorder.onstop = () => {
    console.log("录音结束");
    setIsRecording(false);
    setRecordingStatus("录音已停止");
  };

  // 每100ms保存一次数据，提高响应速度
  mediaRecorder.start(100);

  // 设置自动停止录音的计时器 (8秒)
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, 8000);
}

const STYLES = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
    position: 'relative'
  },
  speechArea: {
    position: 'absolute',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 500,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '90%',
    maxWidth: '600px'
  },
  conversationBox: {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    bottom: '180px',
    width: '90%',
    maxWidth: '600px',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: '10px',
    padding: '15px',
    zIndex: 499,
    display: 'flex',
    flexDirection: 'column',
    gap: '10px'
  },
  userBubble: {
    backgroundColor: 'rgba(0, 101, 255, 0.7)',
    padding: '10px 15px',
    borderRadius: '18px 18px 18px 0',
    alignSelf: 'flex-start',
    maxWidth: '80%',
    wordBreak: 'break-word',
    color: 'white',
    marginBottom: '5px'
  },
  aiBubble: {
    backgroundColor: 'rgba(70, 70, 70, 0.7)',
    padding: '10px 15px',
    borderRadius: '18px 18px 0 18px',
    alignSelf: 'flex-end',
    maxWidth: '80%',
    wordBreak: 'break-word',
    color: 'white',
    marginBottom: '5px'
  },
  recordButton: {
    width: '80px',
    height: '80px',
    borderRadius: '50%',
    backgroundColor: '#cc0000',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
    transition: 'all 0.2s ease',
    marginTop: '15px'
  },
  recordingButton: {
    backgroundColor: '#ff0000',
    boxShadow: '0 0 0 10px rgba(255, 0, 0, 0.3)'
  },
  micIcon: {
    fontSize: '30px',
    color: 'white'
  },
  statusText: {
    color: 'white',
    marginTop: '10px',
    textAlign: 'center',
    fontSize: '16px',
    textShadow: '1px 1px 3px rgba(0, 0, 0, 0.7)'
  },
  statusBar: {
    position: 'absolute',
    bottom: '5px',
    left: '10px',
    color: '#00FF00',
    fontSize: '12px',
    zIndex: 500
  },
  hidden: {
    display: 'none'
  }
};

function App() {
  // 状态变量
  const [speak, setSpeak] = useState(false);
  const [response, setResponse] = useState("");
  const [playing, setPlaying] = useState(false);
  const [backendStatus, setBackendStatus] = useState("正在连接...");
  const [animReady, setAnimReady] = useState(false);
  const [animationData, setAnimationData] = useState(null);
  const [micStream, setMicStream] = useState(null);
  const [micStatus, setMicStatus] = useState("正在初始化麦克风...");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("点击麦克风按钮开始录音");
  const [statusMessage, setStatusMessage] = useState("");
  const [conversation, setConversation] = useState([]);
  const [wsReady, setWsReady] = useState(false);
  
  // 流式音频处理状态
  const [audioQueue, setAudioQueue] = useState([]);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [currentAudioIndex, setCurrentAudioIndex] = useState(0);
  const currentAudioRef = useRef(null);
  const processingRef = useRef(false);
  const responseTextRef = useRef("");

  // 限制对话历史长度
  useEffect(() => {
    if (conversation.length > 10) {
      setConversation(prev => prev.slice(prev.length - 10));
    }
  }, [conversation]);

  // 停止录音
  const stopRecording = useCallback(async () => {
    if (!mediaRecorder) return;

    return new Promise((resolve) => {
      mediaRecorder.onstop = async () => {
        console.log("录音已完成，处理中...");
        setIsRecording(false);
        setRecordingStatus("处理录音...");
        setStatusMessage("正在处理语音...");

        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        console.log("录音文件大小:", audioBlob.size, "字节", "类型:", audioBlob.type);

        try {
          // 发送音频到服务器
          if (websocket && websocket.readyState === WebSocket.OPEN) {
            // 通过WebSocket发送
            sendAudioViaWebSocket(audioBlob);
          } else {
            // 通过HTTP请求发送
            await sendAudioViaHttp(audioBlob);
          }
        } catch (error) {
          console.error("处理音频时出错:", error);
          setStatusMessage(`处理音频失败: ${error.message}`);
        }

        resolve();
      };

      mediaRecorder.stop();
    });
  }, []);

  // 通过WebSocket发送音频
  const sendAudioViaWebSocket = useCallback((audioBlob) => {
    console.log("通过WebSocket发送音频");

    // 重置所有流式处理状态
    setAudioQueue([]);
    setCurrentAudioIndex(0);
    processingRef.current = false;
    responseTextRef.current = "";
    setAnimationData(null);
    
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }

    // 创建表单数据
    const reader = new FileReader();
    reader.onload = () => {
      const base64data = reader.result.split(',')[1];

      websocket.send(JSON.stringify({
        type: "speech_recognition",
        audio_data: base64data,
        audio_format: audioBlob.type || "audio/webm",
        client_id: clientId
      }));

      setStatusMessage("语音识别中...");
    };

    reader.readAsDataURL(audioBlob);
  }, []);

  // 通过HTTP发送音频
  const sendAudioViaHttp = useCallback(async (audioBlob) => {
    console.log("通过HTTP API发送音频");

    // 重置所有流式处理状态
    setAudioQueue([]);
    setCurrentAudioIndex(0);
    processingRef.current = false;
    responseTextRef.current = "";
    setAnimationData(null);
    
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }

    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('client_id', clientId);
    formData.append('audio_format', audioBlob.type || "audio/webm");

    try {
      setStatusMessage("发送音频到服务器...");
      const response = await axios.post(`${host}/recognize_speech`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log("语音识别响应:", response.data);
      if (response.data && response.data.text) {
        // 更新对话历史
        setConversation(prev => [...prev, {
          role: 'user',
          content: response.data.text
        }]);
        setStatusMessage("开始处理识别到的文本");
      } else {
        setStatusMessage("语音识别失败");
      }
    } catch (error) {
      console.error("发送音频失败:", error);
      setStatusMessage(`发送音频失败: ${error.message}`);
    }
  }, []);

  // 取消进行中的任务
  const cancelProcessingTask = useCallback(() => {
    // 停止当前播放的音频
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    
    // 重置流式处理状态
    setAudioQueue([]);
    setCurrentAudioIndex(0);
    setIsAudioPlaying(false);
    processingRef.current = false;
    responseTextRef.current = "";
    
    // 重置状态
    setPlaying(false);
    setAnimationData(null);
    setStatusMessage("");
    
    // 重置动画状态
    resetAnimationState();
  }, []);

  // 处理流式音频队列
  useEffect(() => {
    // 当有新的音频添加到队列时，检查是否需要开始播放
    if (audioQueue.length > 0 && !isAudioPlaying && currentAudioIndex < audioQueue.length) {
      console.log(`开始播放音频片段 ${currentAudioIndex+1}/${audioQueue.length}`);
      
      const currentSegment = audioQueue[currentAudioIndex];
      const audio = new Audio(currentSegment.audioUrl);
      currentAudioRef.current = audio;
      
      // 设置播放事件
      audio.oncanplay = () => {
        console.log(`音频片段 ${currentAudioIndex+1} 已加载，准备播放`);
        
        // 关键修改：先设置动画数据，再播放音频
        if (currentSegment.blendData && currentSegment.blendData.length > 0) {
          console.log(`设置动画数据，帧数: ${currentSegment.blendData.length}`);
          setAnimationData({ blendData: currentSegment.blendData });
          setPlaying(true);
          
          // 稍微延迟播放音频，确保动画已设置
          setTimeout(() => {
            audio.play()
              .then(() => {
                console.log(`开始播放音频片段 ${currentAudioIndex+1}`);
                setIsAudioPlaying(true);
              })
              .catch(error => {
                console.error(`播放音频片段 ${currentAudioIndex+1} 失败:`, error);
                playNextAudio();
              });
          }, 50);
        } else {
          console.warn(`音频片段 ${currentAudioIndex+1} 没有动画数据`);
          // 即使没有动画数据也播放音频
          audio.play()
            .then(() => {
              console.log(`开始播放音频片段 ${currentAudioIndex+1} (无动画)`);
              setIsAudioPlaying(true);
            })
            .catch(error => {
              console.error(`播放音频片段 ${currentAudioIndex+1} 失败:`, error);
              playNextAudio();
            });
        }
      };
      
      // 当前音频播放结束时
      audio.onended = () => {
        console.log(`音频片段 ${currentAudioIndex+1} 播放结束`);
        playNextAudio();
      };
      
      // 音频加载错误
      audio.onerror = () => {
        console.error(`音频片段 ${currentAudioIndex+1} 加载出错`);
        playNextAudio();
      };
    }
  }, [audioQueue, isAudioPlaying, currentAudioIndex]);
  
  // 播放下一个音频片段
  const playNextAudio = useCallback(() => {
    // 清理当前音频
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    
    // 更新索引
    setCurrentAudioIndex(prev => prev + 1);
    setIsAudioPlaying(false);

    // 检查是否所有片段都已播放完毕
    if (currentAudioIndex + 1 >= audioQueue.length) {
      console.log("所有音频片段播放完毕");
     
      // 所有片段播放完毕
      if (!processingRef.current) {
        // 没有更多待处理的片段，完全结束
        setPlaying(false);
        setStatusMessage("");
      } else {
        // 仍在处理中，等待下一个片段
        setStatusMessage("继续处理更多内容...");
      }
    }
  }, [currentAudioIndex, audioQueue.length]);

 // 初始化WebSocket
 useEffect(() => {
   const ws = setupWebSocket(setBackendStatus, setWsReady);
 }, []);

 // 初始化麦克风
 useEffect(() => {
   async function initMic() {
     const stream = await setupMicrophone(setMicStatus);
     setMicStream(stream);
   }

   initMic();

   // 组件卸载时关闭麦克风
   return () => {
     if (micStream) {
       micStream.getTracks().forEach(track => track.stop());
     }
   };
 }, []);

 // 设置WebSocket消息处理 - 关键修改点
 useEffect(() => {
   if (websocket) {
     websocket.onmessage = (event) => {
       try {
         const message = JSON.parse(event.data);
         
         console.log("收到WebSocket消息类型:", message.type);
         
         // 处理流式音频片段
         if (message.type === "stream_audio_segment") {
           console.log("收到流式音频片段", {
             segment_index: message.segment_index,
             is_first_segment: message.is_first_segment,
             blendData_length: message.blendData ? message.blendData.length : 0,
             text: message.text
           });
           
           // 确保音频URL是完整路径
           let audioPath = message.filename;
           if (!audioPath.startsWith('http')) {
             audioPath = `${host}${audioPath}`;
           }
           
           // 将音频片段添加到队列
           setAudioQueue(prev => [...prev, {
             audioUrl: audioPath,
             blendData: message.blendData,
             text: message.text,
             segmentIndex: message.segment_index
           }]);
           
           // 更新处理状态
           processingRef.current = true;
           
           // 更新文本显示
           if (message.text) {
             responseTextRef.current += message.text;
             setResponse(responseTextRef.current);
             
             // 如果这是第一个片段，添加到对话历史
             if (message.is_first_segment) {
               setConversation(prev => [...prev, {
                 role: 'assistant',
                 content: responseTextRef.current
               }]);
             } else {
               // 更新最后一个对话
               setConversation(prev => {
                 const newConversation = [...prev];
                 if (newConversation.length > 0) {
                   newConversation[newConversation.length - 1] = {
                     ...newConversation[newConversation.length - 1],
                     content: responseTextRef.current
                   };
                 }
                 return newConversation;
               });
             }
           }
           
           // 更新状态消息
           setStatusMessage(`处理片段 ${message.segment_index + 1}...`);
         }
         
         // 处理完成所有片段
         else if (message.type === "processing_complete_all") {
           console.log("所有片段处理完成");
           processingRef.current = false;
           
           // 更新最终响应文本
           if (message.message) {
             responseTextRef.current = message.message;
             setResponse(message.message);
             
             // 更新对话历史中的最后一条消息
             setConversation(prev => {
               const newConversation = [...prev];
               if (newConversation.length > 0) {
                 newConversation[newConversation.length - 1] = {
                   ...newConversation[newConversation.length - 1],
                   content: message.message
                 };
               }
               return newConversation;
             });
           }
           
           setStatusMessage("");
         }
         
         // 保留对旧消息类型的处理以兼容性
         else if (message.type === "processing_complete") {
           console.log("处理完成，获取Blendshape数据和音频");

           // 确保音频URL是完整路径
           let audioPath = message.filename;
           if (!audioPath.startsWith('http')) {
             audioPath = `${host}${audioPath}`;
           }
           console.log("完整音频URL:", audioPath);

           // 保存动画数据
           if (message.blendData && message.blendData.length > 0) {
             console.log("设置动画数据，帧数:", message.blendData.length);
             setAnimationData({ blendData: message.blendData });
           }

           // 创建音频元素
           const audio = new Audio(audioPath);
           audio.oncanplay = () => {
             console.log("音频已加载，准备播放");
             audio.play()
               .then(() => {
                 console.log("开始播放音频");
                 setPlaying(true);
                 setIsAudioPlaying(true);
               })
               .catch(error => {
                 console.error("播放音频失败:", error);
               });
           };

           audio.onended = () => {
             console.log("音频播放结束");
             setPlaying(false);
             setIsAudioPlaying(false);
             setAnimationData(null);
           };

           currentAudioRef.current = audio;

           if (message.text) {
             setResponse(message.text);
             // 添加到对话历史
             setConversation(prev => [...prev, {
               role: 'assistant',
               content: message.text
             }]);
           }

           // 重置状态消息
           setStatusMessage("");
         }
         else if (message.type === "ai_response") {
           console.log("收到AI响应:", message.text);
           if (message.text) {
             setStatusMessage("AI已响应，正在生成语音...");
           }
         }
         else if (message.type === "speech_recognition_result") {
           console.log("收到语音识别结果:", message.text);
           if (message.text) {
             // 添加到对话历史
             setConversation(prev => [...prev, {
               role: 'user',
               content: message.text
             }]);
             setStatusMessage("识别完成，等待回复...");
             
             // 重置流式处理状态
             responseTextRef.current = "";
             setAudioQueue([]);
             setCurrentAudioIndex(0);
             processingRef.current = true;
           } else {
             setStatusMessage("语音识别结果为空");
           }
         }
         else if (message.type === "processing_status") {
           // 处理状态更新
           setStatusMessage(message.message || `正在${message.status}...`);
         }
         else if (message.type === "error") {
           console.error("WebSocket错误:", message.message);
           alert(`处理时出错: ${message.message}`);
           setStatusMessage("");
           
           // 重置处理状态
           processingRef.current = false;
         }
       } catch (error) {
         console.error("解析WebSocket消息时出错:", error);
         setStatusMessage("");
       }
     };
   }
 }, [websocket]);

 // 检查后端服务状态
 useEffect(() => {
   fetch(`${host}/docs`)
     .then(response => {
       if(response.ok) {
         setBackendStatus("REST API已连接");
         // 如果REST API可用，尝试连接WebSocket
         setupWebSocket(setBackendStatus, setWsReady);
       } else {
         setBackendStatus("后端服务连接失败");
       }
     })
     .catch(err => {
       console.error("后端服务连接错误:", err);
       setBackendStatus(`后端服务不可用: ${err.message}`);
     });
 }, []);

 // 处理录音按钮点击
 const handleRecordClick = useCallback(async () => {
   // 如果当前正在播放或处理中，则取消当前任务
   if (isAudioPlaying || processingRef.current) {
     cancelProcessingTask();
     return;
   }
   
   if (isRecording) {
     // 停止录音
     await stopRecording();
   } else {
     // 开始录音
     if (micStream) {
       startRecording(micStream, setIsRecording, setRecordingStatus);
     } else {
       // 尝试重新获取麦克风权限
       const stream = await setupMicrophone(setMicStatus);
       if (stream) {
         setMicStream(stream);
         startRecording(stream, setIsRecording, setRecordingStatus);
       } else {
         alert("无法访问麦克风。请检查权限设置。");
       }
     }
   }
 }, [isRecording, micStream, stopRecording, isAudioPlaying, cancelProcessingTask]);

 // 获取最后一条消息用于显示
 const lastMessage = conversation.length > 0 ? conversation[conversation.length - 1] : null;

 // 计算麦克风按钮文本
 const getMicButtonText = useCallback(() => {
   if (isRecording) return '■';
   if (isAudioPlaying || processingRef.current) return '✕';
   return '🎤';
 }, [isRecording, isAudioPlaying]);

 return (
   <div style={STYLES.container}>
     {/* 对话历史 (显示最后一条) */}
     {lastMessage && (
       <div style={STYLES.conversationBox}>
         <div
           style={lastMessage.role === 'user' ? STYLES.userBubble : STYLES.aiBubble}
         >
           {lastMessage.content}
         </div>
       </div>
     )}

     {/* 语音输入区域 */}
     <div style={STYLES.speechArea}>
       <button
         onClick={handleRecordClick}
         style={{
           ...STYLES.recordButton,
           ...(isRecording ? STYLES.recordingButton : {})
         }}
       >
         <span style={STYLES.micIcon}>
           {getMicButtonText()}
         </span>
       </button>

       <div style={STYLES.statusText}>
         {isRecording ? '正在录音...' : (statusMessage || '点击麦克风开始语音输入')}
       </div>
     </div>

     {/* 状态栏 */}
     <div style={STYLES.statusBar}>
       {backendStatus} | {wsReady ? 'WebSocket已连接' : 'HTTP模式'} | 动画: {animReady ? '已加载' : '无'} | {micStatus}
     </div>

     <Canvas dpr={2} onCreated={(ctx) => {
         ctx.gl.physicallyCorrectLights = true;
       }}>

       <OrthographicCamera
         makeDefault
         zoom={2000}
         position={[0, 1.65, 1]}
       />

       <Suspense fallback={null}>
         <Environment background={false} files="/images/photo_studio_loft_hall_1k.hdr" />
       </Suspense>

       <Suspense fallback={null}>
         <Bg />
       </Suspense>

       <Suspense fallback={null}>
         <Avatar
           avatar_url="/model.glb"
           speak={speak}
           setSpeak={setSpeak}
           text={response}
           playing={playing}
           setPlaying={setPlaying}
           setResponse={setResponse}
           setAnimReady={setAnimReady}
           animationData={animationData}
           isAudioPlaying={isAudioPlaying}
           currentSegmentIndex={currentAudioIndex}
         />
       </Suspense>

     </Canvas>
     <Loader dataInterpolation={(p) => `加载中... ${Math.round(p * 100)}%`} />
   </div>
 )
}

function Bg() {
 const texture = useTexture('/images/bg.webp');

 return(
   <mesh position={[0, 1.5, -2]} scale={[0.8, 0.8, 0.8]}>
     <planeBufferGeometry />
     <meshBasicMaterial map={texture} />
   </mesh>
 )
}

export default App;

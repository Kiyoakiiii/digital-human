import React, { Suspense, useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, useTexture, Loader, Environment, useFBX, useAnimations, OrthographicCamera } from '@react-three/drei';
import { MeshStandardMaterial } from 'three/src/materials/MeshStandardMaterial';

import { LinearEncoding, sRGBEncoding } from 'three/src/constants';
import { LineBasicMaterial, MeshPhysicalMaterial, Vector2 } from 'three';

import createAnimation from './converter';
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

function Avatar({ avatar_url, speak, setSpeak, text, playing, setPlaying, setResponse, setAnimReady, animationData, setAudioElement }) {
  let gltf = useGLTF(avatar_url);
  let morphTargetDictionaryBody = null;
  let morphTargetDictionaryLowerTeeth = null;
  const mixerRef = useRef(null);

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

      if (node.name.includes("TSHIRT")) {
        node.material = new MeshStandardMaterial();
        node.material.map = tshirtDiffuseTexture;
        node.material.roughnessMap = tshirtRoughnessTexture;
        node.material.normalMap = tshirtNormalTexture;
        node.material.color.setHex(0xffffff);
        node.material.envMapIntensity = 0.5;
      }

      if (node.name.includes("TeethLower")) {
        morphTargetDictionaryLowerTeeth = node.morphTargetDictionary;
      }
    }
  });

  const [clips, setClips] = useState([]);
  const mixer = useMemo(() => {
    const newMixer = new THREE.AnimationMixer(gltf.scene);
    mixerRef.current = newMixer;
    return newMixer;
  }, []);

  // 当clips变化时，通知App组件动画状态
  useEffect(() => {
    if (setAnimReady) {
      setAnimReady(clips.length > 0);
    }
  }, [clips, setAnimReady]);

  // 更新处理动画和音频数据的方法 - 优化性能
  useEffect(() => {
    if (animationData && animationData.blendData) {
      console.log("处理动画数据，帧数:", animationData.blendData.length);

      // 优化：限制处理的帧数，增加性能
      const maxFrames = 300; // 限制最大帧数以提高性能
      let processedData = animationData.blendData;
      
      if (animationData.blendData.length > maxFrames) {
        const skipFactor = Math.ceil(animationData.blendData.length / maxFrames);
        processedData = animationData.blendData.filter((_, index) => index % skipFactor === 0);
        console.log(`优化：从 ${animationData.blendData.length} 帧减少到 ${processedData.length} 帧`);
      }

      // 创建动画剪辑
      const newClips = [
        createAnimation(processedData, morphTargetDictionaryBody, 'HG_Body'),
        createAnimation(processedData, morphTargetDictionaryLowerTeeth, 'HG_TeethLower')
      ];

      console.log("动画剪辑已创建:", newClips.map(c => c.tracks.length + "个轨道"));

      // 设置动画剪辑
      setClips(newClips);
    }
  }, [animationData]);

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

  useEffect(() => {
    let idleClipAction = mixer.clipAction(idleClips[0]);
    idleClipAction.play();

    let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body');
    let blinkAction = mixer.clipAction(blinkClip);
    blinkAction.play();
  }, []);

  // Play animation clips when available - 优化性能
  useEffect(() => {
    if (playing === false || !clips || clips.length === 0)
      return;

    console.log("开始播放动画剪辑，数量:", clips.length);

    // 停止所有正在播放的动作（除了眨眼和idle）
    mixer.stopAllAction();

    // 重新播放基础动画
    let idleClipAction = mixer.clipAction(idleClips[0]);
    idleClipAction.play();

    let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body');
    let blinkAction = mixer.clipAction(blinkClip);
    blinkAction.play();

    // 播放新的表情动画
    _.each(clips, clip => {
      if (clip && clip.tracks && clip.tracks.length > 0) {
        let clipAction = mixer.clipAction(clip);
        clipAction.setLoop(THREE.LoopOnce);
        clipAction.clampWhenFinished = true; // 动画结束时保持最后一帧
        clipAction.play();
        console.log("播放动画剪辑，轨道数:", clip.tracks.length);
      } else {
        console.warn("跳过无效的动画剪辑");
      }
    });
  }, [playing, clips]);

  // 暂停播放时重置动画
  useEffect(() => {
    if (playing === false && clips && clips.length > 0 && mixerRef.current) {
      // 暂停所有当前播放的表情动画
      _.each(clips, clip => {
        if (clip && clip.tracks && clip.tracks.length > 0) {
          const action = mixerRef.current.existingAction(clip);
          if (action) {
            action.stop();
          }
        }
      });

      // 确保基础动画继续播放
      let idleClipAction = mixerRef.current.clipAction(idleClips[0]);
      idleClipAction.play();

      let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body');
      let blinkAction = mixerRef.current.clipAction(blinkClip);
      blinkAction.play();
    }
  }, [playing]);

  useFrame((state, delta) => {
    mixer.update(delta);
  });

  return (
    <group name="avatar">
      <primitive object={gltf.scene} dispose={null} />
    </group>
  );
}

// 优化: 使用缓存和重试机制的WebSocket连接
function setupWebSocket(setBackendStatus, setWsReady) {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    console.log("WebSocket已连接，不需要重新连接");
    return websocket; // WebSocket已经连接
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

// 优化: 更高效的麦克风初始化
async function setupMicrophone(setMicStatus) {
  try {
    // 优化: 更高效的音频配置
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,    // 设置采样率为16kHz
        channelCount: 1       // 单声道
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

// 优化: 更高效的录音配置
function startRecording(stream, setIsRecording, setRecordingStatus) {
  if (!stream) return;

  audioChunks = [];

  // 配置MediaRecorder，强制使用合适的编码
  const options = {
    mimeType: 'audio/webm',  // 更改为webm格式
    audioBitsPerSecond: 16000 // 16kHz采样率
  };

  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    console.warn("WebM格式不支持，尝试使用替代格式");
    try {
      // 尝试备用格式
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

  // 设置自动停止录音的计时器 (最长录音时间，例如8秒)
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, 8000);  // 减少最大录音时间以提高响应速度
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
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    pointerEvents: 'none'
  },
  loadingSpinner: {
    width: '50px',
    height: '50px',
    border: '5px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '50%',
    borderTop: '5px solid white',
    animation: 'spin 1s linear infinite'
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
  const [loadingStatus, setLoadingStatus] = useState("");
  const [conversation, setConversation] = useState([]);
  const [audioElement, setAudioElement] = useState(null);
  const [wsReady, setWsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // 保存处理中的任务
  const processingTaskRef = useRef(null);

  // 优化: 减少对话历史的最大长度，以提高性能
  useEffect(() => {
    if (conversation.length > 10) {
      // 保留最新的10条消息
      setConversation(prev => prev.slice(prev.length - 10));
    }
  }, [conversation]);

  // 优化: 使用useCallback减少不必要的重渲染
  const stopRecording = useCallback(async () => {
    if (!mediaRecorder) return;

    return new Promise((resolve) => {
      mediaRecorder.onstop = async () => {
        console.log("录音已完成，处理中...");
        setIsRecording(false);
        setRecordingStatus("处理录音...");
        setLoadingStatus("正在处理语音...");
        setIsLoading(true);

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
          setLoadingStatus(`处理音频失败: ${error.message}`);
          setIsLoading(false);
        }

        resolve();
      };

      mediaRecorder.stop();
    });
  }, []);

  // 通过WebSocket发送音频
  const sendAudioViaWebSocket = useCallback((audioBlob) => {
    console.log("通过WebSocket发送音频");

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

      setLoadingStatus("语音识别中...");
    };

    reader.readAsDataURL(audioBlob);
  }, []);

  // 通过HTTP发送音频
  const sendAudioViaHttp = useCallback(async (audioBlob) => {
    console.log("通过HTTP API发送音频");

    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('client_id', clientId);
    formData.append('audio_format', audioBlob.type || "audio/webm");

    try {
      setLoadingStatus("发送音频到服务器...");
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
        setLoadingStatus("开始处理识别到的文本");
      } else {
        setLoadingStatus("语音识别失败");
        setIsLoading(false);
      }
    } catch (error) {
      console.error("发送音频失败:", error);
      setLoadingStatus(`发送音频失败: ${error.message}`);
      setIsLoading(false);
    }
  }, []);

  // 优化: 取消进行中的任务
  const cancelProcessingTask = useCallback(() => {
    if (processingTaskRef.current) {
      clearTimeout(processingTaskRef.current);
      processingTaskRef.current = null;
    }
    
    // 停止当前播放的音频
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
    
    // 重置状态
    setPlaying(false);
    setAnimationData(null);
    setIsLoading(false);
    setLoadingStatus("");
  }, [audioElement]);

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

  // 设置WebSocket消息处理
  useEffect(() => {
    if (websocket) {
      websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log("收到WebSocket消息:", message);

          if (message.type === "processing_complete") {
            console.log("处理完成，获取Blendshape数据和音频");
            setIsLoading(false);

            // 确保音频URL是完整路径
            let audioPath = message.filename;
            if (!audioPath.startsWith('http')) {
              audioPath = `${host}${audioPath}`;
            }
            console.log("完整音频URL:", audioPath);

            // 保存动画数据
            setAnimationData({ blendData: message.blendData });

            // 创建音频元素
            const audio = new Audio(audioPath);
            audio.oncanplay = () => {
              console.log("音频已加载，准备播放");
              audio.play()
                .then(() => {
                  console.log("开始播放音频");
                  setPlaying(true);
                })
                .catch(error => {
                  console.error("播放音频失败:", error);
                });
            };

            audio.onended = () => {
              console.log("音频播放结束");
              setPlaying(false);
              setAnimationData(null);
            };

            setAudioElement(audio);

            if (message.text) {
              setResponse(message.text); // You might still need this for the Avatar if it uses it directly
              // 添加到对话历史
              setConversation(prev => [...prev, {
                role: 'assistant',
                content: message.text
              }]);
            }

            // 重置加载状态
            setLoadingStatus("");
          }
          else if (message.type === "ai_response") {
            console.log("收到AI响应:", message.text);
            if (message.text) {
              setResponse(message.text); // Update response for Avatar
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
              setLoadingStatus("识别完成，等待回复...");
            } else {
              setLoadingStatus("语音识别结果为空");
              setIsLoading(false);
            }
          }
          else if (message.type === "processing_status") {
            // 新增: 处理状态更新
            setLoadingStatus(message.message || `正在${message.status}...`);
          }
          else if (message.type === "error") {
            console.error("WebSocket错误:", message.message);
            alert(`处理时出错: ${message.message}`);
            setLoadingStatus("");
            setIsLoading(false);
          }
        } catch (error) {
          console.error("解析WebSocket消息时出错:", error);
          setLoadingStatus("");
          setIsLoading(false);
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
    // 如果当前正在加载或播放，则取消当前任务
    if (isLoading || playing) {
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
  }, [isRecording, micStream, stopRecording, isLoading, playing, cancelProcessingTask]);

  // Get the last message for display
  const lastMessage = conversation.length > 0 ? conversation[conversation.length - 1] : null;

  // 计算麦克风按钮文本
  const getMicButtonText = useCallback(() => {
    if (isRecording) return '■';
    if (isLoading || playing) return '✕';
    return '🎤';
  }, [isRecording, isLoading, playing]);

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
          {isRecording ? '正在录音...' : (loadingStatus || '点击麦克风开始语音输入')}
        </div>
      </div>

      {/* 加载覆盖层 */}
      {isLoading && (
        <div style={STYLES.loadingOverlay}>
          <div style={STYLES.loadingSpinner}></div>
        </div>
      )}

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
            setAudioElement={setAudioElement}
          />
        </Suspense>

      </Canvas>
      <Loader dataInterpolation={(p) => `加载中... ${Math.round(p * 100)}%`} />
      
      {/* CSS Animations */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
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

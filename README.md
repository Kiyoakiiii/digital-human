# Real-Time Streaming Conversational Digital Human

A full-stack system enabling real-time conversational AI with synchronized 3D facial animation. The digital human listens to speech, processes it through a streaming pipeline, and responds with natural voice and accurate lip-sync animation.

The entire pipeline is optimized for minimal latency through parallel processing—from user speech input to animated avatar response with synchronized audio and blendshapes.

## Core Features

**End-to-End Streaming Architecture**  
Built on a fully asynchronous streaming pipeline that processes user speech in real-time. Audio is transcribed on-the-fly via NVIDIA Riva, tokens are streamed from the LLM, and responses are immediately segmented for parallel TTS and facial animation generation—eliminating buffering delays.

**Intelligent Text Segmentation**  
Implements smart sentence segmentation logic that splits LLM token streams at optimal punctuation points, enabling immediate processing of each sentence fragment without waiting for complete responses. First segments use strict punctuation rules for faster initial response, while subsequent segments use relaxed rules for natural flow.

**High-Fidelity Speech Synthesis**  
Utilizes GPT-SoVITS with reference audio to generate emotionally expressive speech with consistent voice characteristics. Includes automatic timeout handling and fallback mechanisms for robustness.

**Accurate Facial Animation**  
Generates complete 52-blendshape ARKit-compatible facial animations using NVIDIA Audio2Face gRPC service, with intelligent fallback to a custom procedural blendshape generator when needed. Supports both short-form direct processing and long-form segmented processing for audio over 10 seconds.

**Asynchronous Backend Architecture**  
FastAPI backend with async I/O handles concurrent operations across multiple AI services (ASR, LLM, TTS, A2F) using threading and asyncio. Includes WebSocket support for bidirectional real-time communication with automatic connection recovery.

**Interactive 3D Frontend**  
React and Three.js (React Three Fiber) render a fully rigged 3D character model with real-time morph target animation driven by streaming blendshape data. Implements audio-animation synchronization with frame-perfect timing at 60 FPS.

## System Architecture

The pipeline operates through highly parallelized streaming:

1. **Audio Capture**: Frontend captures microphone input via MediaRecorder API (WebM format, 16kbps) and streams to backend over WebSocket or HTTP multipart upload.

2. **Audio Transcoding**: Backend receives audio and uses FFmpeg subprocess to transcode to 16kHz, 16-bit mono WAV format required by Riva ASR.

3. **Speech Recognition**: Transcoded audio is sent to NVIDIA Riva ASR service with Chinese language model (`zh-CN`) configured for automatic punctuation and single-channel recognition.

4. **LLM Streaming**: Recognized text triggers streaming completion request to DeepSeek LLM API. Token stream is consumed in real-time with immediate processing.

5. **Intelligent Segmentation**: Backend implements custom segmentation logic inspired by professional TTS systems—splits token stream at punctuation boundaries with different rules for first vs. subsequent segments to optimize perceived latency.

6. **Parallel TTS & A2F**:
   - **Thread 1**: Monitors text segment queue and processes each segment immediately
   - **Thread 2**: GPT-SoVITS generates WAV audio for each text segment
   - **Thread 3**: Audio2Face gRPC processes each WAV file to generate 52-blendshape CSV data
   - All threads operate concurrently with automatic audio length detection and segmentation for long audio files

7. **Blendshape Conversion**: CSV data from A2F is parsed and converted to JSON format with frame-by-frame blendshape weights mapped to ARKit naming convention.

8. **Streaming to Client**: Each completed audio-animation pair is immediately sent to frontend via WebSocket as a discrete segment, enabling playback to begin before full response generation completes.

9. **Frontend Rendering**: React Three Fiber loads audio segments into a queue and plays them sequentially. Three.js AnimationMixer drives morph target animations synchronized with audio playback using AnimationClip with LoopOnce mode.

## Technical Implementation Details

### Backend Architecture

**Asynchronous Processing**
- FastAPI handles WebSocket connections and HTTP endpoints concurrently
- Threading for CPU-bound tasks (TTS, A2F processing)
- Asyncio for I/O-bound operations (file operations, network requests)
- Custom connection manager for WebSocket session tracking

**Audio Processing Pipeline**
- FFmpeg subprocess for real-time audio format conversion
- Dynamic audio segmentation for files exceeding buffer limits (default 10s)
- Automatic cleanup of temporary files with timestamp-based naming

**Blendshape Generation**
- Primary: NVIDIA Audio2Face gRPC service with optimized parameters
- Fallback: Custom procedural generator using STFT frequency analysis
  - Multi-band energy extraction (low/mid/high frequency)
  - Kalman filtering for smooth motion estimation
  - Gaussian-weighted eye movement events
  - Natural blink patterns with randomized intervals

### Frontend Architecture

**3D Rendering**
- React Three Fiber for declarative Three.js scene management
- GLTF model loading with custom material assignments
- PBR materials with separate textures for body, eyes, teeth, hair
- Morph target animation via AnimationMixer with frame-delta updates

**Audio Queue Management**
- State-driven audio segment queue with sequential playback
- Audio element lifecycle management with cleanup on completion
- Animation-audio synchronization through shared state updates
- Automatic transition to next segment on audio `onended` event

**Real-Time Communication**
- WebSocket primary connection with automatic reconnection logic
- HTTP fallback for audio upload when WebSocket unavailable
- Base64 encoding for audio data transmission over WebSocket
- JSON message protocol with typed message handlers

## Deployment

### Prerequisites

**NVIDIA Services** (Critical - must be deployed first):
- **NVIDIA Riva Container**: ASR service with Chinese language model support
- **Audio2Face Container**: gRPC service enabled on port 52000 or 50051
- Verify network accessibility from your application server to both services

**System Requirements**:
- Python 3.8+
- Node.js 14+
- FFmpeg installed and available in PATH
- CUDA-compatible GPU (recommended for GPT-SoVITS)
- Minimum 8GB RAM, 16GB recommended

### Setup

**1. Clone Repository**
```bash
git clone https://github.com/Kiyoakiiii/digital-human.git
cd digital-human
```

**2. Configure Services**

Update hardcoded configurations to match your environment:

**Backend (`backend/app.py`):**
```python
# Update these paths to your actual directories
GPT_SOVITS_PATH = "/path/to/your/GPT-SoVITS"
AUDIO_DIR = "/path/to/audio"
BLENDSHAPE_DIR = "/path/to/blendshape"
TEMP_DIR = "/path/to/temp"

# Update service addresses
audio2face_client = Audio2Face3DClient(
    server_address="YOUR_A2F_IP:52000",  # Your Audio2Face server
    max_buffer_seconds=10.0
)

auth = riva.client.Auth(uri='YOUR_RIVA_IP:50051')  # Your Riva server
```

**Backend (`backend/chat_digital_human_lib.py`):**
```python
# Update LLM API configuration
address = "YOUR_LLM_API_ADDRESS:PORT"
api_key = "YOUR_API_KEY"
chat_id = "YOUR_CHAT_ID"

# Update GPT-SoVITS path
GPT_SOVITS_PATH = "/path/to/your/GPT-SoVITS/GPT_SoVITS"
```

**Frontend (`frontend/src/App.js`):**
```javascript
// Update backend server address
const SERVER_IP = "YOUR_BACKEND_SERVER_IP";
```

**Scripts (`start_all.sh` & `stop_all.sh`):**
```bash
# Update project directory path
PROJECT_DIR="/path/to/your/digital-human"
```

> **Production Note**: Migrate these configurations to environment variables or a centralized config file for easier deployment management.

**3. Install Dependencies**

Backend:
```bash
cd backend
pip install -r requirements.txt
```

Frontend:
```bash
cd frontend
npm install
```

**4. Prepare Reference Audio**

Place your reference voice file at:
```bash
mkdir -p /path/to/audio
# Copy your reference audio file
cp your_reference_voice.wav /path/to/audio/reference_voice.wav
```

**5. Start Services**

```bash
# Make scripts executable
chmod +x start_all.sh stop_all.sh

# Start backend and frontend
./start_all.sh

# Monitor logs
tail -f logs/backend.log
tail -f logs/frontend.log

# Stop services
./stop_all.sh
```

**6. Access Application**

Open your browser and navigate to:
```
http://<your-frontend-server-ip>:3000
```

Click the microphone button to start voice interaction.


## Technical Stack

- **ASR**: NVIDIA Riva (gRPC, Chinese ASR model)
- **LLM**: DeepSeek API (streaming completions)
- **TTS**: GPT-SoVITS (reference-based voice cloning)
- **Facial Animation**: NVIDIA Audio2Face (gRPC service, 52 ARKit blendshapes)
- **Backend**: FastAPI, Python asyncio, threading, FFmpeg
- **Frontend**: React, Three.js (React Three Fiber), WebSocket, MediaRecorder API
- **3D Rendering**: Three.js AnimationMixer, GLTF model with morph targets, PBR materials

## Performance Characteristics

- **End-to-end latency**: < 1s from speech end to first animation frame
- **First segment response**: Optimized segmentation delivers initial speech within 2-3 seconds
- **Audio transcoding**: Real-time FFmpeg processing with < 100ms overhead
- **Parallel processing**: TTS and A2F generation run concurrently per sentence
- **Streaming efficiency**: Segments delivered to client immediately upon completion
- **Animation sync**: 60 FPS rendering with frame-delta updates for smooth morph targets
- **Blendshape fallback**: Procedural generator activates within 5s timeout for A2F failures
- **Connection resilience**: Automatic WebSocket reconnection with 5s intervals






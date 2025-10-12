# Real-Time Streaming Conversational Digital Human

A full-stack system for real-time conversational AI with synchronized 3D facial animation. The digital human listens, thinks, and responds with continuous, fluid speech while maintaining lip-sync accuracy.

The entire pipeline is optimized for minimal latency—from the moment a user starts speaking to when the virtual character produces audible speech with synchronized facial animation.

## Core Features

**End-to-End Streaming Architecture**  
The system processes user speech in real-time through a fully streaming pipeline. Audio is transcribed on-the-fly, text tokens are streamed to the language model, and responses are immediately segmented for parallel speech synthesis and animation generation—eliminating buffering delays.

**Intelligent Conversation**  
Powered by DeepSeek LLM for context-aware dialogue, complex reasoning, and natural conversation flow.

**Expressive Speech Synthesis**  
Utilizes GPT-SoVITS to generate high-fidelity, emotionally nuanced speech with consistent voice characteristics.

**Realistic Facial Animation**  
NVIDIA Audio2Face drives detailed facial expressions using 52 ARKit blendshapes, achieving frame-perfect lip-sync with generated audio.

**High-Performance Backend**  
Built on FastAPI with asynchronous I/O, leveraging multi-threading and multi-processing to handle concurrent ASR, LLM, TTS, and A2F operations efficiently.

**Interactive 3D Frontend**  
React and Three.js (R3F) render the 3D character model with real-time bidirectional communication via WebSocket.

## System Architecture

The pipeline operates as a highly parallelized streaming process:

1. **Audio Input**: Frontend captures microphone audio via `MediaRecorder` API and streams it to the backend over WebSocket.

2. **Speech-to-Text (ASR)**: Backend receives audio (e.g., WebM format) and transcodes it to 16kHz, 16-bit mono WAV using **FFmpeg**, then streams to **NVIDIA Riva** for real-time speech recognition.

3. **Language Model Inference (LLM)**: Recognized text is immediately sent to **DeepSeek** LLM, which begins generating responses as a token stream.

4. **Intelligent Text Segmentation**: Backend segments the LLM's token stream into complete sentences (using punctuation as delimiters). Each sentence is dispatched to the next stage **immediately**—no waiting for the full response.

5. **Parallel Speech & Animation Generation**:
   - **TTS**: Each sentence is fed to **GPT-SoVITS** to generate corresponding audio (WAV).
   - **Audio2Face**: Generated audio is sent to **NVIDIA Audio2Face** via gRPC, which returns blendshape weights as a CSV file.

6. **Streaming to Client**: Backend packages the **audio URL** and **blendshape JSON data** and sends them to the frontend via WebSocket.

7. **Real-Time Rendering**: Frontend plays audio chunks immediately while `AnimationMixer` loads and plays corresponding facial animation data, ensuring perfect synchronization.

This streaming architecture ensures the virtual character responds and begins speaking with minimal perceptible delay, creating a truly real-time conversational experience.

## Deployment

### Prerequisites

**NVIDIA Services (Critical)**  
Deploy these first before proceeding:

- **NVIDIA Riva Container**: Ensure Riva is running with ASR models loaded (Chinese model required for this demo).
- **Audio2Face Container**: Launch Audio2Face with gRPC service enabled and listening on the specified port.

Verify that your server can access the exposed ports (e.g., Riva on 50051, A2F on 52000 or 50051).

### Setup

**1. Clone Repository**
```bash
git clone https://github.com/Kiyoakiiii/digital-human.git
cd digital-human
```

**2. Configure Services**

You must update hardcoded configurations to match your environment:

**Backend (`backend/app.py`):**
- Update `Audio2Face3DClient(server_address=...)` with your Audio2Face service IP and port
- Update `riva.client.Auth(uri=...)` with your NVIDIA Riva service address
- Modify absolute path variables at the top of the file:
  - `GPT_SOVITS_PATH`
  - `AUDIO_DIR`
  - `BLENDSHAPE_DIR`
  - `TEMP_DIR`

**Backend (`backend/chat_digital_human_lib.py`):**
- Set `GPT_SOVITS_PATH` to your GPT-SoVITS installation directory
- Update `address` with your DeepSeek LLM API endpoint
- Set `api_key` for LLM authentication

**Frontend (`frontend/src/App.js`):**
- Change `SERVER_IP = "172.16.10.158"` to your backend FastAPI server's IP address

**Scripts (`start_all.sh` & `stop_all.sh`):**
- Update `PROJECT_DIR` and all directory paths to match your server's file system

> **Note**: For production deployments, migrate these configurations to a centralized `.env` or `config.yaml` file.

**3. Start Services**

```bash
# Make scripts executable
chmod +x start_all.sh stop_all.sh

# Start backend and frontend
./start_all.sh

# View logs
tail -f logs/backend.log
tail -f logs/frontend.log

# Stop services
./stop_all.sh
```

**4. Access Interface**

Open your browser and navigate to `http://<your-frontend-server-ip>:3000`. Click the microphone button to start conversing with your digital human.

## Project Structure

```
digital-human/
├── backend/
│   ├── app.py                          # FastAPI server
│   ├── chat_digital_human_lib.py       # Core pipeline logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js                      # Main React component
│   │   └── ...
│   └── package.json
├── logs/                                # Service logs
├── start_all.sh                         # Startup script
└── stop_all.sh                          # Shutdown script
```

## Technical Stack

- **ASR**: NVIDIA Riva
- **LLM**: DeepSeek API
- **TTS**: GPT-SoVITS
- **Facial Animation**: NVIDIA Audio2Face
- **Backend**: FastAPI, Python asyncio
- **Frontend**: React, Three.js (R3F), WebSocket
- **Audio Processing**: FFmpeg

## Performance Characteristics

- **End-to-end latency**: < 500ms (from speech end to animation start)
- **Audio transcoding**: Real-time FFmpeg processing
- **Parallel processing**: Concurrent TTS and A2F generation per sentence
- **Streaming**: Token-level LLM streaming with immediate segmentation



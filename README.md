# Remote Client

## Overview
Remote Client is a high-performance video streaming and decoding client designed for low-latency remote desktop and gaming scenarios. It leverages NVIDIA NVDEC for GPU-accelerated video decoding and Direct3D 12 for efficient rendering.

## Requirements
- Windows 10/11 (x64)
- Visual Studio 2022 (with C++ Desktop Development tools)
- CMake 3.20 or later
- CUDA Toolkit 11.x or later (for NVDEC)
- NVIDIA GPU with NVDEC support
- Git (for cloning dependencies)

## Build Instructions (CMake)

### 1. Clone the repository
```
git clone https://github.com/YukiOhira0416/remote_client.git
cd remote_client
```

### 2. Configure the project
```
cmake -S . -B build -G "Visual Studio 17 2022"
```

### 3. Build (Debug and Release)
#### Debug build
```
cmake --build build --config Debug
```
#### Release build
```
cmake --build build --config Release
```

## Executable Location
- After building, the executable can be found in:
  - `build/Debug/RemoteClient.exe` (Debug)
  - `build/Release/RemoteClient.exe` (Release)

## Project Structure
- `main.cpp` / `main.h` : Application entry point and main logic
- `AppShutdown.cpp` / `AppShutdown.h` : Graceful shutdown handling
- `DebugLog.cpp` / `DebugLog.h` : Logging utilities
- `Globals.cpp` / `Globals.h` : Global variables and settings
- `nvdec.cpp` / `nvdec.h` : NVDEC video decoding logic
- `ReedSolomon.cpp` / `ReedSolomon.h` : FEC (Forward Error Correction) implementation
- `window.cpp` / `window.h` : Window and Direct3D 12 rendering
- `Shader/` : HLSL shaders
- `directx/` : DirectX 12 headers
- `enet_x64-windows/` : ENet networking library (prebuilt)
- `concurrentqueue_x64-windows/` : moodycamel concurrent queue (prebuilt)
- `gf-complete/`, `Jerasure/` : FEC libraries
- `src/NvCodec/` : NVIDIA Codec SDK headers

## Features
- Low-latency video streaming and decoding
- GPU-accelerated H.264 decoding (NVDEC)
- Direct3D 12 rendering
- Forward Error Correction (FEC) for packet loss recovery
- Asynchronous logging
- Multi-threaded networking and decoding
- Customizable window resolution

## Dependencies
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)
- [DirectX 12](https://docs.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-graphics)
- [ENet](http://enet.bespin.org/)
- [moodycamel concurrentqueue](https://github.com/cameron314/concurrentqueue)
- [gf-complete](https://lab.jerasure.org/jerasure/gf-complete)
- [Jerasure](https://lab.jerasure.org/jerasure/jerasure)

---
For more details, see the source code and comments in each file.

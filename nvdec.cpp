#include "nvdec.h"
#include <stdexcept>

// CUDA API error checking
#define CUDA_RUNTIME_CHECK(call)                                                                    \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            const char* error_string = cudaGetErrorString(err);                                     \
            DebugLog(L"CUDA Runtime Error: " + std::wstring(error_string, error_string + strlen(error_string)) + L" in " + \
                     std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            throw std::runtime_error("CUDA Runtime error");                                         \
        }                                                                                           \
    } while (0)

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        CUresult err = call;                                                                        \
        if (err != CUDA_SUCCESS) {                                                                  \
            const char* error_string;                                                               \
            cuGetErrorString(err, &error_string);                                                   \
            DebugLog(L"CUDA Error: " + std::wstring(error_string, error_string + strlen(error_string)) + L" in " + \
                     std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            throw std::runtime_error("CUDA error");                                                 \
        }                                                                                           \
    } while (0)

FrameDecoder::FrameDecoder(CUcontext cuContext, ID3D12Device* pD3D12Device)
    : m_cuContext(cuContext), m_pD3D12Device(pD3D12Device) {
    CUDA_CHECK(cuCtxPushCurrent(m_cuContext));
    // Further initialization
    CUDA_CHECK(cuCtxPopCurrent(NULL));
}

FrameDecoder::~FrameDecoder() {
    cuCtxPushCurrent(m_cuContext);

    if (m_hDecoder) {
        cuvidDestroyDecoder(m_hDecoder);
    }

    // Free CUDA external memory and mapped pointers
    for (auto& resource : m_frameResources) {
        if (resource.mappedCudaPtrY) {
            cuMemFree(resource.mappedCudaPtrY);
        }
        if (resource.mappedCudaPtrUV) {
            cuMemFree(resource.mappedCudaPtrUV);
        }
        if (resource.cudaExtMemY) {
            cudaDestroyExternalMemory(resource.cudaExtMemY);
        }
        if (resource.cudaExtMemUV) {
            cudaDestroyExternalMemory(resource.cudaExtMemUV);
        }
        if(resource.sharedHandleY) CloseHandle(resource.sharedHandleY);
        if(resource.sharedHandleUV) CloseHandle(resource.sharedHandleUV);
    }

    if (m_hParser) {
        cuvidDestroyVideoParser(m_hParser);
    }

    cuCtxPopCurrent(NULL);
}


bool FrameDecoder::Init() {
    CUDA_CHECK(cuCtxPushCurrent(m_cuContext));

    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_H264;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1; // Should be at least 1
    videoParserParameters.ulMaxDisplayDelay = 0; // Low latency
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = HandleVideoSequence;
    videoParserParameters.pfnDecodePicture = HandlePictureDecode;
    videoParserParameters.pfnDisplayPicture = HandlePictureDisplay;

    CUDA_CHECK(cuvidCreateVideoParser(&m_hParser, &videoParserParameters));

    CUDA_CHECK(cuCtxPopCurrent(NULL));
    DebugLog(L"FrameDecoder initialized successfully.");
    return true;
}

void FrameDecoder::Decode(const H264Frame& frame) {
    CUDA_CHECK(cuCtxPushCurrent(m_cuContext));

    CUVIDSOURCEDATAPACKET packet = {};
    packet.payload = frame.data.data();
    packet.payload_size = frame.data.size();
    packet.flags = CUVID_PKT_TIMESTAMP;
    packet.timestamp = frame.timestamp;

    if (!packet.payload || packet.payload_size == 0) {
        DebugLog(L"Decoder::Decode: Empty packet received.");
        return;
    }

    CUDA_CHECK(cuvidParseVideoData(m_hParser, &packet));

    CUDA_CHECK(cuCtxPopCurrent(NULL));
}

int FrameDecoder::HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    DebugLog(L"HandleVideoSequence: Codec: " + std::to_wstring(pVideoFormat->codec) +
             L", Resolution: " + std::to_wstring(pVideoFormat->coded_width) + L"x" + std::to_wstring(pVideoFormat->coded_height));

    if (!self->m_hDecoder) {
        if (!self->createDecoder(pVideoFormat)) {
            DebugLog(L"HandleVideoSequence: Failed to create decoder.");
            return 0; // Stop processing
        }
    } else {
        // Reconfigure decoder if format changes, not handled for simplicity
    }
    return 1; // Proceed with decoding
}

bool FrameDecoder::createDecoder(CUVIDEOFORMAT* pVideoFormat) {
    m_frameWidth = pVideoFormat->coded_width;
    m_frameHeight = pVideoFormat->coded_height;

    memset(&m_videoDecoderCreateInfo, 0, sizeof(m_videoDecoderCreateInfo));
    m_videoDecoderCreateInfo.CodecType = pVideoFormat->codec;
    m_videoDecoderCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    m_videoDecoderCreateInfo.ulWidth = pVideoFormat->coded_width;
    m_videoDecoderCreateInfo.ulHeight = pVideoFormat->coded_height;
    m_videoDecoderCreateInfo.ulNumDecodeSurfaces = 20; // A pool of surfaces
    m_videoDecoderCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    m_videoDecoderCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    m_videoDecoderCreateInfo.ulTargetWidth = m_frameWidth;
    m_videoDecoderCreateInfo.ulTargetHeight = m_frameHeight;
    m_videoDecoderCreateInfo.ulNumOutputSurfaces = 2;
    m_videoDecoderCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

    CUDA_CHECK(cuvidCreateDecoder(&m_hDecoder, &m_videoDecoderCreateInfo));

    if (!allocateFrameBuffers()) {
        DebugLog(L"createDecoder: Failed to allocate frame buffers.");
        return false;
    }
    return true;
}

bool FrameDecoder::allocateFrameBuffers() {
    m_frameResources.resize(m_videoDecoderCreateInfo.ulNumDecodeSurfaces);

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Alignment = 0;
    texDesc.MipLevels = 1;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    for (int i = 0; i < m_videoDecoderCreateInfo.ulNumDecodeSurfaces; ++i) {
        // Y plane
        texDesc.Width = m_frameWidth;
        texDesc.Height = m_frameHeight;
        texDesc.Format = DXGI_FORMAT_R8_UNORM;
        HRESULT hr = m_pD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER, &texDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_frameResources[i].pTextureY));
        if (FAILED(hr)) return false;

        // UV plane
        texDesc.Width = m_frameWidth / 2;
        texDesc.Height = m_frameHeight / 2;
        texDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        hr = m_pD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER, &texDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_frameResources[i].pTextureUV));
        if (FAILED(hr)) return false;

        // Create shared handles
        m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureY.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleY);
        m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureUV.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleUV);

        // Import D3D12 resources into CUDA as external memory
        cudaExternalMemoryHandleDesc extMemHandleDesc = {};
        extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
        extMemHandleDesc.handle.win32.handle = m_frameResources[i].sharedHandleY;
        extMemHandleDesc.size = m_frameResources[i].pTextureY->GetDesc().Width * m_frameResources[i].pTextureY->GetDesc().Height;
        CUDA_RUNTIME_CHECK(cudaImportExternalMemory(&m_frameResources[i].cudaExtMemY, &extMemHandleDesc));

        extMemHandleDesc.handle.win32.handle = m_frameResources[i].sharedHandleUV;
        extMemHandleDesc.size = m_frameResources[i].pTextureUV->GetDesc().Width * m_frameResources[i].pTextureUV->GetDesc().Height * 2;
        CUDA_RUNTIME_CHECK(cudaImportExternalMemory(&m_frameResources[i].cudaExtMemUV, &extMemHandleDesc));

        // Map external memory to CUDA device pointers
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.size = m_frameResources[i].pTextureY->GetDesc().Width * m_frameResources[i].pTextureY->GetDesc().Height;
        CUDA_RUNTIME_CHECK(cudaExternalMemoryGetMappedBuffer(&m_frameResources[i].mappedCudaPtrY, m_frameResources[i].cudaExtMemY, &bufferDesc));

        bufferDesc.size = m_frameResources[i].pTextureUV->GetDesc().Width * m_frameResources[i].pTextureUV->GetDesc().Height * 2;
        CUDA_RUNTIME_CHECK(cudaExternalMemoryGetMappedBuffer(&m_frameResources[i].mappedCudaPtrUV, m_frameResources[i].cudaExtMemUV, &bufferDesc));
    }

    DebugLog(L"Allocated D3D12/CUDA frame buffers.");
    return true;
}

int FrameDecoder::HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    self->m_nDecodePicCnt++;
    CUDA_CHECK(cuvidDecodePicture(self->m_hDecoder, pPicParams));
    return 1;
}

int FrameDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);

    // Map the decoded video frame
    CUVIDPROCPARAMS oVPP = { 0 };
    oVPP.progressive_frame = pDispInfo->progressive_frame;
    oVPP.second_field = 0;
    oVPP.top_field_first = pDispInfo->top_field_first;
    oVPP.unpaired_field = (pDispInfo->progressive_frame == 1 || pDispInfo->repeat_first_field <= 1);

    CUdeviceptr pDecodedFrame = 0;
    unsigned int nDecodedPitch = 0;
    CUDA_CHECK(cuvidMapVideoFrame(self->m_hDecoder, pDispInfo->picture_index, &pDecodedFrame, &nDecodedPitch, &oVPP));

    // Copy to our D3D12 textures
    void* pTexY_void = self->m_frameResources[pDispInfo->picture_index].mappedCudaPtrY;
    void* pTexUV_void = self->m_frameResources[pDispInfo->picture_index].mappedCudaPtrUV;

    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = pDecodedFrame;
    m.srcPitch = nDecodedPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = (CUdeviceptr)pTexY_void;
    m.dstPitch = self->m_frameWidth;
    m.WidthInBytes = self->m_frameWidth;
    m.Height = self->m_frameHeight;
    CUDA_CHECK(cuMemcpy2D(&m));

    m.srcDevice = pDecodedFrame + self->m_frameHeight * nDecodedPitch;
    m.dstDevice = (CUdeviceptr)pTexUV_void;
    m.dstPitch = self->m_frameWidth; // UV plane pitch is also full width in bytes for R8G8
    m.WidthInBytes = self->m_frameWidth;
    m.Height = self->m_frameHeight / 2;
    CUDA_CHECK(cuMemcpy2D(&m));

    // Unmap the frame
    CUDA_CHECK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));

    // Enqueue for rendering
    ReadyGpuFrame readyFrame;
    readyFrame.hw_decoded_texture_Y = self->m_frameResources[pDispInfo->picture_index].pTextureY;
    readyFrame.hw_decoded_texture_UV = self->m_frameResources[pDispInfo->picture_index].pTextureUV;
    readyFrame.timestamp = pDispInfo->timestamp;
    // We need original frame number, but pDispInfo doesn't have it. We can use a counter.
    readyFrame.originalFrameNumber = self->m_nDecodedFrameCount++;
    readyFrame.id = readyFrame.originalFrameNumber;
    readyFrame.width = self->m_frameWidth;
    readyFrame.height = self->m_frameHeight;

    {
        std::lock_guard<std::mutex> lock(g_readyGpuFrameQueueMutex);
        g_readyGpuFrameQueue.push_back(std::move(readyFrame));
    }
    g_readyGpuFrameQueueCV.notify_one();

    return 1;
}

void NvdecThread(int threadId) {
    DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"] started.");

    if (!g_frameDecoder) {
        DebugLog(L"NvdecThread: g_frameDecoder is not initialized!");
        return;
    }

    while (g_fec_worker_Running) { // Use the same global running flag
        H264Frame frame;
        if (g_h264FrameQueue.try_dequeue(frame)) {
            g_frameDecoder->Decode(frame);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"] stopped.");
}

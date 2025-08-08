#include "nvdec.h"
#include "Globals.h"
#include <stdexcept>
#include <fstream>
#include <vector>

// CUDA API error checking
#define CUDA_RUNTIME_CHECK(call)                                                                    \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            const char* error_string_ptr = cudaGetErrorString(err);                                 \
            if (error_string_ptr == nullptr) {                                                      \
                DebugLog(L"CUDA Runtime Error: Unknown error code " + std::to_wstring(err) + L" in " + \
                         std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            } else {                                                                                \
                char safe_error_string[512];                                                        \
                strncpy(safe_error_string, error_string_ptr, sizeof(safe_error_string));            \
                safe_error_string[sizeof(safe_error_string) - 1] = '\0';                            \
                DebugLog(L"CUDA Runtime Error: " + std::wstring(safe_error_string, safe_error_string + strlen(safe_error_string)) + L" in " + \
                         std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            }                                                                                       \
            throw std::runtime_error("CUDA Runtime error");                                         \
        }                                                                                           \
    } while (0)

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        CUresult err = call;                                                                        \
        if (err != CUDA_SUCCESS) {                                                                  \
            const char* error_string_ptr;                                                           \
            cuGetErrorString(err, &error_string_ptr);                                               \
            if (error_string_ptr == nullptr) {                                                      \
                DebugLog(L"CUDA Error: Unknown error code " + std::to_wstring(err) + L" in " +      \
                         std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            } else {                                                                                \
                char safe_error_string[512];                                                        \
                strncpy(safe_error_string, error_string_ptr, sizeof(safe_error_string));            \
                safe_error_string[sizeof(safe_error_string) - 1] = '\0';                            \
                DebugLog(L"CUDA Error: " + std::wstring(safe_error_string, safe_error_string + strlen(safe_error_string)) + L" in " + \
                         std::wstring(__FILEW__, __FILEW__ + wcslen(__FILEW__)) + L" at line " + std::to_wstring(__LINE__)); \
            }                                                                                       \
            throw std::runtime_error("CUDA error");                                                 \
        }                                                                                           \
    } while (0)

// BMP file header structures
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType = 0x4D42;  // "BM"
    uint32_t bfSize;
    uint16_t bfReserved1 = 0;
    uint16_t bfReserved2 = 0;
    uint32_t bfOffBits = 54;
};

struct BMPInfoHeader {
    uint32_t biSize = 40;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes = 1;
    uint16_t biBitCount = 8;
    uint32_t biCompression = 0;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter = 2835;
    int32_t biYPelsPerMeter = 2835;
    uint32_t biClrUsed = 256;
    uint32_t biClrImportant = 0;
};
#pragma pack(pop)

void SaveYUVPlaneAsBMP(void* cudaPtr, int width, int height, int pitch, const std::string& filename) {
    // Copy data from GPU to CPU
    std::vector<uint8_t> hostData(pitch * height);
    CUDA_RUNTIME_CHECK(cudaMemcpy(hostData.data(), cudaPtr, pitch * height, cudaMemcpyDeviceToHost));
    
    // Calculate BMP parameters
    int bmpWidth = width;
    int bmpHeight = height;
    int rowSize = ((bmpWidth + 3) / 4) * 4; // 4-byte alignment
    int imageSize = rowSize * bmpHeight;
    
    BMPFileHeader fileHeader;
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 256*4 + imageSize;
    
    BMPInfoHeader infoHeader;
    infoHeader.biWidth = bmpWidth;
    infoHeader.biHeight = bmpHeight;
    infoHeader.biSizeImage = imageSize;
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;
    
    // Write headers
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
    
    // Write grayscale palette
    for (int i = 0; i < 256; i++) {
        uint8_t color[4] = {static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0};
        file.write(reinterpret_cast<const char*>(color), 4);
    }
    
    // Write image data (bottom-up)
    std::vector<uint8_t> row(rowSize, 0);
    for (int y = bmpHeight - 1; y >= 0; y--) {
        std::memcpy(row.data(), hostData.data() + y * pitch, std::min(bmpWidth, pitch));
        file.write(reinterpret_cast<const char*>(row.data()), rowSize);
    }
}

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
            cuMemFree((CUdeviceptr)resource.mappedCudaPtrY);
        }
        if (resource.mappedCudaPtrUV) {
            cuMemFree((CUdeviceptr)resource.mappedCudaPtrUV);
        }
        if (resource.cudaExtMemY) {
            cudaDestroyExternalMemory(resource.cudaExtMemY);
        }
        if (resource.cudaExtMemUV) {
            cudaDestroyExternalMemory(resource.cudaExtMemUV);
        }
        if(resource.sharedHandleY) CloseHandle(resource.sharedHandleY);
        if(resource.sharedHandleUV) CloseHandle(resource.sharedHandleUV);
        // Heaps are released by ComPtr automatically
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

    for (int i = 0; i < m_videoDecoderCreateInfo.ulNumDecodeSurfaces; ++i) {
        // --- Y Plane ---
        D3D12_RESOURCE_DESC texDescY = {};
        texDescY.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDescY.Alignment = 0;
        texDescY.DepthOrArraySize = 1;
        texDescY.MipLevels = 1;
        texDescY.SampleDesc.Count = 1;
        texDescY.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        texDescY.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        texDescY.Width = m_frameWidth;
        texDescY.Height = m_frameHeight;
        texDescY.Format = DXGI_FORMAT_R8_UNORM;

        D3D12_RESOURCE_ALLOCATION_INFO allocInfoY = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescY);

        D3D12_HEAP_DESC heapDescY = {};
        heapDescY.SizeInBytes = allocInfoY.SizeInBytes;
        heapDescY.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapDescY.Alignment = allocInfoY.Alignment;
        heapDescY.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

        HRESULT hr = m_pD3D12Device->CreateHeap(&heapDescY, IID_PPV_ARGS(&m_frameResources[i].pHeapY));
        if(FAILED(hr)) return false;

        hr = m_pD3D12Device->CreatePlacedResource(m_frameResources[i].pHeapY.Get(), 0, &texDescY, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_frameResources[i].pTextureY));
        if(FAILED(hr)) return false;

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedFootprintY = {};
        m_pD3D12Device->GetCopyableFootprints(&texDescY, 0, 1, 0, &placedFootprintY, nullptr, nullptr, nullptr);
        m_frameResources[i].pitchY = placedFootprintY.Footprint.RowPitch;

        // --- UV Plane ---
        D3D12_RESOURCE_DESC texDescUV = texDescY;
        texDescUV.Width = m_frameWidth / 2;
        texDescUV.Height = m_frameHeight / 2;
        texDescUV.Format = DXGI_FORMAT_R8G8_UNORM;

        D3D12_RESOURCE_ALLOCATION_INFO allocInfoUV = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescUV);

        D3D12_HEAP_DESC heapDescUV = {};
        heapDescUV.SizeInBytes = allocInfoUV.SizeInBytes;
        heapDescUV.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapDescUV.Alignment = allocInfoUV.Alignment;
        heapDescUV.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

        hr = m_pD3D12Device->CreateHeap(&heapDescUV, IID_PPV_ARGS(&m_frameResources[i].pHeapUV));
        if(FAILED(hr)) return false;

        hr = m_pD3D12Device->CreatePlacedResource(m_frameResources[i].pHeapUV.Get(), 0, &texDescUV, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_frameResources[i].pTextureUV));
        if(FAILED(hr)) return false;

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedFootprintUV = {};
        m_pD3D12Device->GetCopyableFootprints(&texDescUV, 0, 1, 0, &placedFootprintUV, nullptr, nullptr, nullptr);
        m_frameResources[i].pitchUV = placedFootprintUV.Footprint.RowPitch;

        // --- CUDA Interop ---
        // Create shared handles for the heaps
        m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pHeapY.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleY);
        m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pHeapUV.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleUV);

        // Import D3D12 heaps into CUDA as external memory
        cudaExternalMemoryHandleDesc extMemHandleDescY = {};
        extMemHandleDescY.type = cudaExternalMemoryHandleTypeD3D12Heap;
        extMemHandleDescY.handle.win32.handle = m_frameResources[i].sharedHandleY;
        extMemHandleDescY.size = allocInfoY.SizeInBytes;
        CUDA_RUNTIME_CHECK(cudaImportExternalMemory(&m_frameResources[i].cudaExtMemY, &extMemHandleDescY));

        cudaExternalMemoryHandleDesc extMemHandleDescUV = {};
        extMemHandleDescUV.type = cudaExternalMemoryHandleTypeD3D12Heap;
        extMemHandleDescUV.handle.win32.handle = m_frameResources[i].sharedHandleUV;
        extMemHandleDescUV.size = allocInfoUV.SizeInBytes;
        CUDA_RUNTIME_CHECK(cudaImportExternalMemory(&m_frameResources[i].cudaExtMemUV, &extMemHandleDescUV));

        // Map external memory to CUDA device pointers
        cudaExternalMemoryBufferDesc bufferDescY = {};
        bufferDescY.offset = 0;
        bufferDescY.size = allocInfoY.SizeInBytes;
        CUDA_RUNTIME_CHECK(cudaExternalMemoryGetMappedBuffer(&m_frameResources[i].mappedCudaPtrY, m_frameResources[i].cudaExtMemY, &bufferDescY));

        cudaExternalMemoryBufferDesc bufferDescUV = {};
        bufferDescUV.offset = 0;
        bufferDescUV.size = allocInfoUV.SizeInBytes;
        CUDA_RUNTIME_CHECK(cudaExternalMemoryGetMappedBuffer(&m_frameResources[i].mappedCudaPtrUV, m_frameResources[i].cudaExtMemUV, &bufferDescUV));
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
    m.dstPitch = self->m_frameResources[pDispInfo->picture_index].pitchY;
    m.WidthInBytes = self->m_frameWidth;
    m.Height = self->m_frameHeight;
    CUDA_CHECK(cuMemcpy2D(&m));

    m.srcDevice = pDecodedFrame + (size_t)self->m_frameHeight * nDecodedPitch;
    m.dstDevice = (CUdeviceptr)pTexUV_void;
    m.dstPitch = self->m_frameResources[pDispInfo->picture_index].pitchUV;
    m.WidthInBytes = self->m_frameWidth / 2 * 2; // width for R8G8
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

    // Save Y and UV planes as BMP files
    static int bmpCounter = 0;
    std::string yFilename = "frame_Y_" + std::to_string(bmpCounter) + ".bmp";
    std::string uvFilename = "frame_UV_" + std::to_string(bmpCounter) + ".bmp";
    SaveYUVPlaneAsBMP(pTexY_void, self->m_frameWidth, self->m_frameHeight,
                      self->m_frameResources[pDispInfo->picture_index].pitchY, yFilename);
    SaveYUVPlaneAsBMP(pTexUV_void, self->m_frameWidth / 2, self->m_frameHeight / 2,
                      self->m_frameResources[pDispInfo->picture_index].pitchUV, uvFilename);
    bmpCounter++;

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

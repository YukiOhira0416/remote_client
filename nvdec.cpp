#include "nvdec.h"
#include "Globals.h"
#include "NvdecKernels.h"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

// CUDA API error checking
#define CUDA_RUNTIME_CHECK(call)                                                                    \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            const char* error_string_ptr = cudaGetErrorString(err);                                 \
            std::wstringstream wss;                                                                 \
            wss << L"CUDA Runtime Error: ";                                                         \
            if (error_string_ptr == nullptr) {                                                      \
                wss << L"Unknown error code " << err;                                               \
            } else {                                                                                \
                std::string narrow_str(error_string_ptr);                                           \
                wss << std::wstring(narrow_str.begin(), narrow_str.end());                          \
            }                                                                                       \
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                 \
            DebugLog(wss.str());                                                                    \
            throw std::runtime_error("CUDA Runtime error");                                         \
        }                                                                                           \
    } while (0)

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        CUresult err = call;                                                                        \
        if (err != CUDA_SUCCESS) {                                                                  \
            const char* error_string_ptr;                                                           \
            cuGetErrorString(err, &error_string_ptr);                                               \
            std::wstringstream wss;                                                                 \
            wss << L"CUDA Error: ";                                                                 \
            if (error_string_ptr == nullptr) {                                                      \
                wss << L"Unknown error code " << err;                                               \
            } else {                                                                                \
                std::string narrow_str(error_string_ptr);                                           \
                wss << std::wstring(narrow_str.begin(), narrow_str.end());                          \
            }                                                                                       \
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                 \
            DebugLog(wss.str());                                                                    \
            throw std::runtime_error("CUDA error");                                                 \
        }                                                                                           \
    } while (0)

// Callback-safe versions of the macros that do not throw exceptions
#define CUDA_RUNTIME_CHECK_CALLBACK(call)                                                           \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            const char* error_string_ptr = cudaGetErrorString(err);                                 \
            std::wstringstream wss;                                                                 \
            wss << L"CUDA Runtime Error in callback: ";                                             \
            if (error_string_ptr == nullptr) {                                                      \
                wss << L"Unknown error code " << err;                                               \
            } else {                                                                                \
                char safe_error_string[256];                                                        \
                strncpy(safe_error_string, error_string_ptr, sizeof(safe_error_string));            \
                safe_error_string[sizeof(safe_error_string) - 1] = '\0';                            \
                std::string narrow_str(safe_error_string);                                          \
                wss << std::wstring(narrow_str.begin(), narrow_str.end());                          \
            }                                                                                       \
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                 \
            DebugLog(wss.str());                                                                    \
            return 0; /* Return 0 on error, do not throw */                                         \
        }                                                                                           \
    } while (0)

#define CUDA_CHECK_CALLBACK(call)                                                                   \
    do {                                                                                            \
        CUresult err = call;                                                                        \
        if (err != CUDA_SUCCESS) {                                                                  \
            const char* error_string_ptr;                                                           \
            cuGetErrorString(err, &error_string_ptr);                                               \
            std::wstringstream wss;                                                                 \
            wss << L"CUDA Error in callback: ";                                                     \
            if (error_string_ptr == nullptr) {                                                      \
                wss << L"Unknown error code " << err;                                               \
            } else {                                                                                \
                char safe_error_string[256];                                                        \
                strncpy(safe_error_string, error_string_ptr, sizeof(safe_error_string));            \
                safe_error_string[sizeof(safe_error_string) - 1] = '\0';                            \
                std::string narrow_str(safe_error_string);                                          \
                wss << std::wstring(narrow_str.begin(), narrow_str.end());                          \
            }                                                                                       \
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                 \
            DebugLog(wss.str());                                                                    \
            return 0; /* Return 0 on error, do not throw */                                         \
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

int SaveYUVPlaneAsBMP(void* cudaPtr, int width, int height, int pitch, const std::string& filename) {
    if (!cudaPtr || width <= 0 || height <= 0 || pitch <= 0) {
        DebugLog(L"SaveYUVPlaneAsBMP: Invalid parameters");
        return 0;
    }

    // 一度に全データをコピーする方式に変更
    std::vector<uint8_t> hostData(pitch * height);
    
    // 非同期コピーの代わりに同期コピーを使用
    CUDA_RUNTIME_CHECK_CALLBACK(cudaMemcpy(
        hostData.data(),
        cudaPtr,
        pitch * height,
        cudaMemcpyDeviceToHost
    ));
    
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
    if (!file) return 0;
    
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
    return 1;
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
        if (resource.cudaExtMemY) {
            cuDestroyExternalMemory(resource.cudaExtMemY);
        }
        if (resource.cudaExtMemUV) {
            cuDestroyExternalMemory(resource.cudaExtMemUV);
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
    videoParserParameters.ulMaxNumDecodeSurfaces = FrameDecoder::NUM_DECODE_SURFACES;
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
    cuCtxPushCurrent(self->m_cuContext);
    int result = 1;

    try {
        DebugLog(L"HandleVideoSequence: Codec: " + std::to_wstring(pVideoFormat->codec) +
            L", Resolution: " + std::to_wstring(pVideoFormat->coded_width) + L"x" + std::to_wstring(pVideoFormat->coded_height));

        if (!self->m_hDecoder) {
            if (!self->createDecoder(pVideoFormat)) {
                DebugLog(L"HandleVideoSequence: Failed to create decoder.");
                result = 0; // Stop processing
            }
        }
        else {
            // Reconfigure decoder if format changes, not handled for simplicity
        }
    }
    catch (const std::runtime_error& e) {
        // The error is already logged by the CUDA_CHECK macro.
        // We catch the exception to prevent it from crossing the C API boundary.
        DebugLog(L"Caught exception in HandleVideoSequence: " + std::wstring(e.what(), e.what() + strlen(e.what())));
        result = 0; // Stop processing
    }
    catch (...) {
        // Catch any other exceptions.
        DebugLog(L"Caught unknown exception in HandleVideoSequence.");
        result = 0;
    }

    cuCtxPopCurrent(NULL);
    return result; // Proceed with decoding
}

bool FrameDecoder::createDecoder(CUVIDEOFORMAT* pVideoFormat) {
    // Get the target display resolution from global variables.
    // This resolution is determined by the window size and sent to the server.
    int targetWidth = ::currentResolutionWidth.load();
    int targetHeight = ::currentResolutionHeight.load();

    // If the target resolution hasn't been set yet, default to the stream's coded size.
    if (targetWidth == 0 || targetHeight == 0) {
        targetWidth = pVideoFormat->coded_width;
        targetHeight = pVideoFormat->coded_height;
    }

    // Set the class members to the target display size.
    // This will be used for texture allocation and the copy size.
    m_frameWidth = targetWidth;
    m_frameHeight = targetHeight;

    // The rest of the function initializes the decoder with the *actual* stream
    // dimensions, but allocates buffers for the *target* dimensions.
    memset(&m_videoDecoderCreateInfo, 0, sizeof(m_videoDecoderCreateInfo));
    m_videoDecoderCreateInfo.CodecType = pVideoFormat->codec;
    m_videoDecoderCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    m_videoDecoderCreateInfo.ulWidth = pVideoFormat->coded_width;
    m_videoDecoderCreateInfo.ulHeight = pVideoFormat->coded_height;
    m_videoDecoderCreateInfo.ulNumDecodeSurfaces = FrameDecoder::NUM_DECODE_SURFACES; // A pool of surfaces
    m_videoDecoderCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    m_videoDecoderCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    // Set target size to the actual coded size so decoder does NO scaling.
    // We will perform a crop manually during the cuMemcpy2D.
    m_videoDecoderCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
    m_videoDecoderCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    m_videoDecoderCreateInfo.ulNumOutputSurfaces = 2;
    m_videoDecoderCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

    CUDA_CHECK(cuvidCreateDecoder(&m_hDecoder, &m_videoDecoderCreateInfo));

    if (!allocateFrameBuffers()) { // This will now use targetWidth x targetHeight
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
        // Allocate based on the coded width of the video stream to avoid cropping during copy.
        // The logical frame width (m_frameWidth) is still used for rendering/cropping.
        texDescY.Width = m_videoDecoderCreateInfo.ulWidth;
        // Allocate based on the coded height of the video stream to avoid cropping during copy.
        // The logical frame height (m_frameHeight) is still used for rendering.
        texDescY.Height = m_videoDecoderCreateInfo.ulHeight;
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
        // The width must also be based on the coded width.
        texDescUV.Width = m_videoDecoderCreateInfo.ulWidth / 2;
        texDescUV.Height = m_videoDecoderCreateInfo.ulHeight / 2;
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

        // Import D3D12 heaps into CUDA as external memory using Driver API
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemHandleDescY = {};
        extMemHandleDescY.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
        extMemHandleDescY.handle.win32.handle = m_frameResources[i].sharedHandleY;
        extMemHandleDescY.size = allocInfoY.SizeInBytes;
        CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemY, &extMemHandleDescY));

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemHandleDescUV = {};
        extMemHandleDescUV.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
        extMemHandleDescUV.handle.win32.handle = m_frameResources[i].sharedHandleUV;
        extMemHandleDescUV.size = allocInfoUV.SizeInBytes;
        CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemUV, &extMemHandleDescUV));

        // Map external memory to CUDA device pointers using Driver API
        CUdeviceptr devPtrY = 0;
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDescY = {};
        bufferDescY.offset = 0;
        bufferDescY.size = allocInfoY.SizeInBytes;
        bufferDescY.flags = 0;
        CUDA_CHECK(cuExternalMemoryGetMappedBuffer(&devPtrY, m_frameResources[i].cudaExtMemY, &bufferDescY));
        m_frameResources[i].mappedCudaPtrY = (void*)devPtrY;

        CUdeviceptr devPtrUV = 0;
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDescUV = {};
        bufferDescUV.offset = 0;
        bufferDescUV.size = allocInfoUV.SizeInBytes;
        bufferDescUV.flags = 0;
        CUDA_CHECK(cuExternalMemoryGetMappedBuffer(&devPtrUV, m_frameResources[i].cudaExtMemUV, &bufferDescUV));
        m_frameResources[i].mappedCudaPtrUV = (void*)devPtrUV;
    }

    DebugLog(L"Allocated D3D12/CUDA frame buffers.");
    return true;
}

int FrameDecoder::HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);

    self->m_nDecodePicCnt++;
    CUDA_CHECK_CALLBACK(cuvidDecodePicture(self->m_hDecoder, pPicParams));

    cuCtxPopCurrent(NULL);
    return 1;
}

int FrameDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);

    // Map the decoded video frame
    CUVIDPROCPARAMS oVPP = { 0 };
    oVPP.progressive_frame = pDispInfo->progressive_frame;
    oVPP.second_field = 0;
    oVPP.top_field_first = pDispInfo->top_field_first;
    oVPP.unpaired_field = (pDispInfo->progressive_frame == 1 || pDispInfo->repeat_first_field <= 1);

    CUdeviceptr pDecodedFrame = 0;
    unsigned int nDecodedPitch = 0;
    CUDA_CHECK_CALLBACK(cuvidMapVideoFrame(self->m_hDecoder, pDispInfo->picture_index, &pDecodedFrame, &nDecodedPitch, &oVPP));

    // --- Y/UV コピーをドライバAPIで行う ---
    // 理由:
    // 1) pDecodedFrame は CUVID が返す CUdeviceptr（ドライバAPI領域）
    // 2) ランタイムAPIの cudaMemcpy2D と混用すると invalid argument を起こすことがある
    // 3) cuMemcpy2D{Async} は CUdeviceptr を正式に扱える

    // 事前検証：nullptrや0ピッチ、幅超過などを明示チェック
    auto& fr = self->m_frameResources[pDispInfo->picture_index];

    if (!fr.mappedCudaPtrY || !fr.mappedCudaPtrUV) {
        DebugLog(L"HandlePictureDisplay: mapped CUDA pointers are null. Aborting.");
        CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));
        cuCtxPopCurrent(NULL);
        return 0;
    }

    const size_t dpitchY  = static_cast<size_t>(fr.pitchY);
    const size_t dpitchUV = static_cast<size_t>(fr.pitchUV);
    const size_t spitch   = static_cast<size_t>(nDecodedPitch);

    const size_t srcWidthBytes_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);       // NV12 Y: 1B/px
    const size_t srcHeightRows_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight);
    const size_t srcWidthBytes_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);       // NV12 UV: 幅は同じ(2chで計2B/2px)
    const size_t srcHeightRows_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight / 2);

    auto logParams = [&](const wchar_t* tag, size_t w, size_t h, size_t dp, size_t sp) {
        std::wstringstream wss;
        wss << L"[CopyParams-" << tag << L"] widthBytes=" << w
            << L", height=" << h
            << L", dpitch=" << dp
            << L", spitch=" << sp;
        DebugLog(wss.str());
    };

    logParams(L"Y",  srcWidthBytes_Y,  srcHeightRows_Y,  dpitchY,  spitch);
    logParams(L"UV", srcWidthBytes_UV, srcHeightRows_UV, dpitchUV, spitch);

    // 幅チェック: widthInBytes は dpitch と spitch を超えてはいけない
    if (srcWidthBytes_Y  == 0 || srcHeightRows_Y  == 0 ||
        srcWidthBytes_UV == 0 || srcHeightRows_UV == 0 ||
        dpitchY  == 0 || dpitchUV == 0 || spitch == 0) {
        DebugLog(L"HandlePictureDisplay: zero width/height/pitch detected. Aborting.");
        CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));
        cuCtxPopCurrent(NULL);
        return 0;
    }

    if (srcWidthBytes_Y  > dpitchY || srcWidthBytes_Y  > spitch ||
        srcWidthBytes_UV > dpitchUV || srcWidthBytes_UV > spitch) {
        DebugLog(L"HandlePictureDisplay: widthInBytes exceeds pitch (Y/UV). Aborting copy.");
        CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));
        cuCtxPopCurrent(NULL);
        return 0;
    }

    // ドライバAPIの2Dコピー記述子を用意
    CUDA_MEMCPY2D cpyY = {};
    cpyY.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyY.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyY.srcDevice     = (CUdeviceptr)pDecodedFrame;
    cpyY.srcPitch      = nDecodedPitch;
    cpyY.dstDevice     = (CUdeviceptr)fr.mappedCudaPtrY;
    cpyY.dstPitch      = fr.pitchY;
    cpyY.WidthInBytes  = (unsigned int)srcWidthBytes_Y;
    cpyY.Height        = (unsigned int)srcHeightRows_Y;

    const uint8_t* pSrcUV = (const uint8_t*)pDecodedFrame + (size_t)srcHeightRows_Y * nDecodedPitch;

    CUDA_MEMCPY2D cpyUV = {};
    cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyUV.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyUV.srcDevice     = (CUdeviceptr)pSrcUV;
    cpyUV.srcPitch      = nDecodedPitch;
    cpyUV.dstDevice     = (CUdeviceptr)fr.mappedCudaPtrUV;
    cpyUV.dstPitch      = fr.pitchUV;
    cpyUV.WidthInBytes  = (unsigned int)srcWidthBytes_UV;
    cpyUV.Height       = (unsigned int)srcHeightRows_UV;

    // 同期コピー（非同期にしたい場合は cuMemcpy2DAsync(&cpy, 0) を使う）
    CUDA_CHECK_CALLBACK(cuMemcpy2D(&cpyY));
    CUDA_CHECK_CALLBACK(cuMemcpy2D(&cpyUV));

    // エラーを早期検出（ドライバAPIなら cuCtxSynchronize を使う）
    CUDA_CHECK_CALLBACK(cuCtxSynchronize());

    // 以降は現行の処理を継続
    CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));

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

    // Save Y and UV planes as BMP files for debugging, capped to first 10 frames.
    static int bmpCounter = 0;
    if (bmpCounter < 10) {
        std::string yFilename = "frame_Y_" + std::to_string(bmpCounter) + ".bmp";
        std::string uvFilename = "frame_UV_" + std::to_string(bmpCounter) + ".bmp";

        // For debugging, save the raw decoded buffer. Use the buffer's actual allocated height
        // (m_videoDecoderCreateInfo.ulHeight) for the copy, not the logical display height (m_frameHeight),
        // to prevent reading out of bounds. The saved BMP will have the dimensions of the buffer.
        if (SaveYUVPlaneAsBMP(fr.mappedCudaPtrY, self->m_frameWidth, self->m_videoDecoderCreateInfo.ulHeight,
            fr.pitchY, yFilename) == 0) {
            // Error is logged within the function.
            DebugLog(L"HandlePictureDisplay: Failed to save Y plane BMP, aborting callback.");
            return 0;
        }
        if (SaveYUVPlaneAsBMP(fr.mappedCudaPtrUV, self->m_frameWidth / 2, self->m_videoDecoderCreateInfo.ulHeight / 2,
            fr.pitchUV, uvFilename) == 0) {
            // Error is logged within the function.
            DebugLog(L"HandlePictureDisplay: Failed to save UV plane BMP, aborting callback.");
            return 0;
        }
        bmpCounter++;
    }


    {
        std::lock_guard<std::mutex> lock(g_readyGpuFrameQueueMutex);
        g_readyGpuFrameQueue.push_back(std::move(readyFrame));
    }
    g_readyGpuFrameQueueCV.notify_one();

    cuCtxPopCurrent(NULL);
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

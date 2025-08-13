#include "nvdec.h"
#include "Globals.h"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include "src/NvCodec/bmp_writer.hpp"
#include <atomic>

// ---- BMP writer task queue (thread-safe) ----
static BmpWriter g_bmpWriter;
static std::atomic<uint64_t> g_bmpSeq{0};

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
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                \
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
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                \
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
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                \
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
            wss << L" in " << __FILEW__ << L" at line " << __LINE__;                                \
            DebugLog(wss.str());                                                                    \
            return 0; /* Return 0 on error, do not throw */                                         \
        }                                                                                           \
    } while (0)



// === ここまで追加 ===

FrameDecoder::FrameDecoder(CUcontext cuContext, ID3D12Device* pD3D12Device)
    : m_cuContext(cuContext), m_pD3D12Device(pD3D12Device) {
    CUDA_CHECK(cuCtxPushCurrent(m_cuContext));
    // NVDEC公式のコンテキストロックを作る（コールバック/内部スレッドと競合しないように）
    CUresult cr = cuvidCtxLockCreate(&m_ctxLock, m_cuContext);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuvidCtxLockCreate failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        throw std::runtime_error("cuvidCtxLockCreate failed");
    }
    CUDA_CHECK(cuCtxPopCurrent(NULL));
}

FrameDecoder::~FrameDecoder() {
    cuCtxPushCurrent(m_cuContext);

    if (m_hDecoder) {
        cuvidDestroyDecoder(m_hDecoder);
    }

    // Free CUDA external memory and mapped pointers
    for (auto& resource : m_frameResources) {
        // The CUarray and CUmipmappedArray are derived from the external memory and
        // are invalidated when the external memory is destroyed. We don't need to
        // free them separately, but we null them out for correctness.
        resource.pCudaArrayY = nullptr;
        resource.pCudaArrayUV = nullptr;
        resource.pMipmappedArrayY = nullptr;
        resource.pMipmappedArrayUV = nullptr;

        if (resource.cudaExtMemY) {
            cuDestroyExternalMemory(resource.cudaExtMemY);
        }
        if (resource.cudaExtMemUV) {
            cuDestroyExternalMemory(resource.cudaExtMemUV);
        }
        if(resource.sharedHandleY) CloseHandle(resource.sharedHandleY);
        if(resource.sharedHandleUV) CloseHandle(resource.sharedHandleUV);
        // Heaps and textures are released by ComPtr automatically
    }

    if (m_hParser) {
        cuvidDestroyVideoParser(m_hParser);
    }

    // 追加：ロック破棄
    if (m_ctxLock) {
        cuvidCtxLockDestroy(m_ctxLock);
        m_ctxLock = nullptr;
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

    // timestamp -> frameNumber を記録 (from original)
    {
        std::lock_guard<std::mutex> lk(m_tsMapMutex);
        m_tsToFrameNo[frame.timestamp] = frame.frameNumber;
    }

    { // 複数スレッドからの呼び出しを直列化
        std::lock_guard<std::mutex> lk(m_parseMutex);
        CUVIDSOURCEDATAPACKET packet = {};
        packet.payload = frame.data.data();
        packet.payload_size = frame.data.size();
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.timestamp = frame.timestamp;

        if (!packet.payload || packet.payload_size == 0) {
            DebugLog(L"Decoder::Decode: Empty packet received.");
            // Fall through to pop context and return.
        } else {
            // NVDEC API 呼び出し前後をロック（公式の方法）
            cuvidCtxLock(m_ctxLock, 0);
            CUresult cr = cuvidParseVideoData(m_hParser, &packet);
            cuvidCtxUnlock(m_ctxLock, 0);

            if (cr != CUDA_SUCCESS) {
                const char* es = nullptr; cuGetErrorString(cr, &es);
                std::wstring msg = L"cuvidParseVideoData failed: ";
                if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
                DebugLog(msg);
                // 継続可能：ここでは throw せず、末尾の PopCurrent に任せる
            }
        }
    }
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
    // pHeapY/pHeapUV are no longer used but can remain in the struct
    m_frameResources.resize(m_videoDecoderCreateInfo.ulNumDecodeSurfaces);

    for (UINT i = 0; i < m_videoDecoderCreateInfo.ulNumDecodeSurfaces; ++i) {
        // -----------------------------
        // 1) D3D12 Texture (Committed)
        // -----------------------------
        // --- Y plane ---
        D3D12_RESOURCE_DESC texDescY = {};
        texDescY.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDescY.Alignment = 0;
        texDescY.Width  = m_videoDecoderCreateInfo.ulWidth;   // coded width
        texDescY.Height = m_videoDecoderCreateInfo.ulHeight;  // coded height
        texDescY.DepthOrArraySize = 1;
        texDescY.MipLevels = 1;
        texDescY.Format = DXGI_FORMAT_R8_UNORM;
        texDescY.SampleDesc.Count = 1;
        texDescY.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; // No more ROW_MAJOR
        texDescY.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS; // No more CrossAdapter

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        Microsoft::WRL::ComPtr<ID3D12Resource> texY;
        HRESULT hr = m_pD3D12Device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_SHARED,                 // Shared
            &texDescY,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&texY)
        );
        if (FAILED(hr)) {
            DebugLog(L"allocateFrameBuffers: CreateCommittedResource(Y) failed.");
            return false;
        }

        // Calculate RowPitch (for CUDA copy size and desc.size reference)
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpY = {};
        UINT numRowsY = 0; UINT64 rowSizeInBytesY = 0; UINT64 totalBytesY = 0;
        m_pD3D12Device->GetCopyableFootprints(&texDescY, 0, 1, 0, &fpY, &numRowsY, &rowSizeInBytesY, &totalBytesY);
        UINT pitchY = fpY.Footprint.RowPitch;

        // Get allocation size (also from desc just in case)
        D3D12_RESOURCE_ALLOCATION_INFO allocInfoY = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescY);
        UINT64 ySizeCandidate = static_cast<UINT64>(pitchY) * texDescY.Height;
        UINT64 yImportSize    = (allocInfoY.SizeInBytes > ySizeCandidate) ? allocInfoY.SizeInBytes : ySizeCandidate;

        // --- UV plane ---
        D3D12_RESOURCE_DESC texDescUV = texDescY;
        texDescUV.Width  = m_videoDecoderCreateInfo.ulWidth  / 2;
        texDescUV.Height = m_videoDecoderCreateInfo.ulHeight / 2;
        texDescUV.Format = DXGI_FORMAT_R8G8_UNORM;

        Microsoft::WRL::ComPtr<ID3D12Resource> texUV;
        hr = m_pD3D12Device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_SHARED,
            &texDescUV,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&texUV)
        );
        if (FAILED(hr)) {
            DebugLog(L"allocateFrameBuffers: CreateCommittedResource(UV) failed.");
            return false;
        }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpUV = {};
        UINT numRowsUV = 0; UINT64 rowSizeInBytesUV = 0; UINT64 totalBytesUV = 0;
        m_pD3D12Device->GetCopyableFootprints(&texDescUV, 0, 1, 0, &fpUV, &numRowsUV, &rowSizeInBytesUV, &totalBytesUV);
        UINT pitchUV = fpUV.Footprint.RowPitch;

        D3D12_RESOURCE_ALLOCATION_INFO allocInfoUV = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescUV);
        UINT64 uvSizeCandidate = static_cast<UINT64>(pitchUV) * texDescUV.Height;
        UINT64 uvImportSize    = (allocInfoUV.SizeInBytes > uvSizeCandidate) ? allocInfoUV.SizeInBytes : uvSizeCandidate;

        // Assign ComPtr to members (consider cleanup on failure from here)
        m_frameResources[i].pTextureY = texY;
        m_frameResources[i].pTextureUV = texUV;
        m_frameResources[i].pitchY = pitchY;
        m_frameResources[i].pitchUV = pitchUV;

        // -----------------------------
        // 2) Shared Handle (Resource)
        // -----------------------------
        HANDLE hY = nullptr, hUV = nullptr;
        hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureY.Get(), nullptr, GENERIC_ALL, nullptr, &hY);
        if (FAILED(hr)) {
            DebugLog(L"allocateFrameBuffers: CreateSharedHandle(Y) failed.");
            return false;
        }
        hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureUV.Get(), nullptr, GENERIC_ALL, nullptr, &hUV);
        if (FAILED(hr)) {
            CloseHandle(hY);
            DebugLog(L"allocateFrameBuffers: CreateSharedHandle(UV) failed.");
            return false;
        }
        m_frameResources[i].sharedHandleY = hY;
        m_frameResources[i].sharedHandleUV = hUV;

        // -----------------------------
        // 3) Import as CUDA External Memory (Dedicated)
        // -----------------------------
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC extY = {};
        extY.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        extY.handle.win32.handle = m_frameResources[i].sharedHandleY;
        extY.size  = yImportSize;                    // Use the larger of allocation or RowPitch*Height
        extY.flags = cudaExternalMemoryDedicated;    // Must be Dedicated for Committed
        CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemY, &extY));

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC extUV = {};
        extUV.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        extUV.handle.win32.handle = m_frameResources[i].sharedHandleUV;
        extUV.size  = uvImportSize;
        extUV.flags = cudaExternalMemoryDedicated;   // Must be Dedicated for Committed
        CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemUV, &extUV));

        // -----------------------------
        // 4) Map to MipmappedArray
        // -----------------------------
        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mmY = {};
        mmY.offset = 0; // Always 0 for Resources
        mmY.arrayDesc.Width       = static_cast<unsigned int>(m_videoDecoderCreateInfo.ulWidth);
        mmY.arrayDesc.Height      = static_cast<unsigned int>(m_videoDecoderCreateInfo.ulHeight);
        mmY.arrayDesc.Depth       = 0;
        mmY.arrayDesc.NumChannels = 1;
        mmY.arrayDesc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
        mmY.arrayDesc.Flags       = 0;
        mmY.numLevels = 1;
        CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayY,
                                                           m_frameResources[i].cudaExtMemY, &mmY));

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mmUV = {};
        mmUV.offset = 0;
        mmUV.arrayDesc.Width       = static_cast<unsigned int>(m_videoDecoderCreateInfo.ulWidth  / 2);
        mmUV.arrayDesc.Height      = static_cast<unsigned int>(m_videoDecoderCreateInfo.ulHeight / 2);
        mmUV.arrayDesc.Depth       = 0;
        mmUV.arrayDesc.NumChannels = 2;
        mmUV.arrayDesc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
        mmUV.arrayDesc.Flags       = 0;
        mmUV.numLevels = 1;
        CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayUV,
                                                           m_frameResources[i].cudaExtMemUV, &mmUV));

        // Get Level 0
        CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayY,  m_frameResources[i].pMipmappedArrayY,  0));
        CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayUV, m_frameResources[i].pMipmappedArrayUV, 0));
    }

    DebugLog(L"Allocated D3D12/CUDA frame buffers (Committed/Dedicated).");
    return true;
}

int FrameDecoder::HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);

    self->m_nDecodePicCnt++; // Preserved from original

    cuvidCtxLock(self->m_ctxLock, 0);
    CUresult cr = cuvidDecodePicture(self->m_hDecoder, pPicParams);
    cuvidCtxUnlock(self->m_ctxLock, 0);

    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuvidDecodePicture failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        cuCtxPopCurrent(NULL);
        return 0;
    }
    cuCtxPopCurrent(NULL);
    return 1;
}

int FrameDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);

    CUVIDPROCPARAMS oVPP = {};
    oVPP.progressive_frame = pDispInfo->progressive_frame;
    oVPP.second_field = 0;
    oVPP.top_field_first = pDispInfo->top_field_first;
    oVPP.unpaired_field = (pDispInfo->progressive_frame == 1 || pDispInfo->repeat_first_field <= 1);

    CUdeviceptr pDecodedFrame = 0;
    unsigned int nDecodedPitch = 0;
    bool mapped = false;
    bool ok = true;

    // NVDEC公式のロックで排他（Map/Unmapは必ずロック下で）
    cuvidCtxLock(self->m_ctxLock, 0);

    CUresult cr = cuvidMapVideoFrame(self->m_hDecoder, pDispInfo->picture_index,
                                     &pDecodedFrame, &nDecodedPitch, &oVPP);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuvidMapVideoFrame failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        ok = false;
        goto cleanup; // まだ mapped=false なので unmap は呼ばれない
    }
    mapped = true;

    // ここからコピー。driver API を使用（既存方針を踏襲）。
    auto& fr = self->m_frameResources[pDispInfo->picture_index];
    if (!fr.pCudaArrayY || !fr.pCudaArrayUV) {
        DebugLog(L"HandlePictureDisplay: mapped CUDA arrays are null. Aborting.");
        ok = false;
        goto cleanup; // map 済みなので cleanup で unmap される
    }

    const size_t srcWidthBytes_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
    const size_t srcHeightRows_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight);
    const size_t srcWidthBytes_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
    const size_t srcHeightRows_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight / 2);

    CUDA_MEMCPY2D cpyY = {};
    cpyY.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyY.srcDevice     = pDecodedFrame;
    cpyY.srcPitch      = nDecodedPitch;
    cpyY.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpyY.dstArray      = fr.pCudaArrayY;
    cpyY.WidthInBytes  = srcWidthBytes_Y;
    cpyY.Height        = srcHeightRows_Y;

    cr = cuMemcpy2D(&cpyY);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuMemcpy2D(Y) failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        ok = false;
        goto cleanup;
    }

    const CUdeviceptr pSrcUV = pDecodedFrame + (size_t)srcHeightRows_Y * nDecodedPitch;
    CUDA_MEMCPY2D cpyUV = {};
    cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyUV.srcDevice     = pSrcUV;
    cpyUV.srcPitch      = nDecodedPitch;
    cpyUV.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpyUV.dstArray      = fr.pCudaArrayUV;
    cpyUV.WidthInBytes  = srcWidthBytes_UV;
    cpyUV.Height        = srcHeightRows_UV;

    cr = cuMemcpy2D(&cpyUV);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuMemcpy2D(UV) failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        ok = false;
        goto cleanup;
    }

    // 同期（必要最小限。必要なければ削っても良い）
    cr = cuCtxSynchronize();
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        std::wstring msg = L"cuCtxSynchronize failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        ok = false;
        goto cleanup;
    }

cleanup:
    if (mapped) {
        CUresult ur = cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame);
        if (ur != CUDA_SUCCESS) {
            const char* es = nullptr; cuGetErrorString(ur, &es);
            std::wstring msg = L"cuvidUnmapVideoFrame failed: ";
            if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
            DebugLog(msg);
            // ここでは復帰を続行（ログのみ）
        } else {
            // デバッグ用：map/unmap の対応を観測
            std::wstringstream wss;
            wss << L"[NVDEC] Unmapped picture_index=" << pDispInfo->picture_index
                << L" ptr=0x" << std::hex << (unsigned long long)pDecodedFrame;
            DebugLog(wss.str());
        }
    }

    cuvidCtxUnlock(self->m_ctxLock, 0);

    if (!ok) {
        cuCtxPopCurrent(NULL);
        return 0;
    }

    // ---- ここから下は既存の「描画キュー投入」ロジックを踏襲 ----
    ReadyGpuFrame readyFrame;
    readyFrame.hw_decoded_texture_Y  = self->m_frameResources[pDispInfo->picture_index].pTextureY;
    readyFrame.hw_decoded_texture_UV = self->m_frameResources[pDispInfo->picture_index].pTextureUV;
    readyFrame.timestamp             = pDispInfo->timestamp;
    readyFrame.originalFrameNumber   = self->m_nDecodedFrameCount++;
    readyFrame.id                    = readyFrame.originalFrameNumber;
    readyFrame.width                 = self->m_frameWidth;
    readyFrame.height                = self->m_frameHeight;

    {   // 送信フレーム番号への合流（既存通り）
        std::lock_guard<std::mutex> lk(self->m_tsMapMutex);
        auto it = self->m_tsToFrameNo.find(readyFrame.timestamp);
        if (it != self->m_tsToFrameNo.end()) {
            readyFrame.streamFrameNumber = it->second;
            self->m_lastStreamFrameNo = readyFrame.streamFrameNumber;
            self->m_tsToFrameNo.erase(it);
        } else {
            readyFrame.streamFrameNumber = self->m_lastStreamFrameNo + 1;
            self->m_lastStreamFrameNo = readyFrame.streamFrameNumber;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_readyGpuFrameQueueMutex);
        g_readyGpuFrameQueue.push_back(std::move(readyFrame));
    }
    g_readyGpuFrameQueueCV.notify_one();

    cuCtxPopCurrent(NULL);
    return 1;
}

UINT64 DecoderCount = 0;
void NvdecThread(int threadId) {
    DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"] started.");

    if (!g_frameDecoder) {
        DebugLog(L"NvdecThread: g_frameDecoder is not initialized!");
        return;
    }

    while (g_decode_worker_Running) { // Use the same global running flag
        H264Frame frame;
        if (g_h264FrameQueue.try_dequeue(frame)) {
            auto nvdec_start = std::chrono::high_resolution_clock::now();
            g_frameDecoder->Decode(frame);
            auto nvdec_end = std::chrono::high_resolution_clock::now();
            auto nvdec_us = std::chrono::duration_cast<std::chrono::microseconds>(nvdec_end - nvdec_start).count();
            if(DecoderCount++ % 200 == 0)DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"]: NVDEC decode Time: " + std::to_wstring(nvdec_us) + L" us");
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"] stopped.");
}

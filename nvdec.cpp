#include "nvdec.h"
#include "Globals.h"
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>
#include <windows.h> // ensure HANDLE is available
#include <cuda.h>    // we use Driver API external semaphore calls

// [NEW] Extern accessor provided by window.cpp
extern "C" HANDLE GetCopyFenceSharedHandleForCuda();

// [NEW] CUDA external semaphore for the D3D12 copy fence
static CUexternalSemaphore g_cuCopyFenceSemaphore = nullptr;
// [NEW] Per-frame fence value counter (monotonic). If there is another global frame counter, you may reuse it.
static std::atomic<uint64_t> g_cudaCopyFenceValue{0};

namespace my_nvtx_domains {
    struct nvdec {
        static constexpr char const* name = "NVDEC";
    };
}

#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

#include <atomic>
#include <chrono>
// Forward declaration from window.cpp
extern void ClearReorderState();

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

void FrameDecoder::releaseDecoderResources() {
    if (m_hDecoder) {
        cuvidDestroyDecoder(m_hDecoder);
        m_hDecoder = nullptr;
    }

    for (auto& resource : m_frameResources) {
        if (resource.copyStream) {
            cuStreamDestroy(resource.copyStream);
            resource.copyStream = nullptr;
        }
        if (resource.copyDone) {
            cuEventDestroy(resource.copyDone);
            resource.copyDone = nullptr;
        }

        resource.pCudaArrayY = nullptr;
        resource.pCudaArrayUV = nullptr;
        resource.pCudaArrayU = nullptr;
        resource.pCudaArrayV = nullptr;
        resource.pMipmappedArrayY = nullptr;
        resource.pMipmappedArrayUV = nullptr;
        resource.pMipmappedArrayU = nullptr;
        resource.pMipmappedArrayV = nullptr;

        if (resource.cudaExtMemY) {
            cuDestroyExternalMemory(resource.cudaExtMemY);
            resource.cudaExtMemY = nullptr;
        }
        if (resource.cudaExtMemUV) {
            cuDestroyExternalMemory(resource.cudaExtMemUV);
            resource.cudaExtMemUV = nullptr;
        }
        if (resource.cudaExtMemU) {
            cuDestroyExternalMemory(resource.cudaExtMemU);
            resource.cudaExtMemU = nullptr;
        }
        if (resource.cudaExtMemV) {
            cuDestroyExternalMemory(resource.cudaExtMemV);
            resource.cudaExtMemV = nullptr;
        }
        if(resource.sharedHandleY) {
            CloseHandle(resource.sharedHandleY);
            resource.sharedHandleY = nullptr;
        }
        if(resource.sharedHandleUV) {
            CloseHandle(resource.sharedHandleUV);
            resource.sharedHandleUV = nullptr;
        }
        if (resource.sharedHandleU) {
            CloseHandle(resource.sharedHandleU);
            resource.sharedHandleU = nullptr;
        }
        if (resource.sharedHandleV) {
            CloseHandle(resource.sharedHandleV);
            resource.sharedHandleV = nullptr;
        }
        resource.pTextureY.Reset();
        resource.pTextureUV.Reset();
        resource.pTextureU.Reset();
        resource.pTextureV.Reset();
    }
    m_frameResources.clear();
    DebugLog(L"Released decoder resources and frame buffers.");
}


FrameDecoder::~FrameDecoder() {
    cuCtxPushCurrent(m_cuContext);

    // Destroying the parser will flush any pending callbacks.
    // This must be done BEFORE releasing the resources that the callbacks use.
    if (m_hParser) {
        cuvidDestroyVideoParser(m_hParser);
        m_hParser = nullptr;
    }

    // Now it's safe to release resources.
    releaseDecoderResources();

// [NEW] Destroy external semaphore if created
if (g_cuCopyFenceSemaphore) {
    cuDestroyExternalSemaphore(g_cuCopyFenceSemaphore);
    g_cuCopyFenceSemaphore = nullptr;
}

    if (m_ctxLock) {
        cuvidCtxLockDestroy(m_ctxLock);
        m_ctxLock = nullptr;
    }

    cuCtxPopCurrent(NULL);
}


bool FrameDecoder::Init() {
    CUDA_CHECK(cuCtxPushCurrent(m_cuContext));

    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_HEVC; // HEVC 4:4:4 入力を扱う
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
    // Store frame timings for latency calculation
    {
        std::lock_guard<std::mutex> lk(m_tsTimingsMutex);
        m_tsToTimings[frame.timestamp] = { frame.rx_done_ms, frame.decode_start_ms };
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

namespace {
    // RAII helper for cuvidCtxLock
    class CudaCtxLocker {
    public:
        explicit CudaCtxLocker(CUvideoctxlock lock) : m_lock(lock) {
            if (m_lock) {
                cuvidCtxLock(m_lock, 0);
            }
        }
        ~CudaCtxLocker() {
            if (m_lock) {
                cuvidCtxUnlock(m_lock, 0);
            }
        }
        CudaCtxLocker(const CudaCtxLocker&) = delete;
        CudaCtxLocker& operator=(const CudaCtxLocker&) = delete;
    private:
        CUvideoctxlock m_lock;
    };

    // RAII helper for a mapped video frame
    class MappedVideoFrame {
    public:
        MappedVideoFrame(CUvideodecoder decoder, int picture_index, CUVIDPROCPARAMS* proc_params)
            : m_decoder(decoder) {
            m_result = cuvidMapVideoFrame(m_decoder, picture_index, &m_pDecodedFrame, &m_nDecodedPitch, proc_params);
        }
        ~MappedVideoFrame() {
            if (IsValid()) {
                CUresult ur = cuvidUnmapVideoFrame(m_decoder, m_pDecodedFrame);
                if (ur != CUDA_SUCCESS) {
                    const char* es = nullptr; cuGetErrorString(ur, &es);
                    std::wstring msg = L"cuvidUnmapVideoFrame failed in RAII destructor: ";
                    if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
                    DebugLog(msg);
                }
            }
        }
        bool IsValid() const { return m_result == CUDA_SUCCESS; }
        CUdeviceptr GetPointer() const { return m_pDecodedFrame; }
        unsigned int GetPitch() const { return m_nDecodedPitch; }
        CUresult GetMapResult() const { return m_result; }

        MappedVideoFrame(const MappedVideoFrame&) = delete;
        MappedVideoFrame& operator=(const MappedVideoFrame&) = delete;

    private:
        CUvideodecoder m_decoder = nullptr;
        CUdeviceptr m_pDecodedFrame = 0;
        unsigned int m_nDecodedPitch = 0;
        CUresult m_result = CUDA_ERROR_INVALID_VALUE;
    };
} // anonymous namespace

bool FrameDecoder::reconfigureDecoder(CUVIDEOFORMAT* pVideoFormat) {
    DebugLog(L"Reconfiguring decoder for new video format or resolution.");

    // Clean up existing decoder and resources
    releaseDecoderResources();

    // Create a new decoder with the new format
    bool ok = createDecoder(pVideoFormat);
    if (ok) {
        ClearReorderState(); // ★ 追加
    }
    return ok;
}

int FrameDecoder::HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);
    int result = 1;

    CudaCtxLocker ctxLocker(self->m_ctxLock);

    try {
        DebugLog(L"HandleVideoSequence: Codec: " + std::to_wstring(pVideoFormat->codec) +
            L", Coded Resolution: " + std::to_wstring(pVideoFormat->coded_width) + L"x" + std::to_wstring(pVideoFormat->coded_height) +
            L", Target Resolution: " + std::to_wstring(currentResolutionWidth.load()) + L"x" + std::to_wstring(currentResolutionHeight.load()));

        if (!self->m_hDecoder) {
            // First time initialization
            if (!self->createDecoder(pVideoFormat)) {
                DebugLog(L"HandleVideoSequence: Failed to create decoder for the first time.");
                result = 0; // Stop processing
            }
        } else {
            // Check if a reconfiguration is needed
            bool needsReconfig = false;

            // Reconfigure ONLY if the actual coded stream parameters change.
            if (pVideoFormat->coded_width  != self->m_videoDecoderCreateInfo.ulWidth  ||
                pVideoFormat->coded_height != self->m_videoDecoderCreateInfo.ulHeight ||
                pVideoFormat->codec        != self->m_videoDecoderCreateInfo.CodecType ||
                pVideoFormat->chroma_format!= self->m_videoDecoderCreateInfo.ChromaFormat) {
                DebugLog(L"HandleVideoSequence: Stream format changed. Reconfiguring.");
                needsReconfig = true;
            }

            // DO NOT reconfigure just because currentResolutionWidth/Height changed.
            // Target display size is a renderer concern; decoder operates at coded size.

            if (needsReconfig) {
                if (!self->reconfigureDecoder(pVideoFormat)) {
                    DebugLog(L"HandleVideoSequence: Failed to reconfigure decoder.");
                    result = 0; // Stop processing
                }
            }
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
    // int targetWidth = ::currentResolutionWidth.load();
    // int targetHeight = ::currentResolutionHeight.load();

    // If the target resolution hasn't been set yet, default to the stream's coded size.
    // if (targetWidth == 0 || targetHeight == 0) {
        // targetWidth = pVideoFormat->coded_width;
        // targetHeight = pVideoFormat->coded_height;
    // }

    // Set the class members to the actual coded size of the video stream.
    // This is the intrinsic resolution of the video, which is needed for correct
    // aspect ratio calculations in the renderer.
    m_frameWidth = pVideoFormat->coded_width;
    m_frameHeight = pVideoFormat->coded_height;

    // The rest of the function initializes the decoder with the *actual* stream
    // dimensions, but allocates buffers for the *target* dimensions.
    memset(&m_videoDecoderCreateInfo, 0, sizeof(m_videoDecoderCreateInfo));
    m_videoDecoderCreateInfo.CodecType = pVideoFormat->codec;
    // HEVC 4:4:4 settings
    if (pVideoFormat->codec == cudaVideoCodec_HEVC) {
        m_videoDecoderCreateInfo.ChromaFormat = cudaVideoChromaFormat_444;
    } else {
        m_videoDecoderCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    }
    m_videoDecoderCreateInfo.ulWidth = pVideoFormat->coded_width;
    m_videoDecoderCreateInfo.ulHeight = pVideoFormat->coded_height;
    m_videoDecoderCreateInfo.ulNumDecodeSurfaces = FrameDecoder::NUM_DECODE_SURFACES; // A pool of surfaces
    m_videoDecoderCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    m_videoDecoderCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    // Set target size to the actual coded size so decoder does NO scaling.
    // We will perform a crop manually during the cuMemcpy2D.
    m_videoDecoderCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
    m_videoDecoderCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    m_videoDecoderCreateInfo.ulNumOutputSurfaces = 12; // more headroom during resize/IDR

    // For HEVC 4:4:4, the output format depends on bit depth.
    // P010/P016 for 10/12-bit 4:2:0, but for 4:4:4, we receive planar data.
    // The underlying surface format can still be NV12 for 8-bit or P016 for 10/12-bit
    // as it mainly defines the raw decode buffer format. We will copy out
    // the planes manually.
    if (pVideoFormat->bit_depth_luma_minus8 > 0) {
        m_videoDecoderCreateInfo.OutputFormat = cudaVideoSurfaceFormat_P016;
    } else {
        m_videoDecoderCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    }

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

// [NEW] Import D3D12 fence as CUDA external semaphore (once)
if (!g_cuCopyFenceSemaphore) {
    HANDLE hFence = GetCopyFenceSharedHandleForCuda();
    if (hFence) {
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semDesc{};
        semDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
        semDesc.handle.win32.handle = hFence;
        CUresult cr = cuImportExternalSemaphore(&g_cuCopyFenceSemaphore, &semDesc);
        if (cr != CUDA_SUCCESS) {
            DebugLog(L"CUDA: cuImportExternalSemaphore(D3D12_FENCE) failed.");
            g_cuCopyFenceSemaphore = nullptr; // fallback to event-only path
        } else {
            DebugLog(L"CUDA: Imported D3D12 fence as external semaphore.");
        }
    } else {
        DebugLog(L"CUDA: Shared fence handle unavailable; staying on event-only path.");
    }
}

    for (UINT i = 0; i < m_videoDecoderCreateInfo.ulNumDecodeSurfaces; ++i) {
        // -----------------------------
        // 1) D3D12 Texture (Committed)
        // -----------------------------
        bool is444 = (m_videoDecoderCreateInfo.ChromaFormat == cudaVideoChromaFormat_444);
        bool isHighBitDepth = (m_videoDecoderCreateInfo.OutputFormat == cudaVideoSurfaceFormat_P016);

        DXGI_FORMAT format = isHighBitDepth ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
        CUarray_format cuFormat = isHighBitDepth ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8;

        // --- Y plane ---
        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Alignment = 0;
        texDesc.Width = m_videoDecoderCreateInfo.ulWidth;
        texDesc.Height = m_videoDecoderCreateInfo.ulHeight;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = format;
        texDesc.SampleDesc.Count = 1;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        auto create_texture = [&](const D3D12_RESOURCE_DESC& desc, Microsoft::WRL::ComPtr<ID3D12Resource>& tex) {
            return m_pD3D12Device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex));
        };

        HRESULT hr = create_texture(texDesc, m_frameResources[i].pTextureY);
        if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateCommittedResource(Y) failed."); return false; }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpY = {};
        UINT numRowsY = 0; UINT64 rowSizeInBytesY = 0; UINT64 totalBytesY = 0;
        m_pD3D12Device->GetCopyableFootprints(&texDesc, 0, 1, 0, &fpY, &numRowsY, &rowSizeInBytesY, &totalBytesY);
        m_frameResources[i].pitchY = fpY.Footprint.RowPitch;

        D3D12_RESOURCE_ALLOCATION_INFO allocInfoY = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDesc);
        UINT64 yImportSize = (allocInfoY.SizeInBytes > totalBytesY) ? allocInfoY.SizeInBytes : totalBytesY;

        hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureY.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleY);
        if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateSharedHandle(Y) failed."); return false; }

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC extY = { cudaExternalMemoryHandleTypeD3D12Resource, { m_frameResources[i].sharedHandleY }, yImportSize, cudaExternalMemoryDedicated };
        CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemY, &extY));

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mmY = {};
        mmY.arrayDesc = { texDesc.Width, texDesc.Height, 0, 1, cuFormat, 0 };
        mmY.numLevels = 1;
        CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayY, m_frameResources[i].cudaExtMemY, &mmY));
        CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayY, m_frameResources[i].pMipmappedArrayY, 0));

        if (is444) {
            // --- U plane ---
            D3D12_RESOURCE_DESC texDescU = texDesc;
            hr = create_texture(texDescU, m_frameResources[i].pTextureU);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateCommittedResource(U) failed."); return false; }
            m_pD3D12Device->GetCopyableFootprints(&texDescU, 0, 1, 0, &fpY, &numRowsY, &rowSizeInBytesY, &totalBytesY);
            m_frameResources[i].pitchU = fpY.Footprint.RowPitch;
            allocInfoY = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescU);
            UINT64 uImportSize = (allocInfoY.SizeInBytes > totalBytesY) ? allocInfoY.SizeInBytes : totalBytesY;
            hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureU.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleU);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateSharedHandle(U) failed."); return false; }
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC extU = { cudaExternalMemoryHandleTypeD3D12Resource, { m_frameResources[i].sharedHandleU }, uImportSize, cudaExternalMemoryDedicated };
            CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemU, &extU));
            CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayU, m_frameResources[i].cudaExtMemU, &mmY));
            CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayU, m_frameResources[i].pMipmappedArrayU, 0));

            // --- V plane ---
            D3D12_RESOURCE_DESC texDescV = texDesc;
            hr = create_texture(texDescV, m_frameResources[i].pTextureV);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateCommittedResource(V) failed."); return false; }
            m_pD3D12Device->GetCopyableFootprints(&texDescV, 0, 1, 0, &fpY, &numRowsY, &rowSizeInBytesY, &totalBytesY);
            m_frameResources[i].pitchV = fpY.Footprint.RowPitch;
            allocInfoY = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescV);
            UINT64 vImportSize = (allocInfoY.SizeInBytes > totalBytesY) ? allocInfoY.SizeInBytes : totalBytesY;
            hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureV.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleV);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateSharedHandle(V) failed."); return false; }
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC extV = { cudaExternalMemoryHandleTypeD3D12Resource, { m_frameResources[i].sharedHandleV }, vImportSize, cudaExternalMemoryDedicated };
            CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemV, &extV));
            CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayV, m_frameResources[i].cudaExtMemV, &mmY));
            CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayV, m_frameResources[i].pMipmappedArrayV, 0));
        } else {
            // --- UV plane (NV12) ---
            D3D12_RESOURCE_DESC texDescUV = texDesc;
            texDescUV.Width /= 2;
            texDescUV.Height /= 2;
            texDescUV.Format = isHighBitDepth ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;

            hr = create_texture(texDescUV, m_frameResources[i].pTextureUV);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateCommittedResource(UV) failed."); return false; }

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpUV = {};
            UINT numRowsUV = 0; UINT64 rowSizeInBytesUV = 0; UINT64 totalBytesUV = 0;
            m_pD3D12Device->GetCopyableFootprints(&texDescUV, 0, 1, 0, &fpUV, &numRowsUV, &rowSizeInBytesUV, &totalBytesUV);
            m_frameResources[i].pitchUV = fpUV.Footprint.RowPitch;

            D3D12_RESOURCE_ALLOCATION_INFO allocInfoUV = m_pD3D12Device->GetResourceAllocationInfo(0, 1, &texDescUV);
            UINT64 uvImportSize = (allocInfoUV.SizeInBytes > totalBytesUV) ? allocInfoUV.SizeInBytes : totalBytesUV;

            hr = m_pD3D12Device->CreateSharedHandle(m_frameResources[i].pTextureUV.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameResources[i].sharedHandleUV);
            if (FAILED(hr)) { DebugLog(L"allocateFrameBuffers: CreateSharedHandle(UV) failed."); return false; }

            CUDA_EXTERNAL_MEMORY_HANDLE_DESC extUV = { cudaExternalMemoryHandleTypeD3D12Resource, { m_frameResources[i].sharedHandleUV }, uvImportSize, cudaExternalMemoryDedicated };
            CUDA_CHECK(cuImportExternalMemory(&m_frameResources[i].cudaExtMemUV, &extUV));

            CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mmUV = {};
            mmUV.arrayDesc = { texDescUV.Width, texDescUV.Height, 0, 2, cuFormat, 0 };
            mmUV.numLevels = 1;
            CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_frameResources[i].pMipmappedArrayUV, m_frameResources[i].cudaExtMemUV, &mmUV));
            CUDA_CHECK(cuMipmappedArrayGetLevel(&m_frameResources[i].pCudaArrayUV, m_frameResources[i].pMipmappedArrayUV, 0));
        }

        // NEW: Create async copy resources
        CUDA_CHECK(cuStreamCreate(&m_frameResources[i].copyStream, CU_STREAM_NON_BLOCKING));
        CUDA_CHECK(cuEventCreate(&m_frameResources[i].copyDone, CU_EVENT_DISABLE_TIMING));
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

UINT64 HandlePictureDisplayCount = 0;
int FrameDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    // Before any heavy work (e.g., before cuvidMapVideoFrame / cuMemcpy2D)
    if (g_isSizing.load(std::memory_order_acquire)) {
        // Keep timing/logging elsewhere intact; we simply skip display queuing.
        // No comment removal; do not alter surrounding formatting.
        return 0; // or the existing success code path that skips queueing
    }

    FrameDecoder* const self = static_cast<FrameDecoder*>(pUserData);
    cuCtxPushCurrent(self->m_cuContext);
    nvtx3::scoped_range r_decode_copy("DecodeCopy");

    // RAII locker for the context lock. It will be unlocked automatically.
    CudaCtxLocker ctxLocker(self->m_ctxLock);

    CUVIDPROCPARAMS oVPP = {};
    oVPP.progressive_frame = pDispInfo->progressive_frame;
    oVPP.second_field = 0;
    oVPP.top_field_first = pDispInfo->top_field_first;
    oVPP.unpaired_field = (pDispInfo->progressive_frame == 1 || pDispInfo->repeat_first_field <= 1);

    // RAII mapper for the video frame. It will be unmapped automatically.
    nvtx3::scoped_range r_map("HandlePictureDisplay::cuvidMapVideoFrame");
    MappedVideoFrame mappedFrame(self->m_hDecoder, pDispInfo->picture_index, &oVPP);
    if (!mappedFrame.IsValid()) {
        const char* es = nullptr; cuGetErrorString(mappedFrame.GetMapResult(), &es);
        std::wstring msg = L"cuvidMapVideoFrame failed: ";
        if (es) { std::string s(es); msg += std::wstring(s.begin(), s.end()); }
        DebugLog(msg);
        cuCtxPopCurrent(NULL);
        return 0;
    }

    // From here, we can return on error and the destructors will handle cleanup.
    auto& fr = self->m_frameResources[pDispInfo->picture_index];
    bool is444 = (self->m_videoDecoderCreateInfo.ChromaFormat == cudaVideoChromaFormat_444);

    if (is444) {
        if (!fr.pCudaArrayY || !fr.pCudaArrayU || !fr.pCudaArrayV) {
            DebugLog(L"HandlePictureDisplay: mapped CUDA arrays for 4:4:4 are null. Aborting.");
            cuCtxPopCurrent(NULL);
            return 0;
        }
    } else {
        if (!fr.pCudaArrayY || !fr.pCudaArrayUV) {
            DebugLog(L"HandlePictureDisplay: mapped CUDA arrays for NV12 are null. Aborting.");
            cuCtxPopCurrent(NULL);
            return 0;
        }
    }

    const CUdeviceptr pDecodedFrame = mappedFrame.GetPointer();
    const unsigned int nDecodedPitch = mappedFrame.GetPitch();
    CUstream s = fr.copyStream;

    if (is444) {
        const size_t widthBytes = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
        const size_t heightRows = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight);
        const size_t planeSize = nDecodedPitch * heightRows;

        CUDA_MEMCPY2D cpy = {};
        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.srcPitch = nDecodedPitch;
        cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cpy.WidthInBytes = widthBytes;
        cpy.Height = heightRows;

        // Y plane
        cpy.srcDevice = pDecodedFrame;
        cpy.dstArray = fr.pCudaArrayY;
        CUDA_CHECK_CALLBACK(cuMemcpy2DAsync(&cpy, s));

        // U plane
        cpy.srcDevice = pDecodedFrame + planeSize;
        cpy.dstArray = fr.pCudaArrayU;
        CUDA_CHECK_CALLBACK(cuMemcpy2DAsync(&cpy, s));

        // V plane
        cpy.srcDevice = pDecodedFrame + planeSize * 2;
        cpy.dstArray = fr.pCudaArrayV;
        CUDA_CHECK_CALLBACK(cuMemcpy2DAsync(&cpy, s));
    } else {
        // NV12 path (existing logic)
        const size_t srcWidthBytes_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
        const size_t srcHeightRows_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight);
        const size_t srcWidthBytes_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
        const size_t srcHeightRows_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight / 2);

        CUDA_MEMCPY2D y = {};
        y.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        y.srcDevice     = pDecodedFrame;
        y.srcPitch      = nDecodedPitch;
        y.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        y.dstArray      = fr.pCudaArrayY;
        y.WidthInBytes  = srcWidthBytes_Y;
        y.Height        = srcHeightRows_Y;
        CUDA_CHECK_CALLBACK(cuMemcpy2DAsync(&y, s));

        CUDA_MEMCPY2D uv = {};
        uv.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        uv.srcDevice     = pDecodedFrame + (size_t)srcHeightRows_Y * nDecodedPitch;
        uv.srcPitch      = nDecodedPitch;
        uv.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        uv.dstArray      = fr.pCudaArrayUV;
        uv.WidthInBytes  = srcWidthBytes_UV;
        uv.Height        = srcHeightRows_UV;
        CUDA_CHECK_CALLBACK(cuMemcpy2DAsync(&uv, s));
    }

    // Record completion event for this frame
    CUevent frameCopyDone = nullptr;
    {
        nvtx3::scoped_range_in<my_nvtx_domains::nvdec> r("EventRecord(copyDone)");
        CUDA_CHECK_CALLBACK(cuEventCreate(&frameCopyDone, CU_EVENT_DISABLE_TIMING));
        CUDA_CHECK_CALLBACK(cuEventRecord(frameCopyDone, s));
    }

    // After CUDA copy to D3D12 textures is enqueued and will complete on 'copyStream'
    UINT64 fenceValue = 0;
    if (g_cuCopyFenceSemaphore) {
        const uint64_t fv = ++g_cudaCopyFenceValue;
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS sParams{};
        sParams.params.fence.value = fv;
        CUDA_CHECK_CALLBACK(cuSignalExternalSemaphoresAsync(&g_cuCopyFenceSemaphore, &sParams, 1, s));
        fenceValue = fv;
    }

    ReadyGpuFrame readyFrame;
    readyFrame.copyDone = frameCopyDone;
    readyFrame.fenceValue = fenceValue;
    readyFrame.hw_decoded_texture_Y = fr.pTextureY;
    if (is444) {
        readyFrame.hw_decoded_texture_U = fr.pTextureU;
        readyFrame.hw_decoded_texture_V = fr.pTextureV;
        readyFrame.hw_decoded_texture_UV = nullptr; // Explicitly null for 444
    } else {
        readyFrame.hw_decoded_texture_U = nullptr;
        readyFrame.hw_decoded_texture_V = nullptr;
        readyFrame.hw_decoded_texture_UV = fr.pTextureUV;
    }
    readyFrame.timestamp             = pDispInfo->timestamp;
    readyFrame.originalFrameNumber   = self->m_nDecodedFrameCount++;
    readyFrame.id                    = readyFrame.originalFrameNumber;
    readyFrame.width                 = self->m_frameWidth;
    readyFrame.height                = self->m_frameHeight;
    // Remove any cuCtxSynchronize() here. Do not add sleeps.
    // ---- end: NVDEC copy (async unified) ----

    // --- Client-side timing (steady clock) ---
    FrameTimings timings;
    {
        std::lock_guard<std::mutex> lk(self->m_tsTimingsMutex);
        auto it = self->m_tsToTimings.find(pDispInfo->timestamp);
        if (it != self->m_tsToTimings.end()) {
            timings = it->second;
            self->m_tsToTimings.erase(it);
        }
    }
    // Propagate rx_done_ms if known (0 otherwise)
    readyFrame.rx_done_ms    = timings.rx_done_ms;

    // Back-compat field will be finalized at RenderEnd; keep 0 here.
    readyFrame.client_fec_end_to_render_end_time_ms = 0;


    {
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
        nvtx3::scoped_range r("HandlePictureDisplay::EnqueueFrame");
        std::lock_guard<std::mutex> lock(g_readyGpuFrameQueueMutex);
        if(HandlePictureDisplayCount++ % 200 == 0) {
            std::wstringstream wss;
            wss << L"HandlePictureDisplay: Pushed Frame Enqueue Size " << g_readyGpuFrameQueue.size()
                << L" Queueing Duration Time " << readyFrame.client_fec_end_to_render_end_time_ms << " ms";
            DebugLog(wss.str());
        }
        // Start a cross-thread NVTX range for this streamFrameNumber
        {
            nvtxEventAttributes_t a{};
            a.version = NVTX_VERSION;
            a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            a.messageType = NVTX_MESSAGE_TYPE_ASCII;

            char label[128];
            sprintf(label, "Frame #%u", readyFrame.streamFrameNumber);
            a.message.ascii = label;

            readyFrame.nvtx_range_id = nvtxDomainRangeStartEx(g_frameDomain, &a);
        }
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
            nvtx3::scoped_range r("CUDA Decode");
            frame.decode_start_ms = SteadyNowMs();
            g_frameDecoder->Decode(frame);
            if(DecoderCount++ % 200 == 0)DebugLog(L"NvdecThread: Dequeue Size " + std::to_wstring(g_h264FrameQueue.size_approx()));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    DebugLog(L"NvdecThread [" + std::to_wstring(threadId) + L"] stopped.");
}

#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <mutex>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// NVDEC includes (assuming these are in the include path)
#include "nvcuvid.h"
#include "cuviddec.h"

#include "DebugLog.h"

// Forward declaration
struct EncodedFrame;
enum class PlaneLayout : uint32_t;

struct FrameTimings {
    uint64_t rx_done_ms = 0;
    uint64_t decode_start_ms = 0;
};

class FrameDecoder {
public:
    // Maximum number of decode surfaces the parser can expose. The actual number
    // of allocated surfaces is determined per-stream based on the sequence
    // parameters that the server provides (e.g., GOP length / reference frames).
    static const int NUM_DECODE_SURFACES = 64;
    
    FrameDecoder(CUcontext cuContext, ID3D12Device* pD3D12Device);
    ~FrameDecoder();

    bool Init();
    void Decode(const EncodedFrame& frame);

    // Static callbacks for CUVIDDECODECREATEINFO
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat);
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo);

private:
    CUvideoctxlock m_ctxLock = nullptr;
    std::mutex m_parseMutex; // Decode() を直列化する場合に使用

    std::unordered_map<uint64_t, uint32_t> m_tsToFrameNo;
    std::mutex m_tsMapMutex;
    uint32_t m_lastStreamFrameNo = 0; // フォールバック用カウンタ

    std::unordered_map<uint64_t, FrameTimings> m_tsToTimings;
    std::mutex m_tsTimingsMutex;

    bool createDecoder(CUVIDEOFORMAT* pVideoFormat);
    bool allocateFrameBuffers();
    void copyDecodedFrameToD3D12(CUVIDPARSERDISPINFO* pDispInfo);
    void releaseDecoderResources();
    bool reconfigureDecoder(CUVIDEOFORMAT* pVideoFormat);
    uint32_t determineDecodeSurfaceCount(const CUVIDEOFORMAT* pVideoFormat) const;

    CUcontext m_cuContext;
    ID3D12Device* m_pD3D12Device;
    CUvideoparser m_hParser = nullptr;
    CUvideodecoder m_hDecoder = nullptr;

    struct DecodedFrameResource {
        Microsoft::WRL::ComPtr<ID3D12Heap> pHeapY;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureY;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureU;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureV;
        CUexternalMemory cudaExtMemY;
        CUexternalMemory cudaExtMemU;
        CUexternalMemory cudaExtMemV;
        CUmipmappedArray pMipmappedArrayY;
        CUmipmappedArray pMipmappedArrayU;
        CUmipmappedArray pMipmappedArrayV;
        CUarray pCudaArrayY;
        CUarray pCudaArrayU;
        CUarray pCudaArrayV;
        HANDLE sharedHandleY;
        HANDLE sharedHandleU;
        HANDLE sharedHandleV;
        UINT pitchY;
        UINT pitchU;
        UINT pitchV;

        // NEW: for async copy
        CUstream copyStream = nullptr;
        CUevent copyDone = nullptr;
    };

    std::vector<DecodedFrameResource> m_frameResources;
    int m_nDecodedFrameCount = 0;
    int m_nDecodePicCnt = 0;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    PlaneLayout m_planeLayout;
    bool m_isHighBitDepth = false;
    uint32_t m_numDecodeSurfaces = NUM_DECODE_SURFACES;

    // Cropping parameters from video sequence
    int m_cropLeft = 0, m_cropTop = 0, m_cropRight = 0, m_cropBottom = 0;
    int m_displayWidth = 0, m_displayHeight = 0;

    // The CUVIDDECODECREATEINFO struct needs to be valid for the life of the decoder
    CUVIDDECODECREATEINFO m_videoDecoderCreateInfo = {};
};

// Main thread function for NVDEC
void NvdecThread(int threadId);

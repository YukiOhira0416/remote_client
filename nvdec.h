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
struct H264Frame;

class FrameDecoder {
public:
    static const int NUM_DECODE_SURFACES = 20;
    
    FrameDecoder(CUcontext cuContext, ID3D12Device* pD3D12Device);
    ~FrameDecoder();

    bool Init();
    void Decode(const H264Frame& frame);

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

    bool createDecoder(CUVIDEOFORMAT* pVideoFormat);
    bool allocateFrameBuffers();
    void copyDecodedFrameToD3D12(CUVIDPARSERDISPINFO* pDispInfo);

    CUcontext m_cuContext;
    ID3D12Device* m_pD3D12Device;
    CUvideoparser m_hParser = nullptr;
    CUvideodecoder m_hDecoder = nullptr;

    struct DecodedFrameResource {
        Microsoft::WRL::ComPtr<ID3D12Heap> pHeapY;
        Microsoft::WRL::ComPtr<ID3D12Heap> pHeapUV;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureY;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureUV;
        CUexternalMemory cudaExtMemY;
        CUexternalMemory cudaExtMemUV;
        CUmipmappedArray pMipmappedArrayY;
        CUmipmappedArray pMipmappedArrayUV;
        CUarray pCudaArrayY;
        CUarray pCudaArrayUV;
        HANDLE sharedHandleY;
        HANDLE sharedHandleUV;
        UINT pitchY;
        UINT pitchUV;
    };

    std::vector<DecodedFrameResource> m_frameResources;
    int m_nDecodedFrameCount = 0;
    int m_nDecodePicCnt = 0;
    int m_frameWidth = 0;
    int m_frameHeight = 0;

    // The CUVIDDECODECREATEINFO struct needs to be valid for the life of the decoder
    CUVIDDECODECREATEINFO m_videoDecoderCreateInfo = {};
};

// Main thread function for NVDEC
void NvdecThread(int threadId);

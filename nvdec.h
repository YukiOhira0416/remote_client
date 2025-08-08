#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <memory>
#include <string>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// NVDEC includes (assuming these are in the include path)
#include "nvcuvid.h"
#include "cuviddec.h"

#include "Globals.h"
#include "DebugLog.h"

class FrameDecoder {
public:
    FrameDecoder(CUcontext cuContext, ID3D12Device* pD3D12Device);
    ~FrameDecoder();

    bool Init();
    void Decode(const H264Frame& frame);

    // Static callbacks for CUVIDDECODECREATEINFO
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat);
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo);

private:
    bool createDecoder(CUVIDEOFORMAT* pVideoFormat);
    bool allocateFrameBuffers();
    void copyDecodedFrameToD3D12(CUVIDPARSERDISPINFO* pDispInfo);

    CUcontext m_cuContext;
    ID3D12Device* m_pD3D12Device;
    CUvideoparser m_hParser = nullptr;
    CUvideodecoder m_hDecoder = nullptr;

    struct DecodedFrameResource {
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureY;
        Microsoft::WRL::ComPtr<ID3D12Resource> pTextureUV;
        cudaExternalMemory_t cudaExtMemY;
        cudaExternalMemory_t cudaExtMemUV;
        CUdeviceptr mappedCudaPtrY;
        CUdeviceptr mappedCudaPtrUV;
        HANDLE sharedHandleY;
        HANDLE sharedHandleUV;
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

#pragma once

#ifdef USE_NVDEC

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <cstdint>

// Forward declarations for NVDEC types to avoid including the full header here if possible
// However, for simplicity in this context, we will include the main header.
#include "NvDec/nvcuvid.h"
#include "NvDec/cuviddec.h"

class NvdecDecoder {
public:
    NvdecDecoder();
    ~NvdecDecoder();

    // Initialize the decoder with the D3D12 device
    bool Init(ID3D12Device* pDevice, ID3D12CommandQueue* pCommandQueue);

    // Decode a single H.264 frame
    bool Decode(const uint8_t* pData, size_t nSize, uint64_t timestamp);

    // Get a completed frame from the decoder
    bool GetDecodedFrame(Microsoft::WRL::ComPtr<ID3D12Resource>& ppDecodedTexture, uint64_t& pTimestamp);

private:
    // Internal helper functions and member variables will go here
    void Cleanup();

    Microsoft::WRL::ComPtr<ID3D12Device> m_pD3D12Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_pCommandQueue;

    // CUDA/NVDEC specific members
    CUcontext m_cuContext = nullptr;
    CUvideodecoder m_cuDecoder = nullptr;

    // ... more members for managing frame buffers, etc.
};

#endif // USE_NVDEC

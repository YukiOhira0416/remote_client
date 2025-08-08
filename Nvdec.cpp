#include "Nvdec.h"
#include "DebugLog.h"

#ifdef USE_NVDEC

#include <cuda.h>

NvdecDecoder::NvdecDecoder() {
    // Constructor
}

NvdecDecoder::~NvdecDecoder() {
    Cleanup();
}

void NvdecDecoder::Cleanup() {
    // Implementation for releasing resources
    if (m_cuDecoder) {
        cuvidDestroyDecoder(m_cuDecoder);
        m_cuDecoder = nullptr;
    }

    if (m_cuContext) {
        cuCtxDestroy(m_cuContext);
        m_cuContext = nullptr;
    }
}

bool NvdecDecoder::Init(ID3D12Device* pDevice, ID3D12CommandQueue* pCommandQueue) {
    if (!pDevice || !pCommandQueue) {
        DebugLog(L"NvdecDecoder::Init: Invalid D3D12 device or command queue.");
        return false;
    }

    m_pD3D12Device = pDevice;
    m_pCommandQueue = pCommandQueue;

    // --- Initialize CUDA ---
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        DebugLog(L"NvdecDecoder::Init: cuInit failed. Error: " + std::to_wstring(cuResult));
        return false;
    }

    // Find a CUDA-capable device
    CUdevice cuDevice = 0; // Assuming the first device
    cuResult = cuDeviceGet(&cuDevice, 0);
    if (cuResult != CUDA_SUCCESS) {
        DebugLog(L"NvdecDecoder::Init: cuDeviceGet failed. Error: " + std::to_wstring(cuResult));
        return false;
    }

    // Create a CUDA context
    cuResult = cuCtxCreate(&m_cuContext, 0, cuDevice);
    if (cuResult != CUDA_SUCCESS) {
        DebugLog(L"NvdecDecoder::Init: cuCtxCreate failed. Error: " + std::to_wstring(cuResult));
        return false;
    }

    // --- Initialize the NVDEC decoder ---
    CUVIDDECODECREATEINFO decode_create_info = { 0 };
    decode_create_info.CodecType = cudaVideoCodec_H264;
    decode_create_info.ulWidth = 1920; // Placeholder, will be determined from bitstream
    decode_create_info.ulHeight = 1080; // Placeholder
    decode_create_info.ulNumDecodeSurfaces = 20; // A pool of surfaces for decoding
    decode_create_info.ChromaFormat = cudaVideoChromaFormat_420;
    decode_create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    // Important: Use D3D12 device
    decode_create_info.target_ext.d3d12.pD3D12Device = m_pD3D12Device.Get();
    decode_create_info.target_ext.d3d12.pD3D12CommandQueue = m_pCommandQueue.Get();


    cuResult = cuvidCreateDecoder(&m_cuDecoder, &decode_create_info);
    if (cuResult != CUDA_SUCCESS) {
        DebugLog(L"NvdecDecoder::Init: cuvidCreateDecoder failed. Error: " + std::to_wstring(cuResult));
        Cleanup();
        return false;
    }

    DebugLog(L"NvdecDecoder::Init: Successfully initialized.");
    return true;
}

bool NvdecDecoder::Decode(const uint8_t* pData, size_t nSize, uint64_t timestamp) {
    if (!m_cuDecoder) {
        return false;
    }

    // This is a simplified decode call. The actual implementation is more complex
    // and involves parsing and handling the bitstream correctly.
    // For now, this is a placeholder.

    CUVIDSOURCEDATAPACKET packet = { 0 };
    packet.payload = pData;
    packet.payload_size = nSize;
    packet.flags = CUVID_PKT_TIMESTAMP;
    packet.timestamp = timestamp;

    CUresult cuResult = cuvidDecodePicture(m_cuDecoder, &packet);
    if (cuResult != CUDA_SUCCESS) {
        DebugLog(L"NvdecDecoder::Decode: cuvidDecodePicture failed. Error: " + std::to_wstring(cuResult));
        return false;
    }

    return true;
}

bool NvdecDecoder::GetDecodedFrame(Microsoft::WRL::ComPtr<ID3D12Resource>& ppDecodedTexture, uint64_t& pTimestamp) {
    if (!m_cuDecoder) {
        return false;
    }

    // This function should map a decoded surface, get the D3D12 resource,
    // and return it. This is a complex process involving synchronization.
    // This is a placeholder for now.

    CUVIDPARSERDISPINFO dispInfo = { 0 };
    // The actual implementation would involve a callback mechanism (pfnDisplayPicture)
    // that queues up dispInfo structs. This function would then dequeue them.
    // For now, we can't implement this without the full callback structure.

    // Placeholder: returning false as we don't have a frame yet.
    return false;
}

#endif // USE_NVDEC

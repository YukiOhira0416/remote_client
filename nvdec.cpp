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

// CUarray から Y/UV 平面を 8bit グレースケールBMPに保存する
// width/height はBMP画像の寸法, pitch は元データの1行あたりバイト数
int SaveCUarrayPlaneAsBMP(CUarray cuArr, int width, int height, int pitch, const std::string& filename) {
    if (!cuArr || width <= 0 || height <= 0 || pitch <= 0) {
        DebugLog(L"SaveCUarrayPlaneAsBMP: Invalid parameters");
        return 0;
    }

    // ホスト側一時バッファ
    // pitch は CUDA Array からコピーする1行あたりのバイト数
    size_t srcBytesPerRow = static_cast<size_t>(pitch);
    std::vector<uint8_t> hostData(srcBytesPerRow * height, 0);

    // CUarray -> Host へ2Dコピー
    CUDA_MEMCPY2D cpy = {};
    cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    cpy.srcArray      = cuArr;
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    cpy.dstHost       = hostData.data();
    cpy.dstPitch      = srcBytesPerRow;           // ホスト側行ピッチ = pitch
    cpy.WidthInBytes  = srcBytesPerRow;           // 1行あたり pitch バイトをコピー
    cpy.Height        = static_cast<size_t>(height);

    // コールバック安全版：エラー時は0を返す
    {
        CUresult err = cuMemcpy2D(&cpy);
        if (err != CUDA_SUCCESS) {
            const char* errStr = nullptr;
            cuGetErrorString(err, &errStr);
            std::wstring msg = L"SaveCUarrayPlaneAsBMP: cuMemcpy2D failed: ";
            if (errStr) {
                std::string narrow(errStr);
                msg += std::wstring(narrow.begin(), narrow.end());
            }
            DebugLog(msg);
            return 0;
        }
    }

    // BMP 書き出し（8bit, 下から上, 4Bアライン）
    int bmpWidth  = width; // BMPの幅は引数 width
    int bmpHeight = height;
    int rowSize   = ((bmpWidth + 3) / 4) * 4;     // 4バイト境界
    int imageSize = rowSize * bmpHeight;

    BMPFileHeader fileHeader;
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 256*4 + imageSize;

    BMPInfoHeader infoHeader;
    infoHeader.biWidth     = bmpWidth;
    infoHeader.biHeight    = bmpHeight;
    infoHeader.biSizeImage = imageSize;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        DebugLog(L"SaveCUarrayPlaneAsBMP: Failed to open file");
        return 0;
    }

    // ヘッダ
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    // 256階調パレット
    for (int i = 0; i < 256; ++i) {
        uint8_t color[4] = { static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0 };
        file.write(reinterpret_cast<const char*>(color), 4);
    }

    // 画像データ（ボトムアップ）
    // hostData (pitchごと) から bmpWidth バイト分だけをコピーする
    std::vector<uint8_t> row(rowSize, 0);
    for (int y = bmpHeight - 1; y >= 0; --y) {
        std::memcpy(row.data(), hostData.data() + static_cast<size_t>(y) * srcBytesPerRow, std::min((size_t)bmpWidth, srcBytesPerRow));
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

    // --- Copy decoded frame to our D3D12 texture that is mapped as a CUDA array ---
    auto& fr = self->m_frameResources[pDispInfo->picture_index];

    if (!fr.pCudaArrayY || !fr.pCudaArrayUV) {
        DebugLog(L"HandlePictureDisplay: mapped CUDA arrays are null. Aborting.");
        CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));
        cuCtxPopCurrent(NULL);
        return 0;
    }

    const size_t srcWidthBytes_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth);
    const size_t srcHeightRows_Y  = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight);
    const size_t srcWidthBytes_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulWidth); // For NV12, UV plane width in bytes is same as Y
    const size_t srcHeightRows_UV = static_cast<size_t>(self->m_videoDecoderCreateInfo.ulHeight / 2);

    // Setup copy for Y plane
    CUDA_MEMCPY2D cpyY = {};
    cpyY.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyY.srcDevice = pDecodedFrame;
    cpyY.srcPitch = nDecodedPitch;
    cpyY.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpyY.dstArray = fr.pCudaArrayY;
    cpyY.WidthInBytes = srcWidthBytes_Y;
    cpyY.Height = srcHeightRows_Y;

    CUDA_CHECK_CALLBACK(cuMemcpy2D(&cpyY));

    // Setup copy for UV plane
    const CUdeviceptr pSrcUV = pDecodedFrame + (size_t)srcHeightRows_Y * nDecodedPitch;
    CUDA_MEMCPY2D cpyUV = {};
    cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpyUV.srcDevice = pSrcUV;
    cpyUV.srcPitch = nDecodedPitch;
    cpyUV.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpyUV.dstArray = fr.pCudaArrayUV;
    cpyUV.WidthInBytes = srcWidthBytes_UV;
    cpyUV.Height = srcHeightRows_UV;

    CUDA_CHECK_CALLBACK(cuMemcpy2D(&cpyUV));

    // Synchronize to ensure copy is complete before unmapping
    CUDA_CHECK_CALLBACK(cuCtxSynchronize());

    // Unmap the video frame
    CUDA_CHECK_CALLBACK(cuvidUnmapVideoFrame(self->m_hDecoder, pDecodedFrame));

    // Enqueue for rendering
    ReadyGpuFrame readyFrame;
    readyFrame.hw_decoded_texture_Y = self->m_frameResources[pDispInfo->picture_index].pTextureY;
    readyFrame.hw_decoded_texture_UV = self->m_frameResources[pDispInfo->picture_index].pTextureUV;
    readyFrame.timestamp = pDispInfo->timestamp;
    readyFrame.originalFrameNumber = self->m_nDecodedFrameCount++;
    readyFrame.id = readyFrame.originalFrameNumber;
    readyFrame.width = self->m_frameWidth;
    readyFrame.height = self->m_frameHeight;

    static int bmpCounter = 0;
    if (bmpCounter < 10) {
        std::string yFilename  = "frame_Y_"  + std::to_string(bmpCounter) + ".bmp";
        std::string uvFilename = "frame_UV_" + std::to_string(bmpCounter) + ".bmp";

        // Y面: width=ulWidth, height=ulHeight, pitch=ulWidth
        if (SaveCUarrayPlaneAsBMP(fr.pCudaArrayY,
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulWidth),
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulHeight),
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulWidth), // pitch
                                  yFilename) == 0) {
            DebugLog(L"HandlePictureDisplay: Failed to save Y plane BMP, aborting callback.");
            return 0;
        }

        // UV面(NV12): bmpWidth=ulWidth/2, height=ulHeight/2, pitch=ulWidth
        if (SaveCUarrayPlaneAsBMP(fr.pCudaArrayUV,
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulWidth / 2),
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulHeight / 2),
                                  static_cast<int>(self->m_videoDecoderCreateInfo.ulWidth), // pitch
                                  uvFilename) == 0) {
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

#pragma once

#include <cuda_runtime.h>

// Launches a kernel to copy a 2D plane of 8-bit data.
void LaunchCopyPlane_8bit(
    const unsigned char* pSrc,
    int nSrcPitch,
    unsigned char* pDst,
    int nDstPitch,
    int nWidthInBytes,
    int nHeight,
    cudaStream_t stream
);

// Launches a kernel to copy a 2D plane of 16-bit data.
// Width is specified in pixels.
void LaunchCopyPlane_16bit(
    const unsigned short* pSrc,
    int nSrcPitch,
    unsigned short* pDst,
    int nDstPitch,
    int nWidthInPixels,
    int nHeight,
    cudaStream_t stream
);

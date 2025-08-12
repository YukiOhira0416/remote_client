#include <cuda_runtime.h>

/*
 *  Simple CUDA kernel to copy a 2D plane of 8-bit data (e.g., Y plane of NV12).
 */
__global__ void CopyPlane_8bit(const unsigned char* pSrc, int nSrcPitch, unsigned char* pDst, int nDstPitch, int nWidth, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nWidth && y < nHeight)
    {
        const unsigned char* pSrc_line = pSrc + ((size_t)y * nSrcPitch);
        unsigned char* pDst_line = pDst + ((size_t)y * nDstPitch);
        pDst_line[x] = pSrc_line[x];
    }
}

/*
 *  Simple CUDA kernel to copy a 2D plane of 16-bit data (e.g., interleaved UV plane of NV12 to an R8G8 texture).
 *  Width is given in pixels (not bytes).
 */
__global__ void CopyPlane_16bit(const unsigned short* pSrc, int nSrcPitch, unsigned short* pDst, int nDstPitch, int nWidthInPixels, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nWidthInPixels && y < nHeight)
    {
        // Pitches are in bytes, so we cast pointers to char to do byte-level arithmetic
        const unsigned short* pSrc_line = (const unsigned short*)((const unsigned char*)pSrc + ((size_t)y * nSrcPitch));
        unsigned short* pDst_line = (unsigned short*)((unsigned char*)pDst + ((size_t)y * nDstPitch));
        pDst_line[x] = pSrc_line[x];
    }
}

// Host-side wrapper to launch the 8-bit copy kernel
void LaunchCopyPlane_8bit(
    const unsigned char* pSrc,
    int nSrcPitch,
    unsigned char* pDst,
    int nDstPitch,
    int nWidthInBytes,
    int nHeight,
    cudaStream_t stream)
{
    dim3 blockDim(32, 8);
    dim3 gridDim((nWidthInBytes + blockDim.x - 1) / blockDim.x, (nHeight + blockDim.y - 1) / blockDim.y);

    CopyPlane_8bit<<<gridDim, blockDim, 0, stream>>>(pSrc, nSrcPitch, pDst, nDstPitch, nWidthInBytes, nHeight);
}

// Host-side wrapper to launch the 16-bit copy kernel
void LaunchCopyPlane_16bit(
    const unsigned short* pSrc,
    int nSrcPitch,
    unsigned short* pDst,
    int nDstPitch,
    int nWidthInPixels,
    int nHeight,
    cudaStream_t stream)
{
    dim3 blockDim(32, 8);
    dim3 gridDim((nWidthInPixels + blockDim.x - 1) / blockDim.x, (nHeight + blockDim.y - 1) / blockDim.y);

    CopyPlane_16bit<<<gridDim, blockDim, 0, stream>>>(pSrc, nSrcPitch, pDst, nDstPitch, nWidthInPixels, nHeight);
}

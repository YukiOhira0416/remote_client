#include <windows.h>
#include <wrl/client.h>
#include <atomic>
#include <cuda.h> // For CUexternalSemaphore

extern std::atomic<bool> g_isSizing;

// For D3D12/CUDA interop
extern CUexternalSemaphore g_cudaCopyFenceSem;
UINT64 NextCopyFenceValue() noexcept;


bool InitWindow(HINSTANCE hInstance, int nCmdShow);
bool InitD3D();
void RenderFrame();
void SendWindowSize();
void CleanupD3DRenderResources();
void WaitForGpu();
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH);

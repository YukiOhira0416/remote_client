#include <windows.h>
#include <wrl/client.h>
#include <atomic>

extern std::atomic<bool> g_isSizing;

bool InitWindow(HINSTANCE hInstance, int nCmdShow);
bool InitD3D();
void RenderFrame();
void SendWindowSize();
void CleanupD3DRenderResources();
void WaitForGpu();
void FlushRenderPipeline();
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH);

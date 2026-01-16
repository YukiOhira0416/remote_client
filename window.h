#include <windows.h>
#include <wrl/client.h>
#include <atomic>

extern std::atomic<bool> g_isSizing;

bool InitWindow(HINSTANCE hInstance, int nCmdShow, HWND parentHwnd = NULL);
bool InitD3D();
void RenderFrame();
void SendWindowSize();
void CleanupD3DRenderResources();
void WaitForGpu();
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH);

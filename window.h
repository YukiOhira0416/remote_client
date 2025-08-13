#include <windows.h>
#include <wrl/client.h>

bool InitWindow(HINSTANCE hInstance, int nCmdShow);
bool InitD3D();
void RenderFrame();
void SendWindowSize();
void CleanupD3DRenderResources();
void WaitForGpu();

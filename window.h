#ifndef WINDOW_H
#define WINDOW_H

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
void ClearReorderState(bool keepLastFrame = false);
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH);
void NotifyResolutionChange(int cw, int ch);

#endif // WINDOW_H

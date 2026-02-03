#ifndef WINDOW_H
#define WINDOW_H

#include <windows.h>
#include <wrl/client.h>
#include <atomic>
#include <string>

extern std::atomic<bool> g_isSizing;

void SetOverlayMessage(const std::wstring& message);

bool InitWindow(HINSTANCE hInstance, int nCmdShow, HWND parentHwnd = NULL);
bool InitD3D();
void RenderFrame();
void SendWindowSize();
void CleanupD3DRenderResources();
void WaitForGpu();
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH);
void NotifyResolutionChange(int cw, int ch);

#endif // WINDOW_H

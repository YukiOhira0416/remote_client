#ifndef WINDOWS_SDK_COMPAT_H
#define WINDOWS_SDK_COMPAT_H

// Windows SDK 10.0.26100 以降の ksmedia.h が TIMECODE を再定義する場合があり、
// Visual Studio ビルド時に C2371 が発生することがあるため、
// 事前に __TIMECODE_DEFINED を定義して再定義を抑止する。
#include <windef.h>
#ifndef __TIMECODE_DEFINED
#define __TIMECODE_DEFINED
// strmif.h と同じ定義を明示しておく。
typedef struct tagTIMECODE {
    WORD  wFrameRate;
    WORD  wFrameFract;
    DWORD dwFrames;
    DWORD dwSeconds;
    DWORD dwMinutes;
    DWORD dwHours;
    DWORD dwFlags;
} TIMECODE;

typedef struct tagTIMECODE_SAMPLE {
    TIMECODE timecode;
    DWORD    dwUser;
    DWORD    dwFlags;
    DWORD    dwReserved1;
    DWORD    dwReserved2;
} TIMECODE_SAMPLE;
#endif // __TIMECODE_DEFINED

#endif // WINDOWS_SDK_COMPAT_H

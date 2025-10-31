#include <atomic>

extern std::atomic<int64_t>  g_TimeOffsetNs;
extern std::atomic<bool>     g_ClientStop;

void TimeSyncClientThread();
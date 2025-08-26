#pragma once
#ifdef ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>
  inline void nvtxPushU64(const char* name, uint64_t id, uint32_t color=0) {
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION; a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII; a.message.ascii = name;
    a.colorType = color ? NVTX_COLOR_ARGB : NVTX_COLOR_UNKNOWN; a.color = color;
    a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64; a.payload.ullValue = id;
    nvtxRangePushEx(&a);
  }
  inline void nvtxPop() { nvtxRangePop(); }
  inline void nvtxMarkU64(const char* name, uint64_t id, uint32_t color=0) {
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION; a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII; a.message.ascii = name;
    a.colorType = color ? NVTX_COLOR_ARGB : NVTX_COLOR_UNKNOWN; a.color = color;
    a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64; a.payload.ullValue = id;
    nvtxMarkEx(&a);
  }
#else
  inline void nvtxPushU64(const char*, uint64_t, uint32_t=0) {}
  inline void nvtxPop() {}
  inline void nvtxMarkU64(const char*, uint64_t, uint32_t=0) {}
#endif

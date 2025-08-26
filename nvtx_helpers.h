// nvtx_helpers.h
#pragma once
#include <nvtx3/nvtx3.hpp>

// Define domains and categories centrally
// Per brief: domain "RemoteClient", categories: Net=1, Decode=2, Render=3, Present=4
namespace NvtxDomains {
    // Using a function ensures the domain object is initialized before first use.
    inline nvtx3::domain& remote_client() {
        static nvtx3::domain d{"RemoteClient"};
        return d;
    }
}

enum class NvtxCategory : uint32_t {
    None = 0,
    Net = 1,
    Decode = 2,
    Render = 3,
    Present = 4
};

// A simple RAII wrapper for NVTX ranges that supports domains, categories, and colors.
struct NvtxRange {
    nvtx3::scoped_range_in range; // scoped_range_in is required for domains

    // Constructor with category and optional color.
    explicit NvtxRange(const char* name, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0)
      : range{ NvtxDomains::remote_client(),
               nvtx3::name{name},
               nvtx3::category{static_cast<uint32_t>(cat)},
               nvtx3::color{colorARGB},
               nvtx3::payload{(uint64_t)0} // Default payload, can be set later if needed.
              }
    {}

    // Constructor with a payload.
    explicit NvtxRange(const char* name, uint64_t payload, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0)
    : range{ NvtxDomains::remote_client(),
             nvtx3::name{name},
             nvtx3::category{static_cast<uint32_t>(cat)},
             nvtx3::color{colorARGB},
             nvtx3::payload{payload}
            }
    {}
};

// Standalone mark function for events.
inline void NvtxMark(const char* name, uint64_t payload, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0) {
    nvtx3::mark(NvtxDomains::remote_client(),
                nvtx3::name{name},
                nvtx3::category{static_cast<uint32_t>(cat)},
                nvtx3::color{colorARGB},
                nvtx3::payload{payload});
}

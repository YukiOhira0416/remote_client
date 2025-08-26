// nvtx_helpers.h
#pragma once
#include <nvtx3/nvtx3.hpp>

// Define a domain for the application.
// This is done by creating a type with a static `name` member.
struct remote_client_domain {
    static constexpr char const* name = "RemoteClient";
};

// Define categories for events.
enum class NvtxCategory : uint32_t {
    None = 0,
    Net = 1,
    Decode = 2,
    Render = 3,
    Present = 4
};

// A RAII wrapper for NVTX ranges within our application's domain.
struct NvtxRange {
    // The scoped_range_in must be templated with the domain type.
    nvtx3::scoped_range_in<remote_client_domain> range;

    // Constructor with category and optional color.
    // It constructs the `range` member by passing attribute objects.
    explicit NvtxRange(const char* name, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0)
      : range{
            nvtx3::message{name},
            nvtx3::category{static_cast<uint32_t>(cat)},
            nvtx3::color{colorARGB},
            nvtx3::payload{(uint64_t)0}
        }
    {}

    // Constructor with a payload.
    explicit NvtxRange(const char* name, uint64_t payload, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0)
    : range{
        nvtx3::message{name},
        nvtx3::category{static_cast<uint32_t>(cat)},
        nvtx3::color{colorARGB},
        nvtx3::payload{payload}
    }
    {}
};

// Standalone mark function for events within our application's domain.
inline void NvtxMark(const char* name, uint64_t payload, NvtxCategory cat = NvtxCategory::None, uint32_t colorARGB = 0) {
    // Use `mark_in` with our domain as a template parameter.
    nvtx3::mark_in<remote_client_domain>(
        nvtx3::message{name},
        nvtx3::category{static_cast<uint32_t>(cat)},
        nvtx3::color{colorARGB},
        nvtx3::payload{payload}
    );
}

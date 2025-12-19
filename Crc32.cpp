#include "Crc32.h"
#include <vector>
#include <numeric>

namespace {
    // Generate the CRC32 lookup table once.
    std::vector<uint32_t> GenerateCrc32Table() {
        std::vector<uint32_t> table(256);
        const uint32_t polynomial = 0xEDB88320; // Reversed polynomial
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (size_t j = 0; j < 8; ++j) {
                if (c & 1) {
                    c = polynomial ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            table[i] = c;
        }
        return table;
    }

    const std::vector<uint32_t> Crc32Table = GenerateCrc32Table();
}

uint32_t Crc32(const void* data, size_t length) {
    if (!data) {
        return 0;
    }
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF; // Initial value

    for (size_t i = 0; i < length; ++i) {
        crc = Crc32Table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    }

    return crc ^ 0xFFFFFFFF; // Final XOR value
}

#pragma once
#include <cstdint>
#include <cstddef>

// Calculates the CRC-32 (IEEE 802.3) checksum for the given data.
// This matches the CRC32 algorithm used in protocols like Ethernet and gzip.
// - Polynomial: 0x04C11DB7 (reflected)
// - Initial value: 0xFFFFFFFF
// - Final XOR value: 0xFFFFFFFF
// - Reflected input and output
uint32_t Crc32(const void* data, size_t length);

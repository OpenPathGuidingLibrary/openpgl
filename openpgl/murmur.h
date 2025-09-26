#pragma once

#include <cstdint>

SHARED_FUNCTION static inline void device_memcpy(void* dest, const void* src, size_t n) {
    // Cast the void pointers to char pointers to perform byte-wise copy
    char* cdest = (char*)dest;
    const char* csrc = (const char*)src;

    // Loop to copy n bytes from source to destination
    for (size_t i = 0; i < n; ++i) {
        cdest[i] = csrc[i];
    }
}

// murmur hash function from wikipedia

SHARED_FUNCTION static inline uint32_t murmur_32_scramble(uint32_t k) {
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    return k;
}

SHARED_FUNCTION static inline uint32_t murmur3_32(const uint8_t* key, size_t len, uint32_t seed = 0)
{
	uint32_t h = seed;
    uint32_t k;
    /* Read in groups of 4. */
    for (size_t i = len >> 2; i; i--) {
        // Here is a source of differing results across endiannesses.
        // A swap here has no effects on hash properties though.
        device_memcpy(&k, key, sizeof(uint32_t));
        key += sizeof(uint32_t);
        h ^= murmur_32_scramble(k);
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }
    /* Read the rest. */
    k = 0;
    for (size_t i = len & 3; i; i--) {
        k <<= 8;
        k |= key[i - 1];
    }
    // A swap is *not* necessary here because the preceding loop already
    // places the low bytes in the low places according to whatever endianness
    // we use. Swaps only apply when the memory is copied in a chunk.
    h ^= murmur_32_scramble(k);
    /* Finalize. */
	h ^= len;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

template<typename T>
SHARED_FUNCTION static inline uint32_t murmur3_32_t(const T* key, size_t len, uint32_t seed = 124124)
{
    return murmur3_32((const uint8_t*)key, sizeof(T)*len, seed);
}

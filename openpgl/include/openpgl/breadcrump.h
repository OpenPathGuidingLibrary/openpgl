#pragma once

#include <iostream>
#include <sstream>
#include <limits>

#include "stdint.h"
#include "assert.h"

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct Breadcrumb {
    uint32_t path = 0;
    uint32_t depth = 0;

    HOST_DEVICE Breadcrumb push(bool right) const {
        assert(depth < std::numeric_limits<uint32_t>::digits);

        Breadcrumb bc = *this;
        if (right)
            bc.path |= 1 << bc.depth;
        bc.depth++;
        return bc;
    }

    HOST_DEVICE bool getRight(int depth) const {
        return (path & (1 << depth)) != 0;
    }

    constexpr static uint32_t PATH = 0b10010;
    constexpr static uint32_t DEPTH = 5;

    HOST_DEVICE bool is() const {
        return path == PATH && depth == DEPTH;
    }

    HOST_DEVICE bool isParent() const {
        if (depth > DEPTH) return false;
        uint32_t mask = (1u << depth) - 1;
        return (mask & (path ^ PATH)) == 0;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "0b";
        for (int i = depth - 1; i >= 0; i--)
            ss << (getRight(i) ? '1' : '0');
        ss << ", " << depth;
        return ss.str();
    }

    HOST_DEVICE void print() const {
        printf("0b");
        for (int i = depth - 1; i >= 0; i--)
            printf("%c", getRight(i) ? '1' : '0');
        printf(", %i", depth);
    }
};

#undef HOST_DEVICE
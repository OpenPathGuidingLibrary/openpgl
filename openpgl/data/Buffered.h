#pragma once

#include <vector>
#include <cstddef>

namespace openpgl {

class BufferedWriter {
    std::vector<char> &buf;

public:
    BufferedWriter(std::vector<char> &buf) : buf(buf) {}

    void write(const char* s, size_t n) {
        buf.insert(buf.end(), s, s + n);
    }

    template<typename T>
    void write(const T *t, size_t n = 1) {
        write(reinterpret_cast<const char*>(t), n * sizeof(T));
    }
};

class BufferedReader {
    const std::vector<char> &buf;
    size_t offset = 0;

public:
    BufferedReader(const std::vector<char> &buf) : buf(buf) {}

    void read(char* s, size_t n) {
        auto first = buf.begin() + offset;
        offset += n;
        auto last = buf.begin() + offset;
        std::copy(first, last, s);
    }

    template<typename T>
    void read(T *t, size_t n = 1) {
        read(reinterpret_cast<char*>(t), n * sizeof(T));
    }

};
    
}

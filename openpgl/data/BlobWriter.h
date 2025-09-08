#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class BlobWriter {
  public:
    BlobWriter(const std::string& filename) : f(filename, std::ios::out | std::ios::binary) {}

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type operator<<(Type Element) {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T> void Write(T* Src, size_t Size) { f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T)); }

  private:
    std::ofstream f;
};

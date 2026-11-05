// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <iterator>

#include "SampleData.h"

namespace openpgl
{

template <typename Type>
struct ContainerInternal
{
    struct Iterator
    {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Type;
        using pointer = Type *;
        using reference = Type &;

        Iterator() : m_ptr(nullptr) {}
        Iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const
        {
            return *m_ptr;
        }
        pointer operator->()
        {
            return m_ptr;
        }

        // Prefix and postfix increment
        Iterator &operator++()
        {
            m_ptr++;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // Prefix and postfix decrement
        Iterator &operator--()
        {
            --m_ptr;
            return *this;
        }
        Iterator operator--(int)
        {
            Iterator tmp(*this);
            operator--();
            return tmp;
        }

        friend Iterator operator+(const Iterator lhs, const int idx)
        {
            return Iterator(lhs.m_ptr + idx);
        }
        friend Iterator operator+=(const Iterator lhs, const int idx)
        {
            return Iterator(lhs.m_ptr + idx);
        }
        friend Iterator operator-(const Iterator lhs, const int idx)
        {
            return Iterator(lhs.m_ptr - idx);
        }
        friend Iterator operator-=(const Iterator lhs, const int idx)
        {
            return Iterator(lhs.m_ptr - idx);
        }

        friend bool operator==(const Iterator &a, const Iterator &b)
        {
            return a.m_ptr == b.m_ptr;
        };
        friend bool operator!=(const Iterator &a, const Iterator &b)
        {
            return a.m_ptr != b.m_ptr;
        };

        Iterator::difference_type operator-(const Iterator &it) const
        {
            return m_ptr - it.m_ptr;
        }

        bool operator<(const Iterator &it) const
        {
            return m_ptr < it.m_ptr;
        }

        bool operator<=(const Iterator &it) const
        {
            return m_ptr <= it.m_ptr;
        }

        bool operator>(const Iterator &it) const
        {
            return m_ptr > it.m_ptr;
        }

        bool operator>=(const Iterator &it) const
        {
            return m_ptr >= it.m_ptr;
        }

        value_type &operator[](size_t idx)
        {
            return m_ptr[idx];
        }
        const value_type &operator[](size_t idx) const
        {
            return m_ptr[idx];
        }

       private:
        pointer m_ptr;
    };

    using iterator = Iterator;
    using value_type = Type;

    ContainerInternal() = default;

    ContainerInternal(const ContainerInternal &cont) = delete;

    ~ContainerInternal()
    {
        delete[] m_data;
        m_data = nullptr;
        m_size = 0;
        m_maxSize = 0;
    }

    Iterator begin()
    {
        OPENPGL_ASSERT(m_data);
        return Iterator(&m_data[0]);
    }
    Iterator end()
    {
        OPENPGL_ASSERT(m_data);
        return Iterator(m_data + m_size);
    }

    const Iterator begin() const
    {
        OPENPGL_ASSERT(m_data);
        return Iterator(&m_data[0]);
    }
    const Iterator end() const
    {
        return Iterator(m_data + m_size);
    }

    inline Type *data()
    {
        return m_data;
    }

    inline const Type *data() const
    {
        return m_data;
    }

    inline void reserve(size_t size)
    {
        if (size > m_maxSize)
        {
            delete[] m_data;
            m_data = new Type[size];
            m_maxSize = size;
        }
    }

    inline size_t capacity() const
    {
        return m_maxSize;
    };

    inline size_t size() const
    {
        return m_size;
    };

    inline void resize(size_t size)
    {
        reserve(size);
        m_size = size;
    }

    inline void clear()
    {
        m_size = 0;
    }

    Type &operator[](size_t idx)
    {
        OPENPGL_ASSERT(m_data);
        OPENPGL_ASSERT(m_size > 0);
        OPENPGL_ASSERT(idx < m_size);
        return m_data[idx];
    }
    const Type &operator[](size_t idx) const
    {
        OPENPGL_ASSERT(m_data);
        OPENPGL_ASSERT(m_size > 0);
        OPENPGL_ASSERT(idx < m_size);
        return m_data[idx];
    }

   private:
    Type *m_data{nullptr};
    size_t m_size{0};
    size_t m_maxSize{0};
};

}  // namespace openpgl
/**
 * Copyright 2015-2016 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP
#define FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP

#include <algorithm>

namespace LibFlatArray {

namespace detail {

template<long SIZE>
class soa_array_member_copy_helper
{
public:
    template<typename MEMBER>
    class inner_a
    {
    public:
        template<long INDEX, typename DUMMY=int>
        class copy_array_in
        {
        public:
            __host__
            __device__
            inline
            void operator()(
                const MEMBER *source,
                MEMBER *data,
                const std::size_t count,
                const std::size_t offset,
                const std::size_t stride)
            {
                copy_array_in<INDEX - 1, DUMMY>()(source, data, count, offset, stride);

                for (std::size_t i = offset; i < (offset + count); ++i) {
                    data[SIZE * (INDEX - 1) + i] = source[stride * (INDEX - 1) + i];
                }
            }
        };

        template<typename DUMMY>
        class copy_array_in<0, DUMMY>
        {
        public:
            __host__
            __device__
            inline
            void operator()(
                const MEMBER *source,
                MEMBER *data,
                const std::size_t count,
                const std::size_t offset,
                const std::size_t stride)
            {}
        };

        template<long INDEX, typename DUMMY=int>
        class copy_array_out
        {
        public:
            __host__
            __device__
            inline
            void operator()(
                MEMBER *target,
                const MEMBER *data,
                const std::size_t count,
                const std::size_t offset,
                const std::size_t stride)
            {
                copy_array_out<INDEX - 1, DUMMY>()(target, data, count, offset, stride);

                for (std::size_t i = offset; i < (offset + count); ++i) {
                    target[stride * (INDEX - 1) + i] = data[SIZE * (INDEX - 1) + i];
                }
            }
        };

        template<typename DUMMY>
        class copy_array_out<0, DUMMY>
        {
        public:
            __host__
            __device__
            inline
            void operator()(
                MEMBER *target,
                const MEMBER *data,
                const std::size_t count,
                const std::size_t offset,
                const std::size_t stride)
            {}
        };
    };

    template<typename CELL>
    class inner1
    {
    public:
        template<typename MEMBER>
        class inner2
        {
        public:

            class reference
            {
            public:
                __host__
                __device__
                inline
                explicit reference(char *data) :
                    data(data)
                {}

                __host__
                __device__
                inline
                MEMBER& operator[](const std::size_t offset)
                {
                    return *(reinterpret_cast<MEMBER*>(data) + offset * SIZE);
                }

            private:
                char *data;
            };

            class const_reference
            {
            public:
                __host__
                __device__
                inline
                explicit const_reference(const char *data) :
                    data(data)
                {}

                __host__
                __device__
                inline
                const MEMBER& operator[](const std::size_t offset)
                {
                    return *(reinterpret_cast<const MEMBER*>(data) + offset * SIZE);
                }

            private:
                const char *data;
            };

            template<long ARITY>
            class inner3
            {
            public:
                template<MEMBER (CELL:: *MEMBER_POINTER)[ARITY]>
                class inner4
                {
                public:
                    template<long INDEX, typename DUMMY = int>
                    class copy_in
                    {
                    public:
                        __host__
                        __device__
                        inline
                        void operator()(const CELL& cell, MEMBER *data)
                        {
                            copy_in<INDEX - 1>()(cell, data);
                            data[SIZE * (INDEX - 1)] = (cell.*MEMBER_POINTER)[INDEX - 1];
                        }
                    };

                    template<typename DUMMY>
                    class copy_in<0, DUMMY>
                    {
                    public:
                        __host__
                        __device__
                        inline
                        void operator()(const CELL& cell, MEMBER *data)
                        {}
                    };

                    template<long INDEX, typename DUMMY = int>
                    class copy_out
                    {
                    public:
                        __host__
                        __device__
                        inline
                        void operator()(CELL& cell, const MEMBER *data)
                        {
                            copy_out<INDEX - 1>()(cell, data);
                            (cell.*MEMBER_POINTER)[INDEX - 1] = data[SIZE * (INDEX - 1)];
                        }
                    };

                    template<typename DUMMY>
                    class copy_out<0, DUMMY>
                    {
                    public:
                        __host__
                        __device__
                        inline
                        void operator()(CELL& cell, const MEMBER *data)
                        {}
                    };
                };
            };
        };
    };

    /**
     * This is a workaround as the plain for loop will segfault with
     * g++ >= 4.9.0. It works with clang++ and icpc, though.
     */
    template<typename ELEMENT>
    __host__
    __device__
    static void copy(const ELEMENT *source, ELEMENT *target, std::size_t count)
    {
#ifdef __CUDACC__
        for (std::size_t i = 0; i < count; ++i) {
            target[i] = source[i];
        }
#else
        std::copy(source, source + count, target);
#endif
    }

};


}

}

#endif

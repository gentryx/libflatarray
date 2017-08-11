/**
 * Copyright 2015-2017 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP
#define FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP

// disable certain warnings from system headers when compiling with
// Microsoft Visual Studio:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4496 4514 )
#endif

#include <algorithm>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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

                for (std::size_t i = 0; i < count; ++i) {
// Overflow is fine on 32-bit systems as these won't instantiate such
// large arrays anyway:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4307 )
#endif

                    data[SIZE * (INDEX - 1) + i] = source[stride * (INDEX - 1) + offset + i];

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif
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
                const MEMBER* /* source */,
                MEMBER* /* data */,
                const std::size_t /* count */,
                const std::size_t /* offset */,
                const std::size_t /* stride */)
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

                for (std::size_t i = 0; i < count; ++i) {
// Overflow is fine on 32-bit systems as these won't instantiate such
// large arrays anyway:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4307 )
#endif

                    target[stride * (INDEX - 1) + offset + i] = data[SIZE * (INDEX - 1) + i];

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif
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
                MEMBER* /* target */,
                const MEMBER* /* data */,
                const std::size_t /* count */,
                const std::size_t /* offset */,
                const std::size_t /* stride */)
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
// Overflow is fine on 32-bit systems as these won't instantiate such
// large arrays anyway:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4307 )
#endif

                            data[SIZE * (INDEX - 1)] = (cell.*MEMBER_POINTER)[INDEX - 1];

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif
                        }
                    };

                    template<typename DUMMY>
                    class copy_in<0, DUMMY>
                    {
                    public:
                        __host__
                        __device__
                        inline
                        void operator()(const CELL& /* cell */, MEMBER* /* data */)
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
                        void operator()(CELL& /* cell */, const MEMBER* /* data */)
                        {}
                    };
                };
            };
        };
    };
};


}

}

#endif

/**
 * Copyright 2015 Andreas Schäfer
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP
#define FLAT_ARRAY_DETAIL_SOA_ARRAY_MEMBER_COPY_HELPER_HPP

namespace LibFlatArray {

namespace detail {

template<int SIZE>
class soa_array_member_copy_helper
{
public:
    template<typename MEMBER>
    class inner_a
    {
    public:
        template<int INDEX, typename DUMMY=int>
        class copy_array_in
        {
        public:
            inline
            void operator()(const MEMBER *source, MEMBER *data, const std::size_t count)
            {
                copy_array_in<INDEX - 1, DUMMY>()(source, data, count);
                std::copy(
                    source + count * (INDEX - 1),
                    source + count * (INDEX + 0),
                    data + SIZE * (INDEX - 1));
            }
        };

        template<typename DUMMY>
        class copy_array_in<0, DUMMY>
        {
        public:
            inline
            void operator()(const MEMBER *source, MEMBER *data, const std::size_t count)
            {}
        };

        template<int INDEX, typename DUMMY=int>
        class copy_array_out
        {
        public:
            inline
            void operator()(MEMBER *target, const MEMBER *data, const std::size_t count)
            {
                copy_array_out<INDEX - 1, DUMMY>()(target, data, count);
                std::copy(
                    data + SIZE * (INDEX - 1),
                    data + SIZE * (INDEX - 1) + count,
                    target + count * (INDEX - 1));
            }
        };

        template<typename DUMMY>
        class copy_array_out<0, DUMMY>
        {
        public:
            inline
            void operator()(MEMBER *target, const MEMBER *data, const std::size_t count)
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
                inline
                reference(char *data) :
                    data(data)
                {}

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
                inline
                const_reference(const char *data) :
                    data(data)
                {}

                inline
                const MEMBER& operator[](const std::size_t offset)
                {
                    return *(reinterpret_cast<const MEMBER*>(data) + offset * SIZE);
                }

            private:
                const char *data;
            };

            template<int ARITY>
            class inner3
            {
            public:
                template<MEMBER (CELL:: *MEMBER_POINTER)[ARITY]>
                class inner4
                {
                public:
                    template<int INDEX, typename DUMMY = int>
                    class copy_in
                    {
                    public:
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
                        inline
                        void operator()(const CELL& cell, MEMBER *data)
                        {}
                    };

                    template<int INDEX, typename DUMMY = int>
                    class copy_out
                    {
                    public:
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
                        inline
                        void operator()(CELL& cell, const MEMBER *data)
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

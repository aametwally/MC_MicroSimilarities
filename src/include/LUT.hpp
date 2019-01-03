//
// Created by asem on 04/12/18.
//

#ifndef MARKOVIAN_FEATURES_LUT_HPP
#define MARKOVIAN_FEATURES_LUT_HPP

#include "common.hpp"

template < typename K , typename V >
class LUT
{
    static_assert( std::numeric_limits<K>::is_integer , "T must be of integer type." );

    static constexpr auto LOWEST = std::numeric_limits<K>::lowest();
    static constexpr auto MAX = std::numeric_limits<K>::max();
    static constexpr size_t CAPACITY = MAX - LOWEST + 1;

private:
    template < class T , size_t N >
    constexpr LUT( std::array<T , N> arr )
            :_buffer( arr )
    {

    }

    template < typename ArrFn >
    constexpr LUT( ArrFn fn )
            :_buffer( fn())
    {

    }

public:
    LUT( const LUT & ) = default;

    inline const V &at( K key ) const
    {
        assert( key - LOWEST >= 0 );
        return _buffer.at( key - LOWEST );
    }

    inline V operator[]( K key ) const
    {
        assert( key - LOWEST >= 0 );
        return _buffer[key - LOWEST];
    }

    template < typename Function >
    static LUT<K , V> makeLUT( Function fn )
    {
        std::array<V , CAPACITY> index{};
        for ( auto i = LOWEST; i < MAX; ++i )
            index.at( i - LOWEST ) = fn( i );
        index.at( CAPACITY - 1 ) = fn( MAX );
        return LUT( [&]() { return index; } );
    }

    template < typename Fn >
    void inline forEach( Fn fn ) const
    {
        for ( K i = LOWEST; i < MAX; ++i )
            fn( i , _buffer.at( i - LOWEST ));
    }

private:
    const std::array<V , CAPACITY> _buffer;
};


#endif //MARKOVIAN_FEATURES_LUT_HPP

//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
#define MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

#include "common.hpp"
#include "aminoacids_grouping.hpp"
#include "Histogram.hpp"

using Order = int8_t;
using HistogramID = size_t;


namespace MC {
    template<typename AAGrouping>
    class MarkovChains
    {
    protected:
        static constexpr size_t StatesN = AAGrouping::StatesN;
        static constexpr std::array<char, StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
        static constexpr std::array<char, 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );

        using Histogram = buffers::Histogram<StatesN>;
        using Buffer =  typename Histogram::Buffer;
        using BufferIterator =  typename Buffer::iterator;
        using BufferConstIterator =  typename Buffer::const_iterator;

    public:

        static inline bool isReducedSequences( const std::vector<std::string> &sequences )
        {
            return std::all_of( sequences.cbegin(), sequences.cend(), isReducedSequence );
        }

        static inline bool isReducedSequence( std::string_view sequence )
        {
            for (auto c : sequence)
                if ( auto it = std::find( ReducedAlphabet.cbegin(), ReducedAlphabet.cend(), c ); it ==
                                                                                                 ReducedAlphabet.cend())
                    return false;
            return true;
        }

        virtual void train( const std::vector<std::string> &sequences ) = 0;


    protected:
        static constexpr inline HistogramID _char2ID( char a )
        {
            assert( a >= ReducedAlphabet.front());
            return HistogramID( a - ReducedAlphabet.front());
        }

        static constexpr inline char _id2Char( HistogramID id )
        {
            assert( id <= 128 );
            return char( id + ReducedAlphabet.front());
        }


    };

}
#endif //MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

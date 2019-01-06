//
// Created by asem on 05/01/19.
//

#ifndef MARKOVIAN_FEATURES_REGULARIZEDMC_HPP
#define MARKOVIAN_FEATURES_REGULARIZEDMC_HPP

#include "MC.hpp"

namespace MC
{

template < size_t States >
class RegularizedMC
{
    struct BooleanTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID , buffers::BooleanHistogram<States> >;
        using HeteroHistograms =  std::unordered_map<Order , IsoHistograms>;
        HeteroHistograms data;
    };

    struct StackedBooleanTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID , std::vector<buffers::BooleanHistogram<States>>>;
        using HeteroHistograms =  std::unordered_map<Order , IsoHistograms>;
        HeteroHistograms data;
    };

    struct AccumulativeTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID , buffers::Histogram<States> >;
        using HeteroHistograms =  std::unordered_map<Order , IsoHistograms>;
        HeteroHistograms data;
    };

public:
    explicit RegularizedMC( Order order )
            : _sensitiveMC( order )
    {

    }

    void addSequence( std::string_view sequence )
    {
        assert( LabeledEntry::isReducedSequence<States>( sequence ));
        _countInstance( sequence );
    }

    void addSequences( const std::vector<std::string> &sequences )
    {
        std::for_each( sequences.cbegin() , sequences.cend() ,
                       [this]( std::string_view s ) { addSequence( s ); } );
    }

protected:
    void _countInstance( std::string_view sequence )
    {

    }

private:
    MC<States> _sensitiveMC;
    StackedBooleanTransitionMatrices _stackedMatrices;
};

}

#endif //MARKOVIAN_FEATURES_REGULARIZEDMC_HPP

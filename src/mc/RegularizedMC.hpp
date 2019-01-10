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
    using Histogram = buffers::Histogram<States>;
    using TransitionMatrices =
    SparseTransitionMatrix2D<States , Histogram , Order , HistogramID>;

    struct StackedBooleanTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID , std::vector<buffers::BooleanHistogram<States>>>;
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

    void regularize()
    {
        TransitionMatrices regularizedHistograms;
        for ( auto &[order , isoHistograms] : _stackedMatrices.data )
        {
            for ( auto &[id , histogramsVector] : isoHistograms )
            {
                auto accumulative = BooleanHistogram<States>::accumulate( histogramsVector );
                regularizedHistograms.set( order , id , std::move( accumulative ));
            }
        }
        _stackedMatrices.data.clear();
        regularizedHistograms.forEach( []( Order , HistogramID , Histogram &histogram )
                                       {
                                           histogram.normalize();
                                       } );
        _sensitiveMC.setHistograms( std::move( regularizedHistograms ));
    }

    Order getOrder() const
    {
        return _sensitiveMC.getOrder();
    }

    inline double probability( std::string_view context , char current ) const
    {
        return _sensitiveMC.probability( context , current );
    }

    inline double transitionalPropensity( std::string_view context , char current ) const
    {
        return _sensitiveMC.transitionalPropensity( context , current );
    }

    inline double propensity( std::string_view query , double acc = 0 ) const
    {
        return _sensitiveMC.propensity( query , acc );
    }

    inline double probability( char a ) const
    {
        return _sensitiveMC.probability( a );
    }

protected:

    void _countInstance( std::string_view sequence )
    {
        MC<States> model( getOrder());
        model.train( sequence );
        auto histograms = std::move( model.stealHistograms());
        for ( auto &[order , isoHistograms] : histograms )
        {
            auto &stacked = _stackedMatrices.data[order];
            for ( auto &[id , histogram] : isoHistograms )
            {
                auto bHist = BooleanHistogram<States>::binarizeHistogram( std::move( histogram ));
                stacked[id].emplace_back( std::move( bHist ));
            }
        }
    }

private:
    MC<States> _sensitiveMC;
    StackedBooleanTransitionMatrices _stackedMatrices;
};

}

#endif //MARKOVIAN_FEATURES_REGULARIZEDMC_HPP

//
// Created by asem on 05/01/19.
//

#ifndef MARKOVIAN_FEATURES_REGULARIZEDMC_HPP
#define MARKOVIAN_FEATURES_REGULARIZEDMC_HPP

#include "MC.hpp"

namespace MC {

template<typename CoreMCModel, size_t States>
class RegularizedMC
{
    static_assert( CoreMCModel::t_States == States, "States mismatch!" );
public:
    static constexpr size_t t_States = States;
    using Histogram = buffers::Histogram<States>;
    using TransitionMatrices2D =
    SparseTransitionMatrix2D<States, Histogram, Order, HistogramID>;
    using BackboneProfile = std::unique_ptr<RegularizedMC>;
    using BackboneProfiles = std::map<std::string_view, std::unique_ptr<RegularizedMC>>;

private:
    struct StackedBooleanTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID, std::vector<buffers::BooleanHistogram<States>>>;
        using HeteroHistograms =  std::unordered_map<Order, IsoHistograms>;
        HeteroHistograms data;
    };

public:
    explicit RegularizedMC( Order order, double epsilon = AbstractMC<States>::TransitionMatrixEpsilon )
            : _sensitiveMC( order, epsilon ), _n( 0 )
    {}

    void addSequence( std::string_view sequence )
    {
        assert( LabeledEntry::isReducedSequence<States>( sequence ));
        _countInstance( sequence );
    }

    void addSequences( const std::vector<std::string> &sequences )
    {
        std::for_each( sequences.cbegin(), sequences.cend(),
                       [this]( std::string_view s ) { addSequence( s ); } );
    }

    void regularize()
    {
        TransitionMatrices2D regularizedHistograms;
        for (auto &[order, isoHistograms] : _stackedMatrices.data)
        {
            for (auto &[id, histogramsVector] : isoHistograms)
            {
                auto accumulative = BooleanHistogram<States>::accumulate( histogramsVector );
                auto standardDeviation = BooleanHistogram<States>::standardDeviationBernoulli( accumulative, _n );

                regularizedHistograms.set( order, id, std::move( accumulative ));
                _standardDeviations.set( order, id, std::move( standardDeviation ));
            }
        }
        _stackedMatrices.data.clear();
        regularizedHistograms.forEach( []( Order, HistogramID, Histogram &histogram ) {
            histogram.normalize();
        } );
        _sensitiveMC.setHistograms( std::move( regularizedHistograms ));
    }

    void train( const std::vector<std::string> &sequences )
    {
        addSequences( sequences );
        regularize();
    }

    void train( std::string_view sequence )
    {
        addSequence( sequence );
        regularize();
    }

    Order getOrder() const
    {
        return _sensitiveMC.getOrder();
    }

    inline double propensity( std::string_view query, double acc = 0 ) const
    {
        return _sensitiveMC.propensity( query, acc );
    }

    inline double transitionalPropensity( std::string_view context, char state ) const
    {
        return _sensitiveMC.transitionalPropensity( context, state );
    }

    inline auto histograms() const
    {
        return _sensitiveMC.histograms();
    }

    inline std::reference_wrapper<const TransitionMatrices2D> standardDeviations() const
    {
        return _standardDeviations;
    }

    inline auto histogram( Order order, HistogramID id ) const
    {
        return _sensitiveMC.histogram( order, id );
    }

    inline auto standardDeviations( Order order , HistogramID id ) const
    {
        return _standardDeviations( order , id );
    }

    inline auto stealHistograms()
    {
        return _sensitiveMC.stealHistograms();
    }

protected:

    void _countInstance( std::string_view sequence )
    {
        ++_n;
        CoreMCModel model( getOrder());
        model.train( sequence );
        auto histograms = std::move( model.stealHistograms());
        for (auto &[order, isoHistograms] : histograms)
        {
            auto &stacked = _stackedMatrices.data[order];
            for (auto &[id, histogram] : isoHistograms)
            {
                auto bHist = BooleanHistogram<States>::binarizeHistogram(
                        std::move( histogram ), model.getEpsilon());

                stacked[id].emplace_back( std::move( bHist ));
            }
        }
    }

private:
    CoreMCModel _sensitiveMC;
    TransitionMatrices2D _standardDeviations;
    StackedBooleanTransitionMatrices _stackedMatrices;
    size_t _n;
};

}

#endif //MARKOVIAN_FEATURES_REGULARIZEDMC_HPP

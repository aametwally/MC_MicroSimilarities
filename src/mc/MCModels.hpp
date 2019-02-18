//
// Created by asem on 04/02/19.
//

#ifndef MARKOVIAN_FEATURES_MCMODELS_HPP
#define MARKOVIAN_FEATURES_MCMODELS_HPP

#include "AbstractMC.hpp"

namespace MC {

template<size_t States>
class MC : public AbstractMC<States>
{
public:
    using Base = AbstractMC<States>;
    using Histogram = typename Base::Histogram;
    using TransitionMatrices2D  = typename Base::TransitionMatrices2D;

public:
    explicit MC(
            Order order,
            double epsilon = Base::TransitionMatrixEpsilon
    )
            : Base( order, epsilon )
    {
        assert( order >= 1 );
    }

    MC() = delete;

    virtual ~MC() = default;

    MC( const MC &mE ) = default;

    MC( MC &&mE ) noexcept
            : Base( std::move( mE ))
    {}

    MC &operator=( const MC &mE )
    {
        if ( this->getOrder() != mE.getOrder() || this->_epsilon != mE._epsilon )
            throw std::runtime_error( "Orders mismatch!" );
        this->_centroids = mE._centroids;
        return *this;
    }

    MC &operator=( MC &&mE ) noexcept
    {
        if ( this->getOrder() != mE.getOrder() || this->_epsilon != mE._epsilon )
            throw std::runtime_error( "Orders mismatch!" );
        this->_centroids.swap( mE._centroids );
        return *this;
    }

    using Base::probability;
    using Base::normalize;

    double probability(
            std::string_view context,
            char state
    ) const override
    {
        if ( context.size() > this->getOrder())
        {
            context.remove_prefix( context.size() - this->getOrder());
        }

        if ( LabeledEntry::isPolymorphicReducedSequence<States>( context ) ||
             LabeledEntry::isPolymorphicReducedAA( state ))
        {
            return 1;
        } else
        {
            auto distance = Order( context.length());
            auto id = Base::_sequence2ID( context );
            auto stateID = Base::_char2ID( state );
            if ( auto value = this->_centroids( distance, id, stateID ); value )
            {
                return value.value();
            } else return 0.0;
        }
    }

protected:
    virtual void _incrementInstance(
            std::string_view context,
            char state
    )
    {
        if ( !LabeledEntry::isPolymorphicReducedSequence<States>( context ) &&
             !LabeledEntry::isPolymorphicReducedAA( state ))
        {
            auto order = static_cast<Order>(context.size());
            auto id = Base::_sequence2ID( context );
            auto c = Base::_char2ID( state );
            this->_centroids.increment( order, id, this->_epsilon )( c );
        }
    }

    void _countInstance( std::string_view sequence ) override
    {
        for (auto a : sequence)
        {
            if ( !LabeledEntry::isPolymorphicReducedAA( a ))
            {
                auto c = Base::_char2ID( a );
                this->_centroids.increment( 0, 0, this->_epsilon )( c );
            }
        }

        for (Order distance = 1; distance <= this->getOrder(); ++distance)
            for (auto i = 0; i < sequence.size() - distance; ++i)
                _incrementInstance( sequence.substr( static_cast<size_t>(i), static_cast<size_t>(distance)),
                                    sequence[i + distance] );
    }
};

/**
 * @brief ZYMC
 * Zheng Yuan Approximated Higher-order Markov Chains
 * Paper: https://febs.onlinelibrary.wiley.com/doi/pdf/10.1016/S0014-5793%2899%2900506-2
 */
template<size_t States>
class ZYMC : public AbstractMC<States>
{
public:
    using Base = AbstractMC<States>;
    using Histogram = typename Base::Histogram;

    using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
    using HeteroHistograms = std::unordered_map<Order, IsoHistograms>;

public:
    explicit ZYMC(
            Order order,
            double epsilon = Base::TransitionMatrixEpsilon
    ) : Base( order, epsilon )
    {
        assert( order >= 1 );
    }

    virtual ~ZYMC() = default;

    static constexpr inline HistogramID lowerOrderID( HistogramID id )
    { return id / States; }

    inline double pairwiseProbability(
            char context,
            char state,
            Order distance
    ) const
    {
        auto c = Base::_char2ID( context );
        auto s = Base::_char2ID( state );
        auto value = this->_centroids( distance, c, s );
        return value.value_or( 0.0 );
    }

    double probability(
            std::string_view context,
            char state
    ) const override
    {
        if ( context.size() > this->getOrder())
        {
            context.remove_prefix( context.size() - this->getOrder());
        }

        if ( LabeledEntry::isPolymorphicReducedSequence<States>( context ) ||
             LabeledEntry::isPolymorphicReducedAA( state ))
        {
            return 1;
        } else
        {
            double p = 1.0;
            for (auto i = 0; i < context.size(); ++i)
            {
                auto distance = Order( context.size() - i );
                auto c = context[i];
                p *= pairwiseProbability( c, state, distance );
            }
            return p;
        }
    }

protected:
    virtual void _incrementInstance(
            std::string_view context,
            char state,
            Order distance
    )
    {
        assert( context.size() == 1 );

        if ( !LabeledEntry::isPolymorphicReducedSequence<States>( context ) &&
             !LabeledEntry::isPolymorphicReducedAA( state ))
        {
            auto c = Base::_char2ID( context.front());
            auto s = Base::_char2ID( state );
            this->_centroids.increment( distance, c, this->_epsilon )( s );
        }
    }

    void _countInstance( std::string_view sequence ) override
    {
        for (auto a : sequence)
        {

            if ( !LabeledEntry::isPolymorphicReducedAA( a ))
            {
                auto c = Base::_char2ID( a );
                this->_centroids.increment( 0, 0, this->_epsilon )( c );
            }
        }

        for (Order distance = 1; distance <= this->_order; ++distance)
            for (auto i = 0; i + distance < sequence.size(); ++i)
                _incrementInstance( sequence.substr( static_cast<size_t>(i), 1 ),
                                    sequence[i + distance], distance );
    }
};

template<size_t States>
class GappedMC : public ZYMC<States>
{
public:
    using Base = ZYMC<States>;
    using Histogram = typename Base::Histogram;

    using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
    using HeteroHistograms = std::unordered_map<Order, IsoHistograms>;

    explicit GappedMC(
            Order order,
            double epsilon = Base::TransitionMatrixEpsilon
    )
            : Base( order, epsilon )
    {}

    virtual ~GappedMC() = default;

    template<typename HistogramsCollection>
    explicit GappedMC(
            Order order,
            HistogramsCollection &&histograms,
            double epsilon = Base::TransitionMatrixEpsilon
    )
            : Base( order, std::forward<HistogramsCollection>( histograms ), epsilon )
    {}

    double probability(
            std::string_view context,
            char currentState
    ) const override
    {
        if ( context.size() > this->getOrder())
        {
            context.remove_prefix( context.size() - this->getOrder());
        }

        if ( LabeledEntry::isPolymorphicReducedSequence<States>( context ) ||
             LabeledEntry::isPolymorphicReducedAA( currentState ))
        {
            return 1;
        } else
        {
            double p = 1.0;
            constexpr float eps = std::numeric_limits<float>::epsilon() * 2;
            double min = 1;
            int iFrom = std::max( 0, int( context.length()) - this->_order );
            for (auto i = iFrom; i < context.size(); ++i)
            {
                auto distance = Order( context.size() - i );
                auto c = context[i];
                auto pBayes = Base::pairwiseProbability( c, currentState, distance );
                min = std::min( min, pBayes );
                p *= (pBayes + eps);
            }
            return p / (min + eps);
        }
    }
};

template<typename AAGrouping>
class PolymorphicMC : public MC<AAGrouping::StatesN>
{
    static constexpr auto States = AAGrouping::StatesN;
    using Base = AbstractMC<States>;

public:
    virtual ~PolymorphicMC() = default;

    double probability(
            std::string_view polymorphicContext,
            char polymorphicState
    ) const override
    {
        if ( polymorphicContext.size() > this->getOrder())
        {
            polymorphicContext.remove_prefix( polymorphicContext.size() - this->getOrder());
        }

        return LabeledEntry::polymorphicSummer<AAGrouping>(
                polymorphicContext, polymorphicState,
                [this](
                        std::string_view context,
                        char state
                ) {
                    auto distance = Order( context.length());
                    auto id = Base::_sequence2ID( context );
                    auto stateID = Base::_char2ID( state );
                    if ( auto value = this->_centroids( distance, id, stateID ); value )
                    {
                        return value.value();
                    } else return 0.0;
                } );
    }

protected:
    void _incrementInstance(
            std::string_view context,
            char state
    ) override
    {
        LabeledEntry::polymorphicApply<AAGrouping>(
                context, state,
                [this](
                        std::string_view context,
                        char state
                ) {
                    auto order = static_cast<Order>( context.size());
                    auto id = Base::_sequence2ID( context );
                    auto c = Base::_char2ID( state );
                    this->_centroids.increment( order, id, this->_epsilon )( c );
                } );
    }

    void _countInstance( std::string_view sequence ) override
    {
        for (auto a : sequence)
        {
            LabeledEntry::polymorphicApply<AAGrouping>( a, [this]( char state ) {
                auto c = Base::_char2ID( state );
                this->_centroids.increment( 0, 0, this->_epsilon )( c );
            } );
        }

        for (Order distance = 1; distance <= this->getOrder(); ++distance)
            for (auto i = 0; i < sequence.size() - distance; ++i)
                _incrementInstance( sequence.substr( static_cast<size_t>(i), static_cast<size_t>(distance)),
                                    sequence[i + distance] );
    }
};

template<typename AAGrouping>
class PolymorphicZYMC : public ZYMC<AAGrouping::StatesN>
{
    static constexpr auto States = AAGrouping::StatesN;
    using Base = AbstractMC<States>;

public:
    virtual ~PolymorphicZYMC() = default;

    double probability(
            std::string_view polymorphicContext,
            char polymorphicState
    ) const override
    {
        if ( polymorphicContext.size() > this->getOrder())
        {
            polymorphicContext.remove_prefix( polymorphicContext.size() - this->getOrder());
        }

        return LabeledEntry::polymorphicSummer<AAGrouping>(
                polymorphicContext, polymorphicState,
                [this](
                        std::string_view context,
                        char state
                ) {
                    double p = 1.0;
                    for (auto i = 0; i < context.size(); ++i)
                    {
                        auto distance = Order( context.size() - i );
                        auto c = context[i];
                        p *= this->pairwiseProbability( c, state, distance );
                    }
                    return p;
                } );
    }

    double probability( char a ) const override
    {
        return LabeledEntry::polymorphicSummer<AAGrouping>( a, [this]( char state ) {
            return this->_centroids( 0, 0, this->_char2ID( state )).value_or( 0 );
        } );
    }

protected:
    virtual void _incrementInstance(
            std::string_view context,
            char state,
            Order distance
    )
    {
        LabeledEntry::polymorphicApply<AAGrouping>(
                context, state,
                [this, distance](
                        std::string_view context,
                        char state
                ) {
                    assert( context.size() == 1 );
                    auto c = Base::_char2ID( context.front());
                    auto s = Base::_char2ID( state );
                    this->_centroids.increment( distance, c, this->_epsilon )( s );
                } );
    }

};

template<size_t States, typename CoreMCModel = MC<States> >
class RegularizedBinaryMC : public CoreMCModel
{
    static_assert( CoreMCModel::t_States == States, "States mismatch!" );
public:
    static constexpr size_t t_States = States;
    using Histogram = buffers::Histogram<States>;
    using TransitionMatrices2D =
    SparseTransitionMatrix2D<States, Histogram, Order, HistogramID>;
    using BackboneProfile = std::unique_ptr<RegularizedBinaryMC>;
    using BackboneProfiles = std::map<std::string_view, std::unique_ptr<RegularizedBinaryMC>>;

private:
    struct StackedBooleanTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID, std::vector<buffers::BooleanHistogram<States>>>;
        using HeteroHistograms =  std::unordered_map<Order, IsoHistograms>;
        HeteroHistograms data;
    };

public:
    explicit RegularizedBinaryMC(
            Order order,
            double epsilon = AbstractMC<States>::TransitionMatrixEpsilon
    )
            : CoreMCModel( order, epsilon )
    {}

    using AbstractMC<States>::normalize;

    void normalize( std::optional<size_t> minimumOccurrence ) override
    {
        assert( minimumOccurrence.value_or( 2 ) > 1 );
        TransitionMatrices2D regularizedHistograms;
        for (auto &[order, isoHistograms] : _stackedMatrices.data)
        {
            for (auto &[id, histogramsVector] : isoHistograms)
            {
                if ( histogramsVector.size() > minimumOccurrence.value_or( 0 ))
                {
                    std::vector<Histogram> normalized;

                    std::transform(
                            std::make_move_iterator( histogramsVector.begin()),
                            std::make_move_iterator( histogramsVector.end()),
                            std::back_inserter( normalized ), []( auto &&v ) { return v.normalize(); } );

                    auto mean =
                            Histogram::mean( normalized, this->_n );
                    auto standardDeviation =
                            Histogram::standardError( normalized, mean, this->_n );

                    this->_centroids.set( order, id, std::move( mean ));
                    this->_standardDeviations.set( order, id, std::move( standardDeviation ));
                }
            }
        }
        _stackedMatrices.data.clear();
    }

protected:
    virtual void _countInstance( std::string_view sequence )
    {
        ++this->_n; // A not funny mutable member.
        CoreMCModel model( this->getOrder(), this->getEpsilon());
        model.train( sequence );
        auto histograms = std::move( model.stealCentroids());
        for (auto &[order, isoHistograms] : histograms)
        {
            auto &stacked = _stackedMatrices.data[order];
            for (auto &[id, histogram] : isoHistograms)
            {
                auto bHist = BooleanHistogram<States>::binarizeHistogram(
                        std::move( histogram ), this->getEpsilon());

                stacked[id].emplace_back( std::move( bHist ));
            }
        }
    }

private:
    StackedBooleanTransitionMatrices _stackedMatrices;
};

template<size_t States, typename CoreMCModel = MC<States> >
class RegularizedVectorsMC : public CoreMCModel
{
    static_assert( CoreMCModel::t_States == States, "States mismatch!" );
public:
    static constexpr size_t t_States = States;
    using Histogram = buffers::Histogram<States>;
    using TransitionMatrices2D =
    SparseTransitionMatrix2D<States, Histogram, Order, HistogramID>;
    using BackboneProfile = std::unique_ptr<RegularizedVectorsMC>;
    using BackboneProfiles = std::map<std::string_view, std::unique_ptr<RegularizedVectorsMC>>;

private:
    struct StackedTransitionMatrices
    {
        using IsoHistograms = std::unordered_map<HistogramID, std::vector<Histogram>>;
        using HeteroHistograms =  std::unordered_map<Order, IsoHistograms>;
        HeteroHistograms data;
    };

public:
    explicit RegularizedVectorsMC(
            Order order,
            double epsilon = AbstractMC<States>::TransitionMatrixEpsilon
    )
            : CoreMCModel( order, epsilon )
    {}

    using AbstractMC<States>::normalize;

    void normalize( std::optional<size_t> minimumOccurrence ) override
    {
        assert( minimumOccurrence.value_or( 1 ) > 0 );
        TransitionMatrices2D regularizedHistograms;
        for (auto &&[order, isoHistograms] : _stackedMatrices.data)
        {
            for (auto &&[id, histogramsVector] : isoHistograms)
            {
                if ( histogramsVector.size() > minimumOccurrence.value_or( 0 ))
                {
                    std::vector<Histogram> normalized;

                    std::transform(
                            std::make_move_iterator( histogramsVector.begin()),
                            std::make_move_iterator( histogramsVector.end()),
                            std::back_inserter( normalized ), []( auto &&v ) { return v.normalize(); } );

                    auto centroid =
                            Histogram::mean( normalized, this->_n );
                    auto standardDeviation =
                            Histogram::standardError( normalized, centroid, this->_n );

                    this->_standardDeviations.set( order, id, std::move( standardDeviation ));
                    this->_centroids.set( order, id, std::move( centroid ));
                }
            }
        }
        _stackedMatrices.data.clear();
    }

protected:
    void _countInstance( std::string_view sequence ) override
    {
        ++this->_n; // mutable variable, annoying.
        CoreMCModel model( this->getOrder(), this->getEpsilon());
        model.train( sequence );

        auto histograms = std::move( model.stealCentroids());

        for (auto &[order, isoHistograms] : histograms)
        {
            auto &stacked = _stackedMatrices.data[order];
            for (auto &[id, histogram] : isoHistograms)
            {
                stacked[id].emplace_back( std::move( histogram ));
            }
        }
    }

private:
    StackedTransitionMatrices _stackedMatrices;
};

}
#endif //MARKOVIAN_FEATURES_MCMODELS_HPP

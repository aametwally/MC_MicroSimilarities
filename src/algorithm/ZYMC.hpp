//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_ZYMC_HPP
#define MARKOVIAN_FEATURES_ZYMC_HPP

#include "MarkovChains.hpp"

namespace MC {
/**
 * @brief ZYMC
 * Zheng Yuan Approximated Higher-order Markov Chains
 * Paper: https://febs.onlinelibrary.wiley.com/doi/pdf/10.1016/S0014-5793%2899%2900506-2
 */
    template<typename AAGrouping = AAGrouping_NOGROUPING20>
    class ZYMC : public MarkovChains<AAGrouping>
    {
    public:
        using MC = MarkovChains<AAGrouping>;
        using Histogram = typename MC::Histogram;

        using Selection = std::set<HistogramID>;
        using BackboneProfiles = std::map<std::string, ZYMC>;
    public:
        explicit ZYMC( Order order ) : _order( order )
        {
            assert( order >= 1 );
        }

        template<typename HistogramsCollection>
        explicit ZYMC( Order order, HistogramsCollection &&histograms ) :
                _order( order ), _pairwiseHistograms( std::forward<HistogramsCollection>( histograms ))
        {
            assert( order >= 1 );
        }

        explicit ZYMC( const std::vector<std::string> &sequences,
                       Order order ) : _order( order )
        {
            assert( order >= 1 );
            train( sequences );
        }

        ZYMC() = delete;

        ZYMC( const ZYMC &mE ) = default;

        ZYMC( ZYMC &&mE ) noexcept
                : _order( mE._order ), _pairwiseHistograms( std::move( mE._pairwiseHistograms ))
        {}

        ZYMC &operator=( const ZYMC &mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            _pairwiseHistograms = mE._pairwiseHistograms;
            return *this;
        }

        ZYMC &operator=( ZYMC &&mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            _pairwiseHistograms = std::move( mE._pairwiseHistograms );
            return *this;
        }

        void train( const std::vector<std::string> &sequences ) override
        {
            for (const auto &s : sequences)
                _countInstance( s );

            for (auto &[distance, pairs] : _pairwiseHistograms)
                for (auto &[context, histogram] : pairs)
                    histogram.normalize();
            _zeroOrderHistogram.normalize();
        }

        const Order order() const
        {
            return _order;
        }

        static constexpr inline HistogramID lowerOrderID( HistogramID id )
        { return id / MC::StatesN; }

        double pairwiseProbability( char a, char b, Order distance ) const
        {
            if ( auto dIt = _pairwiseHistograms.find( distance ); dIt != _pairwiseHistograms.cend())
            {
                auto &pairs = dIt->second;
                if ( auto contextIt = pairs.find( a ); contextIt != pairs.cend())
                {
                    auto &p = contextIt->second;
                    auto _b = MC::_char2ID( b );
                    return p.at(_b);
                } else return 0;
            } else return 0;
        }

        double approximateProbability( std::string_view context, char currentState ) const
        {
            double p = 1.0;
            for (auto i = 0; i < context.size(); ++i)
            {
                auto distance = Order( context.size() - i );
                auto c = context[i];
                p *= pairwiseProbability( c, currentState, distance );
            }
            return p;
        }


        double propensity( std::string_view query ) const
        {
            double acc = 0;
            acc += std::log( _zeroOrderHistogram.at( MC::_char2ID( query.front())));
            for (Order distance = 1; distance < _order && distance < query.size(); ++distance)
            {
                double p = approximateProbability( query.substr( 0, distance ), query[distance] );
                acc += std::log( p );
            }
            for (auto i = 0; i < query.size() - _order - 1; ++i)
            {
                double p = approximateProbability( query.substr( i, _order ), query[i + _order] );
                acc += std::log( p );
            }
            return acc;
        }

    protected:
        void _incrementInstance( char context,
                                 char currentState,
                                 Order distance )
        {
            auto c = MC::_char2ID( currentState );
            _pairwiseHistograms[distance][context].increment( c );
        }

        void _countInstance( std::string_view sequence )
        {
            for (auto a : sequence)
            {
                auto c = MC::_char2ID( a );
                _zeroOrderHistogram.increment( c );
            }

            for (Order distance = 1; distance <= _order; ++distance)
                for (auto i = 0; i < sequence.size() - distance; ++i)
                    _incrementInstance( sequence[i], sequence[i + distance], distance );
        }

    private:
        using PairwiseHistograms = std::unordered_map<char, Histogram>;
        using DistantPairwiseHistograms = std::unordered_map<Order, PairwiseHistograms>;
        const Order _order;
        DistantPairwiseHistograms _pairwiseHistograms;
        Histogram _zeroOrderHistogram;
    };
}
#endif //MARKOVIAN_FEATURES_ZYMC_HPP

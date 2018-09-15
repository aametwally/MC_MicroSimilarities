//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_ZYMC_HPP
#define MARKOVIAN_FEATURES_ZYMC_HPP

#include "AbstractMC.hpp"

namespace MC {
/**
 * @brief ZYMC
 * Zheng Yuan Approximated Higher-order Markov Chains
 * Paper: https://febs.onlinelibrary.wiley.com/doi/pdf/10.1016/S0014-5793%2899%2900506-2
 */
    template<typename AAGrouping >
    class ZYMC : public AbstractMC<AAGrouping>
    {
    public:
        using Base = AbstractMC<AAGrouping>;
        using Histogram = typename Base::Histogram;

        using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
        using HeteroHistograms = std::unordered_map<Order, IsoHistograms>;

    public:
        explicit ZYMC( Order order ) : _order( order )
        {
            assert( order >= 1 );
        }

        template<typename HistogramsCollection>
        explicit ZYMC( Order order, HistogramsCollection &&histograms ) :
                _order( order ), Base( std::forward<HistogramsCollection>( histograms ))
        {
            assert( order >= 1 );
        }

        explicit ZYMC( const std::vector<std::string> &sequences,
                       Order order ) : _order( order )
        {
            assert( order >= 1 );
            this->train( sequences );
        }

        ZYMC() = delete;

        ZYMC( const ZYMC &mE ) = default;

        ZYMC( ZYMC &&mE ) noexcept
                : _order( mE._order ), Base( std::move( mE._histograms ))
        {}

        ZYMC &operator=( const ZYMC &mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = mE._histograms;
            return *this;
        }

        ZYMC &operator=( ZYMC &&mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = std::move( mE._histograms );
            return *this;
        }


        const Order order() const
        {
            return _order;
        }

        static constexpr inline HistogramID lowerOrderID( HistogramID id )
        { return id / Base::StatesN; }

        double pairwiseProbability( char a, char b, Order distance ) const
        {
            if ( auto dIt = this->_histograms.find( distance ); dIt != this->_histograms.cend())
            {
                auto &pairs = dIt->second;
                auto _a = Base::_char2ID( a );
                if ( auto contextIt = pairs.find( _a ); contextIt != pairs.cend())
                {
                    auto &p = contextIt->second;
                    auto _b = Base::_char2ID( b );
                    return p.at( _b );
                } else return 0;
            } else return 0;
        }

        double probability( std::string_view context, char currentState ) const override
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

        double propensity( std::string_view query ) const override
        {
            double acc = 0;
            acc += std::log( this->_histograms.at( 0 ).at( 0 ).at( Base::_char2ID( query.front())));
            for (Order distance = 1; distance < _order && distance < query.size(); ++distance)
            {
                double p = probability( query.substr( 0, distance ), query[distance] );
                acc += std::log( p );
            }
            for (auto i = 0; i < int64_t(query.size()) - _order - 1; ++i)
            {
                double p = probability( query.substr( i, _order ), query[i + _order] );
                acc += std::log( p );
            }
            return acc ;
        }


    protected:
        void _incrementInstance( char context,
                                 char currentState,
                                 Order distance )
        {
            auto c = Base::_char2ID( context );
            auto s = Base::_char2ID( currentState );
            this->_histograms[distance][c].increment( s );
        }

        void _countInstance( std::string_view sequence ) override
        {
            for (auto a : sequence)
            {
                auto c = Base::_char2ID( a );
                this->_histograms[0][0].increment( c );
            }

            for (Order distance = 1; distance <= _order; ++distance)
                for (auto i = 0; i < sequence.size() - distance; ++i)
                    _incrementInstance( sequence[i], sequence[i + distance], distance );
        }

    protected:
        Order _order;
    };
}
#endif //MARKOVIAN_FEATURES_ZYMC_HPP

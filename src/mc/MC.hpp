//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MC_HPP
#define MARKOVIAN_FEATURES_MC_HPP

#include "AbstractMC.hpp"

namespace MC {

    template<typename AAGrouping = AAGrouping_NOGROUPING20>
    class MC : public AbstractMC<AAGrouping>
    {

    public:
        using Base = AbstractMC<AAGrouping>;
        using Histogram = typename Base::Histogram;
        using HeteroHistograms  = typename Base::HeteroHistograms ;

    public:
        explicit MC( Order order ) : _order( order )
        {
            assert( order >= 1 );
        }

        template<typename HistogramsCollection>
        explicit MC( Order order, HistogramsCollection &&histograms ) :
                _order( order ), Base( std::forward<HistogramsCollection>( histograms ))
        {
            assert( order >= 1 );
        }

        explicit MC( const std::vector<std::string> &sequences,
                     Order order ) : _order( order )
        {
            assert( order >= 1 );
            this->train( sequences );
        }

        explicit MC( const std::vector<std::string_view> &sequences,
                     Order order ) : _order( order )
        {
            assert( order >= 1 );
            this->train( sequences );
        }

        MC() = delete;

        MC( const MC &mE ) = default;

        MC( MC &&mE ) noexcept
                : _order( mE._order ), Base( std::move( mE._histograms ))
        {}

        MC &operator=( const MC &mE )
        {
            assert( _order == mE._order );
            if ( _order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = mE._histograms;
            return *this;
        }

        MC &operator=( MC &&mE )
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


        double probability( std::string_view context, char currentState ) const override
        {
            auto distance = Order( context.size() );
            auto id = Base::_sequence2ID( context );
            auto stateID = Base::_char2ID( currentState );
            if ( auto isoHistogramsIt = this->_histograms.find( distance ); isoHistogramsIt != this->_histograms.cend())
            {
                auto &isoHistograms = isoHistogramsIt->second;
                if ( auto histogramIt = isoHistograms.find( id ); histogramIt != isoHistograms.cend())
                {
                    auto &histogram = histogramIt->second;
                    return histogram[stateID];
                } else return 0;
            } else return 0;
        }


        double propensity( std::string_view query ) const override
        {
            double acc = 0;
            acc += std::log( this->_histograms.at(0).at(0).at( Base::_char2ID( query.front())));
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
            return acc;
        }

    protected:
        void _incrementInstance( std::string_view context,
                                 char currentState )
        {
            auto order = context.size();
            auto id = Base::_sequence2ID( context );
            auto c = Base::_char2ID( currentState );
            this->_histograms[order][id].increment( c );
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
                    _incrementInstance( sequence.substr( i, distance ), sequence[i + distance] );
        }

    private:
        Order _order;
    };

}
#endif //MARKOVIAN_FEATURES_MC_HPP

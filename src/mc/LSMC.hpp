//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_LSMC_HPP
#define MARKOVIAN_FEATURES_LSMC_HPP

#include "ZYMC.hpp"

namespace MC {

    template<typename AAGrouping>
    class LSMC : public ZYMC<AAGrouping>
    {
    public:
        using Base = AbstractMC<AAGrouping>;
        using Histogram = typename Base::Histogram;

        using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
        using HeteroHistograms = std::unordered_map<Order, IsoHistograms>;

        explicit LSMC( Order order ) : ZYMC<AAGrouping>( order )
        {}

        template<typename HistogramsCollection>
        explicit LSMC( Order order, HistogramsCollection &&histograms ) :
                ZYMC<AAGrouping>( order, std::forward<HistogramsCollection>( histograms ))
        {}

        explicit LSMC( const std::vector<std::string> &sequences,
                       Order order ) : ZYMC<AAGrouping>( sequences, order )
        {}

        LSMC() = delete;

        LSMC( const LSMC &mE ) = default;

        LSMC( LSMC &&mE ) noexcept
                : ZYMC<AAGrouping>( std::move( mE ))
        {}

        LSMC &operator=( const LSMC &mE )
        {
            assert( this->_order == mE._order );
            if ( this->_order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = mE._histograms;
            return *this;
        }

        LSMC &operator=( LSMC &&mE )
        {
            assert( this->_order == mE._order );
            if ( this->_order != mE._order )
                throw std::runtime_error( "Orders mismatch!" );
            this->_histograms = std::move( mE._histograms );
            return *this;
        }

        double propensity( std::string_view query ) const override
        {
            std::vector<double> propensities;
            propensities.push_back( std::log( this->_histograms.at( 0 ).at( 0 ).at( Base::_char2ID( query.front()))));
            for (Order distance = 1; distance < this->_order && distance < query.size(); ++distance)
            {
                double p = this->probability( query.substr( 0, distance ), query[distance] );
                propensities.push_back( std::log( p ));;
            }

            for (auto i = 0; i < int64_t( query.size()) - this->_order - 1; ++i)
            {
                double p = this->probability( query.substr( i, this->_order ), query[i + this->_order] );
                propensities.push_back( std::log( p ));;
            }
//            for( auto p : propensities )
//                fmt::print("[{}]\n", io::join(io::asStringsVector( propensities ), ","));

            auto correlation = correlate( hannWindow( std::min( size_t(this->_order), query.size())), propensities );

            return std::accumulate( correlation.cbegin(), correlation.cend(), double( 0 ));
        }

    };

}
#endif //MARKOVIAN_FEATURES_LSMC_HPP

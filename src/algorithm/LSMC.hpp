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
        using ModelTrainer = typename Base::ModelTrainer;
        using HistogramsTrainer  = typename Base::HistogramsTrainer;
        using Histogram = typename Base::Histogram;

        using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
        using HeteroHistograms = std::unordered_map<Order, IsoHistograms>;

        using Ops = MCOps<AAGrouping>;

        explicit LSMC( Order order ) : ZYMC<AAGrouping>( order )
        {}

        template<typename HistogramsCollection>
        explicit LSMC( Order order, HistogramsCollection &&histograms ) :
                ZYMC<AAGrouping>( order, std::forward<HistogramsCollection>( histograms ))
        {}

        explicit LSMC( const std::vector<std::string> &sequences,
                       Order order ) : ZYMC<AAGrouping>( sequences , order )
        {}

        LSMC() = delete;

        LSMC( const LSMC &mE ) = default;

        LSMC( LSMC &&mE ) noexcept
        : ZYMC<AAGrouping>( std::move( mE ) )
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

        static ModelTrainer getLSMCTrainer( Order order )
        {
            return [=]( const std::vector<std::string> &sequences,
                        std::optional<std::reference_wrapper<const Selection >> selection ) -> std::unique_ptr<Base> {
                if ( selection )
                {
                    auto model = Ops::filter( std::move( LSMC( sequences, order )), selection->get());
                    if ( model ) return std::unique_ptr<Base>( new LSMC( std::move( model.value())));
                    else return nullptr;

                } else return std::unique_ptr<Base>( new LSMC( sequences, order ));
            };
        }

        static HistogramsTrainer getLSMCHistogramsTrainer( Order order )
        {
            return [=]( const std::vector<std::string> &sequences,
                        std::optional<std::reference_wrapper<const Selection >> selection ) -> std::optional<HeteroHistograms> {
                if ( selection )
                {
                    auto model = Ops::filter( LSMC( sequences, order ), selection->get());
                    if ( model ) return std::move( model->convertToHistograms());
                    else return std::nullopt;
                } else return std::move( LSMC( sequences, order ).convertToHistograms());
            };
        }


        double propensity( std::string_view query ) const override
        {
            double acc = 0;
            std::pair<double, double> prev = {inf, inf};
            prev.second = std::log( this->_histograms.at( 0 ).at( 0 ).at( Base::_char2ID( query.front())));
            for (Order distance = 1; distance < this->_order && distance < query.size(); ++distance)
            {
                double p = this->probability( query.substr( 0, distance ), query[distance] );
                double microPropensity = std::log( p );
                if ( prev.first < prev.second && prev.second > microPropensity )
                    acc += prev.second;
                prev.first = prev.second;
                prev.second = microPropensity;
            }

            for (auto i = 0; i < int64_t( query.size()) - this->_order - 1; ++i)
            {
                double p = this->probability( query.substr( i, this->_order ), query[i + this->_order] );
                double microPropensity = std::log( p );
                if ( prev.first < prev.second && prev.second > microPropensity )
                    acc += prev.second;
                prev.first = prev.second;
                prev.second = microPropensity;
            }
            return acc;
        }

    };

}
#endif //MARKOVIAN_FEATURES_LSMC_HPP

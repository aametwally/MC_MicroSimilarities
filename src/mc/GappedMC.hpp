//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_LSMC_HPP
#define MARKOVIAN_FEATURES_LSMC_HPP

#include "ZYMC.hpp"

namespace MC
{

template < typename AAGrouping >
class GappedMC : public ZYMC<AAGrouping>
{
public:
    using Base = ZYMC<AAGrouping>;
    using Histogram = typename Base::Histogram;

    using IsoHistograms = std::unordered_map<HistogramID , Histogram>;
    using HeteroHistograms = std::unordered_map<Order , IsoHistograms>;

    explicit GappedMC( Order order ) : Base( order ) {}

    virtual ~GappedMC() = default;

    template < typename HistogramsCollection >
    explicit GappedMC( Order order , HistogramsCollection &&histograms ) :
            Base( order , std::forward<HistogramsCollection>( histograms )) {}

    explicit GappedMC( const std::vector<std::string> &sequences ,
                       Order order ) : Base( sequences , order ) {}

    double probability( std::string_view context , char currentState ) const override
    {
        return this->_polymorphicSummer(
                context , currentState ,
                [this]( std::string_view context , char currentState )
                {
                    double p = 1.0;
                    constexpr float eps = std::numeric_limits<float>::epsilon() * 2;
                    double min = 1;
                    int iFrom = std::max( 0 , int( context.length()) - this->_order );
                    for ( auto i = iFrom; i < context.size(); ++i )
                    {
                        auto distance = Order( context.size() - i );
                        auto c = context[i];
                        auto pBayes = Base::pairwiseProbability( c , currentState , distance );
                        min = std::min( min , pBayes );
                        p *= (pBayes + eps);
                    }
                    return p / (min + eps);
                } );
    }
};


}
#endif //MARKOVIAN_FEATURES_LSMC_HPP

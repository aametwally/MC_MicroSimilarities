//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_ZYMC_HPP
#define MARKOVIAN_FEATURES_ZYMC_HPP

#include "AbstractMC.hpp"

namespace MC
{
/**
 * @brief ZYMC
 * Zheng Yuan Approximated Higher-order Markov Chains
 * Paper: https://febs.onlinelibrary.wiley.com/doi/pdf/10.1016/S0014-5793%2899%2900506-2
 */
template < typename AAGrouping >
class ZYMC : public AbstractMC<AAGrouping>
{
public:
    using Base = AbstractMC<AAGrouping>;
    using Histogram = typename Base::Histogram;

    using IsoHistograms = std::unordered_map<HistogramID , Histogram>;
    using HeteroHistograms = std::unordered_map<Order , IsoHistograms>;

public:
    explicit ZYMC( Order order ) : Base( order )
    {
        assert( order >= 1 );
    }

    virtual ~ZYMC() = default;

    template < typename HistogramsCollection >
    explicit ZYMC( Order order , HistogramsCollection &&histograms )
            :  Base( std::forward<HistogramsCollection>( histograms ) , order )
    {
        assert( order >= 1 );
    }

    explicit ZYMC( const std::vector<std::string> &sequences ,
                   Order order ) : Base( order )
    {
        assert( order >= 1 );
        this->train( sequences );
    }


    static constexpr inline HistogramID lowerOrderID( HistogramID id ) { return id / Base::StatesN; }

    inline double pairwiseProbability( char context ,
                                       char state ,
                                       Order distance ) const
    {
        if ( auto dIt = this->_histograms.find( distance ); dIt != this->_histograms.cend())
        {
            auto &pairs = dIt->second;
            auto c = Base::_char2ID( context );
            if ( auto contextIt = pairs.find( c ); contextIt != pairs.cend())
            {
                auto &p = contextIt->second;
                auto s = Base::_char2ID( state );
                return p.at( s );
            } else return 0;
        } else return 0;
    }

    double probability( std::string_view polymorphicContext , char polymorphicState ) const override
    {
        if ( polymorphicContext.size() > this->getOrder())
        {
            polymorphicContext.remove_prefix( polymorphicContext.size() - this->getOrder());
        }

        return this->_polymorphicSummer(
                polymorphicContext , polymorphicState ,
                [this]( std::string_view context , char state )
                {
                    double p = 1.0;
                    for ( auto i = 0; i < context.size(); ++i )
                    {
                        auto distance = Order( context.size() - i );
                        auto c = context[i];
                        p *= pairwiseProbability( c , state , distance );
                    }
                    return p;
                } );
    }

protected:
    void _incrementInstance( std::string_view context ,
                             char state ,
                             Order distance )
    {
        this->_polymorphicApply(
                context , state ,
                [this , distance]( std::string_view context ,
                                   char state )
                {
                    assert( context.size() == 1 );
                    auto c = Base::_char2ID( context.front());
                    auto s = Base::_char2ID( state );
                    this->_histograms[distance][c].increment( s );
                } );
    }

    void _countInstance( std::string_view sequence ) override
    {
        for ( auto a : sequence )
        {
            this->_polymorphicApply( a , [this]( char state )
            {
                auto c = Base::_char2ID( state );
                this->_histograms[0][0].increment( c );
            } );
        }

        for ( Order distance = 1; distance <= this->_order; ++distance )
            for ( auto i = 0; i + distance < sequence.size(); ++i )
                _incrementInstance( sequence.substr( i , 1 ) , sequence[i + distance] , distance );
    }
};
}
#endif //MARKOVIAN_FEATURES_ZYMC_HPP

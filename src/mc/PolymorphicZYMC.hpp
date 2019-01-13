//
// Created by asem on 31/12/18.
//

#ifndef MARKOVIAN_FEATURES_POLYMORPHICZYMC_HPP
#define MARKOVIAN_FEATURES_POLYMORPHICZYMC_HPP

#include "ZYMC.hpp"

namespace MC
{

template < typename AAGrouping >
class PolymorphicZYMC : public ZYMC<AAGrouping::StatesN>
{
    static constexpr auto States = AAGrouping::StatesN;
    using Base = AbstractMC<States>;

public:
    virtual ~PolymorphicZYMC() = default;

    double probability( std::string_view polymorphicContext , char polymorphicState ) const override
    {
        if ( polymorphicContext.size() > this->getOrder())
        {
            polymorphicContext.remove_prefix( polymorphicContext.size() - this->getOrder());
        }

        return LabeledEntry::polymorphicSummer<AAGrouping>(
                polymorphicContext , polymorphicState ,
                [this]( std::string_view context , char state )
                {
                    double p = 1.0;
                    for ( auto i = 0; i < context.size(); ++i )
                    {
                        auto distance = Order( context.size() - i );
                        auto c = context[i];
                        p *= this->pairwiseProbability( c , state , distance );
                    }
                    return p;
                } );
    }

    double probability( char a ) const override
    {
        return LabeledEntry::polymorphicSummer<AAGrouping>( a , [this]( char state )
        {
            return this->_histograms( 0 , 0 , this->_char2ID( state )).value_or( 0 );
        } );
    }

protected:
    virtual void _incrementInstance( std::string_view context ,
                                     char state ,
                                     Order distance )
    {
        LabeledEntry::polymorphicApply<AAGrouping>(
                context , state ,
                [this , distance]( std::string_view context ,
                                   char state )
                {
                    assert( context.size() == 1 );
                    auto c = Base::_char2ID( context.front());
                    auto s = Base::_char2ID( state );
                    this->_histograms.increment( distance , c , this->_epsilon )( s );
                } );
    }

};
}

#endif //MARKOVIAN_FEATURES_POLYMORPHICZYMC_HPP

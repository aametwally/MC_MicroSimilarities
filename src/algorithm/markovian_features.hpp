#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>

#include "common.hpp"



const std::set< char > aaSet = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
const std::array< char , 256 > aaId([]{
    std::array< char , 256 > ids {};
    int i = 0;
    for( auto aa : aaSet )
        ids.at( static_cast< size_t >( aa )) = i++;
    return ids;
}());

const std::array< std::string , 11 > reducedAASet_DIAMOND = { "KREDQN" , "C" , "G" , "H" , "ILV" , "M" , "F" , "Y" , "W" , "P" , "STA" };
const std::array< char , 11 > reducedAASetRepresentation_DIAMOND([](){
    std::array< char , 11 > newAlphabet;
    for( auto i = 0 ; i < newAlphabet.size() ;++i)
        newAlphabet.at( i ) = 'A' + i;
    return newAlphabet;
}());

const std::array< char , 256 > reducedAAIds_DIAMOND([]{
    std::array< char , 256 > ids = aaId;
    int i = 0;
    for( const auto &aaGroup : reducedAASet_DIAMOND )
    {
        for( auto aa : aaGroup )
            ids.at( static_cast< size_t >( aa )) = i;
        ++i;
    }

    return ids;
}());

auto geometricDistribution( double p )
{
    return [p]( double exponent )
    {
        return pow( p , exponent );
    };
}

template< size_t statesN , typename Precision = double >
class MarkovianKernel
{
    class KernelUnit
    {
    public:
        KernelUnit( Precision pseudoCount = Precision{0.01} )
        {
            _buffer.fill( pseudoCount );
        }

        inline auto &at( char state ){
            return _buffer.at( state );
        }
        inline auto sum() const
        {
            return std::accumulate( _buffer.cbegin() , _buffer.cend() ,
                                    Precision{0} );
        }
        inline auto normalize()
        {
            const auto total =  sum();
            for( auto &p : _buffer )
                p /= total;
        }

        std::string toString() const
        {
            return io::join2string( _buffer , " ");
        }

    private:
        std::array< Precision , statesN > _buffer;
    };

public:
    MarkovianKernel( uint8_t order, uint8_t gapOrder,
                     const std::function<double(int)> &gapOrderProbability ) :
        _order( order ),
        _gapOrder( gapOrder ),
        _gapOrderProbability( gapOrderProbability )
    {
        assert( order > 0 );
        size_t kernelLength = powi< statesN >( order );
        size_t gappedKernelLength = powi< statesN >( gapOrder );
        _kernel = std::vector< KernelUnit > ( kernelLength );
        _gappedKernel = std::vector< KernelUnit > ( gappedKernelLength );
    }

    void train( const std::vector< std::string > &sequences )
    {
        std::for_each( sequences.cbegin(), sequences.cend(),
                       [this]( const std::string &sequence ){
            _countInstance( sequence );
            _countInstanceGapped( sequence );
        });

        for( auto &u : _kernel ) u.normalize();
        for( auto &u : _gappedKernel ) u.normalize();
    }

    void toFiles( const std::string &dir ,
                  const std::string &prefix ,
                  const std::string &id ) const
    {
        std::ofstream kernelFile;
        std::vector< std::string > names1 = { prefix , "profile" , id };
        kernelFile.open( dir + "/" + io::join( names1 , "_" ) + ".array" );
        for( const auto &u : _kernel )
            kernelFile << u.toString() << std::endl;
        kernelFile.close();


        std::ofstream gKernelFile;
        std::vector< std::string > names2 = { prefix , "gprofile" , id };
        gKernelFile.open( dir + "/" + io::join( names2 , "_" ) + ".array" );
        for( const auto &u : _gappedKernel )
            gKernelFile << u.toString() << std::endl;
        gKernelFile.close();

    }
private:
    void _incrementInstanceGapped( std::string::const_iterator seedFrom ,
                                   std::string::const_iterator seedUntil ,
                                   std::string::const_iterator windowFrom ,
                                   std::string::const_iterator windowUntil )
    {

        for( auto it = seedFrom ; it != seedUntil ; ++it )
        {
            auto distance = std::distance( it , windowFrom );
            auto seed = *it;
            auto index = _sequence2Index( windowFrom , windowUntil - 1 ,
                                          seed -  reducedAASetRepresentation_DIAMOND.front( ));
            auto c = reducedAAIds_DIAMOND.at( *(windowUntil - 1) );
            _gappedKernel.at( index ).at( c ) += _gapOrderProbability( distance );
        }
    }


    void _incrementInstance( std::string::const_iterator from ,
                             std::string::const_iterator until )
    {
        assert( from != until );
        auto index = _sequence2Index( from , until - 1 );
        auto c = reducedAAIds_DIAMOND.at( *(until - 1) );
        ++_kernel.at( index ).at( c );
    }

    void _countInstanceGapped( const std::string &sequence )
    {
        using WindowIteratorType = std::pair< std::string::const_iterator, std::string::const_iterator>;
        assert( sequence.size() > 2 * _gapOrder );
        WindowIteratorType seedsIt { sequence.cbegin() , sequence.cbegin() + 1 };
        WindowIteratorType windowIt { sequence.cbegin() + 2 , sequence.cbegin() + 2 + _gapOrder };

        for( auto i = 0 ; i < _gapOrder ; ++i )
        {
            _incrementInstanceGapped( seedsIt.first , seedsIt.second ,
                                      windowIt.first , windowIt.second );
            ++seedsIt.second;
            ++windowIt.first;
            ++windowIt.second;
        }

        while( windowIt.second != sequence.cend())
        {
            _incrementInstanceGapped( seedsIt.first , seedsIt.second ,
                                      windowIt.first , windowIt.second );
            ++seedsIt.first;
            ++seedsIt.second;
            ++windowIt.first;
            ++windowIt.second;
        }

    }

    void _countInstance( const std::string &sequence )
    {
        for( auto i = 0 ; i < sequence.size() - ( _order + 1 )  ; ++i )
            _incrementInstance( sequence.cbegin() + i , sequence.cbegin() + i + _order + 1 );
    }

    static size_t _sequence2Index( std::string::const_iterator from ,
                                   std::string::const_iterator until ,
                                   size_t init = 0 )
    {
        size_t code = init;
        for( auto it = from ; it != until ; ++it )
            code = code * statesN + *it - reducedAASetRepresentation_DIAMOND.front();
        return code;
    }

private:
    const uint8_t _order;
    const uint8_t _gapOrder;
    const std::function<double(int)> _gapOrderProbability;

    std::vector< KernelUnit > _kernel;
    std::vector< KernelUnit > _gappedKernel;
};




namespace preprocess
{

std::string reduceAlphabets_DIAMOND( const std::string &sequence )
{
    std::string reducedSequence;
    reducedSequence.reserve( sequence.size());
    for( auto a : sequence )
        reducedSequence.push_back( reducedAASetRepresentation_DIAMOND.at( reducedAAIds_DIAMOND.at( a )));

    assert( reducedSequence.find( '\0' ) == std::string::npos );
    return reducedSequence;
}

std::vector< std::vector< std::string >>
unirefClusters2ReducedAASequences_DIAMOND( const std::vector< UnirefItem > &unirefItems )
{
    std::map< std::string , std::vector< std::string >> unirefClusters;
    for( const UnirefItem &ui : unirefItems )
        unirefClusters[ ui.clusterName ].emplace_back( reduceAlphabets_DIAMOND( ui.sequence ));

    std::vector< std::vector< std::string >> reducedAlphabetClusters;
    for( auto it = unirefClusters.cbegin() ; it != unirefClusters.cend() ; ++it )
        reducedAlphabetClusters.emplace_back( std::move( it->second ));

    return reducedAlphabetClusters;
}
}

using MarkovianKernel11Alphabet = MarkovianKernel< 11 , float >;
using UnirefClusters = std::map< uint32_t , std::vector< std::string >>;
using MarkovianProfiles = std::vector< std::pair< uint32_t , MarkovianKernel11Alphabet >>;

std::pair< MarkovianProfiles , MarkovianProfiles >
markovianTraining( const std::vector< std::vector< std::string >> &population ,
                   uint32_t markovianOrder ,
                   uint32_t gapOrder ,
                   uint32_t minimumClusterSize ,
                   float testPercentage = 0.1f )
{
    uint32_t clusterId = 0;
    MarkovianProfiles consensusProfiles;
    MarkovianProfiles testProfiles;

    auto clusterSize = []( const std::vector< std::string > &cluster )
    {
        size_t count = 0;
        for( auto &s : cluster )
            count += s.length();
        return count;
    };

    for( auto &cluster : population )
        if( clusterSize( cluster ) >= minimumClusterSize * 10 )
        {
            MarkovianKernel11Alphabet kernel( markovianOrder , gapOrder ,
                                              geometricDistribution( 0.5f ));
            if( cluster.size() * testPercentage > 1 )
            {
                auto seperatedItems = subsetRandomSeparation( cluster , testPercentage );
                for( auto &s : seperatedItems.first )
                {
                    MarkovianKernel11Alphabet testKernel( markovianOrder , gapOrder ,
                                                          geometricDistribution( 0.5f ));
                    testKernel.train( {s} );
                    testProfiles.emplace_back( clusterId , std::move( testKernel ));
                }
                kernel.train( seperatedItems.second );
            }
            else
                kernel.train( cluster );

            consensusProfiles.emplace_back( clusterId , std::move( kernel ));
            ++clusterId;
        }
    return std::make_pair( consensusProfiles , testProfiles );
}

void writeResults( const std::pair< MarkovianProfiles , MarkovianProfiles > &results ,
                   const std::string &dir )
{
    const std::string consensusPrefix = "consensus";
    for( auto &p : results.first )
        p.second.toFiles( dir , consensusPrefix , std::to_string( p.first ));


    size_t testItem = 0;
    for( auto &p : results.second )
    {
        std::string testProfile = "test";
        testProfile += std::to_string( testItem++ );
        p.second.toFiles( dir , testProfile ,  std::to_string( p.first ) );
    }

}


#endif

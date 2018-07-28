#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>

#include <fmt/format.h>

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

template< typename T >
auto inverseFunction()
{
    return []( T n ){
        return T{1}/n;
    };
}

template< size_t statesN , typename Precision = double >
class MarkovianKernel
{
public:
    class KernelUnit
    {
    public:
        KernelUnit( Precision pseudoCount = Precision{0.01} )
            : _hits( 0 )
        {
            _buffer.fill( pseudoCount );
        }


        inline auto &at( char state ) const {
            return _buffer.at( state );
        }

        inline void increment( char state ) {
            ++_hits;
            ++_buffer.at( state );
        }

        inline void increment( char state , Precision val )
        {
            ++_hits;
            _buffer.at( state ) += val;
        }

        inline auto sum() const
        {
            return std::accumulate( _buffer.cbegin() , _buffer.cend() ,
                                    Precision{0} );
        }
        inline void normalize()
        {
            if( !isPristine())
            {
                const auto total =  sum();
                for( auto &p : _buffer )
                    p /= total;
            }
            else
            {
                constexpr Precision p = Precision{1} / statesN;
                std::fill( _buffer.begin() , _buffer.end() , p );
            }
        }

        constexpr static auto maxInformation()
        {
            Precision ent{0};
            constexpr Precision p = Precision{1} / statesN;
            for( auto i = 0; i < statesN ; ++i )
                ent += p * log2( p );
            return ent;
        }

        inline auto information() const
        {
            Precision ent{0};
            for( auto p : _buffer )
                ent += p * log2( p );
            return ent;
        }

        inline auto chiSquaredDistance( const KernelUnit &unit ) const
        {
            Precision sum{0};
            for( auto i = 0; i < statesN; ++i  )
            {
                auto m = _buffer.at( i ) - unit._buffer.at( i );
                sum += m * m / _buffer.at( i );
            }
            return sum;
        }

        inline size_t hits() const
        {
            return _hits;
        }

        inline bool isPristine() const
        {
            return _hits == 0;
        }

        std::string toString() const
        {
            return io::join2string( _buffer , " ");
        }

    protected:
        std::array< Precision , statesN > _buffer;
        size_t _hits;
    };

    struct FilteredData
    {
        using RowIndex = size_t;
        std::unordered_map< RowIndex , KernelUnit > kernels;
        std::unordered_map< RowIndex , KernelUnit > gappedKernels;
    };

public:
    MarkovianKernel( uint8_t order, uint8_t gapOrder,
                     const std::function<double(int)> &gapOrderProbability ) :
        _order( order ),
        _gapOrder( gapOrder ),
        _gapOrderProbability( gapOrderProbability ),
        _hits(0),
        _gappedHits(0)
    {
        assert( order > 0 );
        size_t kernelLength = powi< statesN >( order );
        size_t gappedKernelLength = powi< statesN >( gapOrder );
        _kernel = std::vector< KernelUnit > ( kernelLength );
        _gappedKernel = std::vector< KernelUnit > ( gappedKernelLength );
    }

    static FilteredData filterPercentile( const FilteredData &data , float percent = 0.9f )
    {
        return { filterPercentile( data.kernels , percent ) ,
                    filterPercentile( data.gappedKernels , percent )};
    }

    FilteredData filterPrisrine() const
    {
        return { filterPristine( _kernel ), filterPristine( _gappedKernel ) };
    }

    static std::unordered_map< size_t , KernelUnit >
    filterPristine( const std::vector< KernelUnit > &kernels )
    {
        std::unordered_map< size_t , KernelUnit > filtered;
        for( size_t i = 0 ; i < kernels.size() ; ++i  )
            if( !kernels.at( i ).isPristine())
                filtered.emplace( i , kernels.at( i ));
        return filtered;
    }

    static std::unordered_map< size_t , KernelUnit >
    filterPercentile( const std::unordered_map< size_t , KernelUnit > &filteredKernel ,
                      float percentile  )
    {
        std::vector< std::pair< size_t , KernelUnit >> v;
        for( const auto &p : filteredKernel ) v.push_back(p);

        auto cmp =  []( const std::pair< size_t, KernelUnit > &p1 ,  const std::pair< size_t, KernelUnit > &p2 ){
            return p1.second.hits() > p2.second.hits();
        };

        size_t percentileTailIdx = filteredKernel.size() * percentile;
        std::nth_element( v.begin() , v.begin() + percentileTailIdx,
                          v.end() , cmp );

        std::unordered_map< size_t , KernelUnit > filteredKernel2( v.begin() , v.begin() + percentileTailIdx );

        return filteredKernel2;
    }

    void train( const std::vector< std::string > &sequences )
    {
        if( _gapOrder > 0 )
        {
            for( const auto &s : sequences )
            {
                _countInstance( s );
                _countInstanceGapped( s );
            }
        }
        else
        {
            for( const auto &s : sequences )
            {
                _countInstance( s );
            }
        }

        for( auto &u : _kernel ) u.normalize();
        for( auto &u : _gappedKernel ) u.normalize();
    }

    size_t hits() const
    {
        return _hits;
    }

    size_t gappedHits() const
    {
        return _gappedHits;
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
        assert( windowFrom != windowUntil );
        for( auto it = seedFrom ; it != seedUntil ; ++it )
        {
            auto distance = std::distance( it , windowFrom );
            auto seed = *it;
            auto index = _sequence2Index( windowFrom , windowUntil - 1 ,
                                          seed -  reducedAASetRepresentation_DIAMOND.front( ));
            auto c = reducedAAIds_DIAMOND.at( *(windowUntil - 1) );
            _gappedKernel.at( index ).increment( c , _gapOrderProbability( distance ));
        }
    }


    void _incrementInstance( std::string::const_iterator from ,
                             std::string::const_iterator until )
    {
        assert( from != until );
        auto index = _sequence2Index( from , until - 1 );
        auto c = reducedAAIds_DIAMOND.at( *(until - 1) );
        _kernel.at( index ).increment( c );
    }

    void _countInstanceGapped( const std::string &sequence )
    {
        ++_gappedHits;
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
        ++_hits;
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
    size_t _hits;
    size_t _gappedHits;
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
using KernelUnit = MarkovianKernel11Alphabet::KernelUnit;
using UnirefClusters = std::map< uint32_t , std::vector< std::string >>;

struct MarkovianFilteredProfile{
    using RowIndex = size_t;
    MarkovianFilteredProfile( MarkovianFilteredProfile &&other) = default;
    size_t clusterId;
    size_t hits;
    size_t gappedHits;
    std::unordered_map< RowIndex , KernelUnit > filteredKernel;
    std::unordered_map< RowIndex , KernelUnit > filteredGappedKernel;
};

using MarkovianFilteredProfiles = std::vector< MarkovianFilteredProfile >;

std::pair< MarkovianFilteredProfiles , MarkovianFilteredProfiles >
markovianTraining( const std::vector< std::vector< std::string >> &population ,
                   uint32_t markovianOrder ,
                   uint32_t gapOrder ,
                   uint32_t minimumClusterSize ,
                   float testPercentage = 0.1f )
{
    size_t clusterId = 0;
    MarkovianFilteredProfiles consensusProfiles;
    MarkovianFilteredProfiles testProfiles;

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
                auto separatedItems = subsetRandomSeparation( cluster , testPercentage );
                for( auto &s : separatedItems.first )
                {
                    MarkovianKernel11Alphabet testKernel( markovianOrder , gapOrder ,
                                                          geometricDistribution( 0.5f ));
                    testKernel.train( {s} );
                    auto filtered =  testKernel.filterPrisrine();
                    auto profile = MarkovianFilteredProfile{clusterId , testKernel.hits(),
                            testKernel.gappedHits(),
                            filtered.kernels  ,  filtered.gappedKernels };
                    testProfiles.emplace_back( std::move( profile ));
                }
                kernel.train( separatedItems.second );
            }
            else
            {
                kernel.train( cluster );
            }

            auto filtered =  kernel.filterPrisrine();
            auto profile = MarkovianFilteredProfile{clusterId ,  kernel.hits(), kernel.gappedHits(),
                    filtered.kernels  ,  filtered.gappedKernels };
            consensusProfiles.emplace_back( std::move( profile ));
            ++clusterId;
        }
    return std::make_pair( std::move( consensusProfiles ) ,
                           std::move( testProfiles ));
}

namespace classification
{
struct MatchDistance
{
    size_t id;
    float distance;

    bool operator>( const MatchDistance &other ) const
    {
        return distance > other.distance;
    }
};

using MatchSet = std::set< MatchDistance , std::greater< MatchDistance >>;

struct Classification
{
    size_t testId;
    size_t trueCluster;
    MatchSet bestMatches;

    bool trueClusterFound() const
    {
        return std::find_if( bestMatches.cbegin(), bestMatches.cend(),
                             [this]( const MatchDistance &m ){
            return m.id == trueCluster;
        }) != bestMatches.cend();
    }
};

bool allHitsCovered( const std::unordered_map< size_t , KernelUnit > &query ,
                     const std::unordered_map< size_t , KernelUnit > &target )
{
    for( auto it = query.cbegin() ; it != query.cend() ; ++it )
        if( target.find( it->first ) == target.cend())
            return false;
    return true;
}

float totalChiSquaredDistance( const MarkovianFilteredProfile &query ,
                               const MarkovianFilteredProfile &target )
{
    float sum{0};
    for( auto it = query.filteredGappedKernel.cbegin() ; it != query.filteredGappedKernel.cend() ; ++it )
    {
        try
        {
            auto &unit1 = it->second;
            auto &unit2 = target.filteredGappedKernel.at( it->first );
            //        float w1 = float{unit1.hits()} / query.hits ;
            //        float w2 = float{unit2.hits()} / target.hits;
            //        float w = w1 / w1 ;
            sum += unit1.chiSquaredDistance( unit2 ) ;
        } catch( const std::out_of_range &e )
        {

        }


    }
    return sum;
}

MatchSet findSimilarities( const MarkovianFilteredProfile &query ,
                           const MarkovianFilteredProfiles &targets ,
                           size_t kNearest = 5 )
{
    MatchSet matchSet;
    for( const MarkovianFilteredProfile &target : targets )
    {
        //        if( allHitsCovered( query.filteredKernel , target.filteredKernel ))
        //        {
        float distance = totalChiSquaredDistance( query ,
                                                  target );
        matchSet.insert({ target.clusterId , distance });

        if( matchSet.size() > kNearest )
            matchSet.erase( matchSet.begin());
        //        }
    }
    return matchSet;
}

std::vector< Classification > classify( const MarkovianFilteredProfiles &queries,
                                        const MarkovianFilteredProfiles &targets )
{
    std::vector< Classification > classifications;
    size_t testId = 0;
    size_t trueCluster = 0;
    for( const MarkovianFilteredProfile &q : queries )
    {
        Classification result{ ++testId , q.clusterId ,
                    findSimilarities( q , targets )};
        classifications.emplace_back( result );
        trueCluster += result.trueClusterFound();
        if( testId % 100 == 0 )
        {
            fmt::print("progress:{}\taccuracy:{}\n",
                       float{testId}/queries.size() ,
                       float{trueCluster}/testId );
        }
    }
    return classifications;
}


}

//void writeResults( const std::pair< MarkovianProfiles , MarkovianProfiles > &results ,
//                   const std::string &dir )
//{
//    const std::string consensusPrefix = "consensus";
//    for( auto &p : results.first )
//        p.second.toFiles( dir , consensusPrefix , std::to_string( p.first ));


//    size_t testItem = 0;
//    for( auto &p : results.second )
//    {
//        std::string testProfile = "test";
//        testProfile += std::to_string( testItem++ );
//        p.second.toFiles( dir , testProfile ,  std::to_string( p.first ) );
//    }

//}


#endif

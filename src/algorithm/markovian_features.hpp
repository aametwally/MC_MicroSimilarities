#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>

#include <fmt/format.h>

#include "common.hpp"
#include "UniRefEntry.hpp"


const std::set< char > aaSet = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
const std::array< char , 256 > aaId([]{
    std::array< char , 256 > ids {};
    int i = 0;
    for( auto aa : aaSet )
        ids.at( static_cast< size_t >( aa )) = i++;
    return ids;
}());

const std::array< std::string , 15 > reducedAASet_Olfer14 = { "KR","E","D","Q","N" , "C" , "G" , "H" , "ILVM" , "F" , "Y" , "W" , "P" , "ST","A" };
const std::array< std::string , 8 > reducedAASet_Olfer8 = { "KRH" , "ED" , "C" , "G" , "AILVM" , "FYW" , "P" , "NQST" };

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

public:
    MarkovianKernel( int order ) :
        _order( order ),
        _hits(0)
    {
        assert( order > 0 );
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
        for( const auto &s : sequences )
            _countInstance( s );

        for( auto & [rowId,row] : _kernel )
            row.normalize();
    }

    size_t hits() const
    {
        return _hits;
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
    }

    const std::unordered_map< size_t , KernelUnit > &kernel() const
    {
        return _kernel;
    }

    int order() const
    {
        return _order;
    }

private:
    void _incrementInstance( std::string::const_iterator from ,
                             std::string::const_iterator until )
    {
        assert( from != until );
        auto index = _sequence2Index( from , until - 1 );
        auto c = reducedAAIds_DIAMOND.at( *(until - 1) );
        _kernel[ index ].increment( c );
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
    const int _order;
    std::unordered_map< size_t , KernelUnit > _kernel;
    size_t _hits;
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

std::vector< UniRefEntry >
reducedAAEntries_DIAMOND( const std::vector< UniRefEntry > &unirefEntries )
{
    std::vector< UniRefEntry > unirefReducedEntries;
    for( const UniRefEntry &ui : unirefEntries )
    {
        auto reduced = ui;
        reduced.setSequence( reduceAlphabets_DIAMOND( ui.getSequence() ));
        unirefReducedEntries.emplace_back( reduced );
    }
    return unirefReducedEntries;
}
}

using MarkovianKernel11Alphabet = MarkovianKernel< 11 , float >;
using MarkovianProfile = MarkovianKernel11Alphabet;
using KernelUnit = MarkovianProfile::KernelUnit;
using UnirefClusters = std::map< uint32_t , std::vector< std::string >>;

using MarkovianProfiles = std::map< std::string , MarkovianProfile >;

MarkovianProfiles
markovianTraining( const std::map< std::string , std::vector< std::string >> &training ,
                   int markovianOrder )
{
    MarkovianProfiles trainedProfiles;

    for( const auto & [id,sequences] : training )
    {
        MarkovianKernel11Alphabet kernel( markovianOrder );
        kernel.train( sequences );
        trainedProfiles.emplace( id , std::move( kernel ));
    }
    return trainedProfiles;
}

namespace classification
{
struct MatchDistance
{
    std::string id;
    float distance;

    bool operator>( const MatchDistance &other ) const
    {
        return distance > other.distance;
    }
};

using MatchSet = std::set< MatchDistance , std::greater< MatchDistance >>;

struct Classification
{
    std::string trueCluster;
    MatchSet bestMatches;

    bool trueClusterFound() const
    {
        return std::find_if( bestMatches.cbegin(), bestMatches.cend(),
                             [this]( const MatchDistance &m ){
            return m.id == trueCluster;
        }) != bestMatches.cend();
    }
};

float totalChiSquaredDistance( const MarkovianProfile &query ,
                               const MarkovianProfile &target )
{
    float sum{0};
    for( const auto &[ rowId , row ] : query.kernel() )
    {
        try
        {
            auto &unit1 = row;
            auto &unit2 = target.kernel().at( rowId );
            sum += unit1.chiSquaredDistance( unit2 ) ;
        } catch( const std::out_of_range &e )
        {

        }
    }
    return sum;
}

MatchSet findSimilarities( const MarkovianProfile &query ,
                           const MarkovianProfiles &targets ,
                           size_t kNearest = 3 )
{
    MatchSet matchSet;
    for( const auto & [clusterId,clusterProfile] : targets )
    {
        float distance = totalChiSquaredDistance( query , clusterProfile );
        matchSet.insert({ clusterId , distance });

        if( matchSet.size() > kNearest )
            matchSet.erase( matchSet.begin());
    }
    return matchSet;
}

std::vector< Classification > classify( const std::vector< FastaEntry > &queries,
                                        const MarkovianProfiles &targets )
{
    const int order = targets.begin()->second.order();
    std::vector< Classification > classifications;
    for( const auto &q : queries )
    {
        MarkovianProfile p( order );
        p.train( {q.getSequence()} );
        Classification result{  "" , findSimilarities( p , targets )};
        classifications.emplace_back( result );
    }
    return classifications;
}

std::vector< Classification > classify_VALIDATION(
        const std::vector< UniRefEntry > &queries,
                                        const MarkovianProfiles &targets )
{
    const int order = targets.begin()->second.order();
    std::vector< Classification > classifications;
    size_t truePositive = 0;
    size_t tested = 0;
    for( const auto &unirefItem : queries )
    {
        MarkovianProfile p( order );
        p.train( {unirefItem.getSequence()} );
        Classification result{ unirefItem.getClusterName() , findSimilarities( p , targets )};
        classifications.emplace_back( result );
        truePositive += result.trueClusterFound();
        ++tested;
        if( (tested * 100) / queries.size() - ((tested - 1) * 100) / queries.size() > 0 )
        {
            fmt::print("[progress:{}%][accuracy:{}]\n",
                       float{tested*100}/queries.size() ,
                       float{truePositive}/tested );
        }
    }
    return classifications;
}
}



#endif

#ifndef COMMON_HH
#define COMMON_HH

// STL containers
#include <vector>
#include <array>
#include <numeric>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <set>

// STL streaming
#include <iostream>
#include <sstream>
#include <fstream>

// STL misc
#include <algorithm>
#include <cmath>
#include <typeinfo>
#include <functional>
#include <cassert>
#include <random>



using FastaItem = std::pair< std::string , std::string >;

struct UnirefItem
{
    std::string memberId;
    std::string clusterName;
    std::string sequence;
};

const std::array< char , 4 > acgt = { 'A' , 'C' , 'G' , 'T' };
const std::array< char , 4 > tgca = { 'T' , 'G' , 'C' , 'A' };
const std::array< std::string , 4 > bpMutants = { "CGT" , "AGT" , "ACT" , "ACG"};

const std::array< int , 256 > codeACGT([]{
    std::array< int , 256 > codes;
    std::fill( codes.begin() , codes.end() , -1 );
    codes['A'] = 0;
    codes['C'] = 1;
    codes['G'] = 2;
    codes['T'] = 3;
    return codes;
}());

template< size_t Base  >
constexpr size_t powi( uint16_t exponent )
{
    return (exponent == 0)? size_t{1} : Base * powi< Base >( exponent - 1);
}

double powd( double x , uint16_t exponent )
{
    return (exponent == 0)? 1.f : x * powd( x , exponent - 1 );
}

/**
 * @brief random_element
 * credits: http://stackoverflow.com/a/6943003
 * @param begin
 * @param end
 * @return
 */
template <typename I>
I randomElement( I begin, I end )
{
    using UIntType = unsigned long;
    const UIntType n = std::distance(begin, end);
    const UIntType divisor = (UIntType(RAND_MAX) + 1) / n;

    unsigned long k;
    do { k = std::rand() / divisor; } while (k >= n);

    std::advance(begin, k);
    return begin;
}

template <typename I>
auto randomElementSampler( I begin, I end )
{
    static std::mt19937 rng( std::random_device{}());
    auto n = std::distance( begin , end );
    return [n,begin]() -> I
    {
        std::uniform_int_distribution< decltype(n) > dist{ 0 , n - 1 };
        return std::next( begin , dist( rng ));
    };
}

template <typename I>
auto randomIndexSampler( I begin, I end )
{
    static std::mt19937 rng( std::random_device{}());
    auto n = std::distance( begin , end );
    return [n,begin]() -> size_t
    {
        std::uniform_int_distribution< decltype(n) > dist{ 0 , n - 1 };
        return dist( rng );
    };
}

template <typename ContainerT>
std::pair< ContainerT , ContainerT >
subsetRandomSeparation( const ContainerT &items , float percentage )
{
    ContainerT subset;
    ContainerT rest;
    std::set< size_t > subsetIndices;

    size_t subsetSize = 0;
    auto sampler = randomIndexSampler( std::begin( items ) , std::end( items ));
    while( static_cast< float >( subsetSize ) / items.size() < percentage )
    {
        auto p = subsetIndices.insert( sampler());
        subsetSize += p.second;
    }


    auto indicesIt = subsetIndices.cbegin();
    for( auto i = 0; i < items.size() ; ++i )
    {
        if( indicesIt != subsetIndices.cend() && i == *indicesIt )
        {
            subset.push_back( items.at( i ));
            ++indicesIt;
        }
        else
            rest.push_back( items.at( i ));
    }
    return std::make_pair( subset , rest );
}

namespace io
{
template< typename CharType = char >
auto getFileLines( const std::string &filePath )
{
    std::basic_ifstream< CharType > f( filePath );
    std::vector< std::basic_string< CharType > > lines;
    std::basic_string< CharType > line;
    if( f )
        while( std::getline( f , line ))
            lines.push_back( line );
    else std::cout << "Failed to open file:" << filePath << std::endl;
    return lines;
}

std::vector< std::string >
readInputStream()
{
    std::string line;
    std::vector< std::string > lines;
    while( std::getline( std::cin , line ))
        lines.push_back( line );

    return lines;
}

auto split( const std::string &s , char delim )
{
    std::stringstream ss( s );
    std::vector< std::string > tokens;
    std::string token;
    while( std::getline( ss , token , delim ))
        tokens.push_back( token );
    return tokens;
}

auto split( const std::string &s )
{
    std::stringstream ss( s );
    std::vector< std::string > tokens;
    while( !ss.eof() )
    {
        std::string i;
        ss >> i;
        tokens.emplace_back( i );
    }
    return tokens;
}

auto split( const std::string &s ,
            std::string delim )
{
    std::vector< std::string > tokens;
    size_t last = 0; size_t next = 0;
    while ((next = s.find(delim, last )) != std::string::npos)
    {
        tokens.push_back( s.substr(last , next - last));
        last = next + 1;
    }
    last += delim.length() - 1;
    if( last < s.length() )
        tokens.push_back( s.substr( last , std::string::npos ));
    return tokens;
}

template< typename SeqIt >
std::string join( SeqIt first , SeqIt last , const std::string &sep )
{
    auto binaryJoinString = [sep]( std::string &a , const std::string &b ) -> std::string&
    {
        return  a += ( ((a.length() > 0) ? sep : "") +  b );
    };
    return std::accumulate( first , last ,
                            std::string() , binaryJoinString  );
}

template< typename Container = std::vector< std::string >>
std::string join( const Container &container ,
                  const std::string &sep )
{
    return join( container.cbegin() , container.cend() , sep );
}

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}


auto
findWithMismatches( const std::string &str ,
                    const std::string &substr,
                    unsigned int d )
{
    assert( substr.size() <= str.size() && substr.size() > d );
    auto cStr = str.data();
    auto cSubstr = substr.data();
    auto k = substr.size();

    auto occurance = [k,d,&cStr,&cSubstr]( std::string::size_type index )
    {
        decltype(d) mismatches = 0;
        decltype(index) i = 0;
        while( mismatches <= d && i < k )
            mismatches += cStr[ index + i ] != cSubstr[ i++ ];
        return mismatches <= d;
    };

    for( std::string::size_type i = 0, until = str.size() - substr.size() + 1;
         i <  until ; i++ )
        if( occurance( i ))
            return i;

    return std::string::npos;
}

template< typename SeqIt >
std::vector< std::string > asStringsVector( SeqIt firstIt , SeqIt lastIt )
{
    std::vector< std::string > stringified;
    if( std::distance( firstIt , lastIt ) > 0 )
    {
        using T = std::remove_reference_t<decltype(*firstIt)>;
        std::transform( firstIt , lastIt ,
                        std::inserter( stringified , std::end( stringified )) ,
                        []( const T &element ) { return std::to_string( element ); } );
    }
    return stringified;
}

template< typename Container >
std::vector< std::string > asStringsVector( const Container &container )
{
    return asStringsVector( container.cbegin() , container.cend());
}

template< typename Container >
std::string join2string( const Container &container, const std::string &sep )
{
    auto s = asStringsVector( container.cbegin() , container.cend());
    return join( s , sep );
}


std::vector< FastaItem >
readFastaFile( const std::string &path ,
               int limit = -1 )
{
    std::ifstream input( path );
    if (!input.good()) {
        std::cerr << "Error opening: " << path << " . You have failed." << std::endl;
        return {};
    }
    std::string line, id, DNA_sequence;
    std::vector< FastaItem > fasta;

    while ( std::getline(input, line) && ( limit == -1 || fasta.size() < limit )) {

        // line may be empty so you *must* ignore blank lines
        // or you have a crash waiting to happen with line[0]
        if(line.empty())
            continue;

        if (line.front() == '>') {
            // output previous line before overwriting id
            // but ONLY if id actually contains something
            if(!id.empty())
                fasta.emplace_back( id , DNA_sequence );

            id = line.substr(1);
            DNA_sequence.clear();
        }
        else {//  if (line[0] != '>'){ // not needed because implicit
            DNA_sequence += line;
        }
    }

    // output final entry
    // but ONLY if id actually contains something
    if(!id.empty())
        fasta.emplace_back( id , DNA_sequence );
    return fasta;
}

UnirefItem fasta2UnirefItem( const FastaItem &fItem )
{
    UnirefItem uniref;
    auto tokens = split( fItem.first );
    uniref.memberId = split( split( tokens.front() , "_").front() , "|" ).front();
    uniref.clusterName = join( tokens.cbegin() + 1 ,
                               std::find_if( tokens.cbegin() + 1 , tokens.cend() ,
                                             []( const std::string &s ){
        return s.find( '=' ) != std::string::npos;
    }) , " ");
    uniref.sequence = fItem.second;
    return uniref;
}

std::vector< UnirefItem >
fasta2UnirefItems( const std::vector< FastaItem > &fastaItems )
{
    std::vector< UnirefItem > unirefItems;
    std::transform( fastaItems.cbegin() , fastaItems.cend() ,
                    std::back_inserter( unirefItems ) , fasta2UnirefItem );
    return unirefItems;
}

}

#endif // COMMON_HH

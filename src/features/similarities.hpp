//
// Created by asem on 03/08/18.
//

#ifndef MARKOVIAN_FEATURES_DISTANCES_HPP
#define MARKOVIAN_FEATURES_DISTANCES_HPP

#include "common.hpp"


struct Cost
{
};
struct Score
{
};


template<typename Derived>
struct Criteria
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();

    template<typename Iterator>
    static double measure( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        return Derived::apply( first1, last1, first2, last2 );
    }

    template<typename Container>
    static double measure( const Container &kernel1, const Container &kernel2 )
    {
        return Derived::apply( std::cbegin( kernel1 ), std::cend( kernel1 ),
                               std::cbegin( kernel2 ), std::cend( kernel2 ));
    }

private:
    Criteria() = default;

    friend Derived;
};

struct ChiSquared : public Criteria<ChiSquared>, Cost
{
    template<typename Iterator>
    static double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            auto m = *it1 - *it2;
            sum += m * m / *it1;
        }
        return sum;
    }

private:
    ChiSquared() = default;
};

struct Cosine : public Criteria<Cosine>, Score
{
    template<typename Iterator>
    static inline double norm2( Iterator first, Iterator last )
    {
        double sum = 0;
        for (auto it = first; it != last; ++it)
        {
            sum += (*it) * (*it);
        }
        return std::sqrt( sum );
    }

    template<typename Iterator>
    static inline double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum{0};
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
            sum += (*it1) * (*it2);

        return sum / (norm2( first1, last1 ) * norm2( first2, last2 ));
    }

private:
    Cosine() = default;
};

struct KullbackLeiblerDivergence : public Criteria<KullbackLeiblerDivergence>, Cost
{

    static KullbackLeiblerDivergence &instance()
    {
        static KullbackLeiblerDivergence inst;
        return inst;
    }


    /**
     * @brief Kullback-Leibler Divergence $D_{KL}(Q||P)$
     * @tparam Iterator
     * @param qFirst
     * @param qLast
     * @param pFirst
     * @param pLast
     * @return
     */
    template<typename Iterator>
    static inline double apply( Iterator qFirst, Iterator qLast, Iterator pFirst, Iterator pLast )
    {
        assert( std::distance( qFirst, qLast ) == std::distance( pFirst, pLast ));
        double sum{0};
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
            sum += (*qIt) * std::log((*qIt + eps) / (*pIt + eps));

        return sum;
    }

private:
    KullbackLeiblerDivergence() = default;
};

struct Measurement
{

    Measurement( std::string label, double val )
            : _label( std::move( label )), _value( val )
    {}

    bool operator==( const Measurement &other ) const
    {
        return _label == other._label && _value == other._value;
    }

    bool operator>( const Measurement &other ) const
    {
        return _value > other._value;
    }

    bool operator<( const Measurement &other ) const
    {
        return _value < other._value;
    }

    bool operator>=( const Measurement &other ) const
    {
        return _value >= other._value;
    }

    bool operator<=( const Measurement &other ) const
    {
        return _value <= other._value;
    }

    const std::string &getLabel() const
    {
        return _label;
    }

private:
    const std::string _label;
    const double _value;
};


template<typename T, typename Comp>
struct PriorityQueueFixed
{
    using Queue = std::set<T, Comp>;

    explicit PriorityQueueFixed( size_t kTop )
            : _kTop( kTop )
    {}

    size_t size() const
    {
        return _q.size();
    }

    const T &top() const
    {
        return *_q.crbegin();
    }

    bool empty() const
    {
        return _q.empty();
    }

    template<class... Args>
    auto emplace( Args &&... args )
    {
        auto res = _q.emplace( args... );
        if ( _q.size() > _kTop )
            _q.erase( _q.begin());
        return res;
    };

    auto insert( T &&val )
    {
        auto res = _q.insert( val );
        if ( _q.size() > _kTop )
            _q.erase( _q.begin());
        return res;
    }

    template<typename Predicate>
    long findRank( const Predicate &predicate ) const
    {
        auto trueClusterIt = std::find_if( _q.crbegin(), _q.crend(), predicate );
        if ( trueClusterIt == _q.crend())
            return -1;
        else return std::distance( _q.crbegin(), trueClusterIt );
    }

    template<typename Predicate>
    bool contains( const Predicate &predicate ) const
    {
        return std::find_if( _q.cbegin(), _q.cend(), predicate ) != _q.cend();
    }

private:
    Queue _q;
    const size_t _kTop;
};

template<typename T, typename Enable = void>
struct MatchSet;

template<typename T>
struct MatchSet<T, typename std::enable_if<std::is_base_of<Cost, T>::value>::type>
{
    using Queue = PriorityQueueFixed<Measurement, std::greater<> >;
    static constexpr double WorstValue = std::numeric_limits<double>::infinity();
};

template<typename T>
struct MatchSet<T, typename std::enable_if<std::is_base_of<Score, T>::value>::type>
{
    using Queue = PriorityQueueFixed<Measurement, std::less<>>;
    static constexpr double WorstValue = -std::numeric_limits<double>::infinity();
};


template<typename Criteria>
struct ClassificationCandidates
{
    using Queue = typename MatchSet<Criteria>::Queue;

    explicit ClassificationCandidates( std::string trueLabel, Queue q )
            : _trueLabel( std::move( trueLabel )), _bestMatches( std::move( q ))
    {}

    std::string bestMatch() const
    {
        return _bestMatches.top().getLabel();
    }

    long trueClusterRank() const
    {
        return _bestMatches.findRank( [this]( const Measurement &m ) {
            return m.getLabel() == _trueLabel;
        } );
    }

    bool trueClusterFound() const
    {
        return _bestMatches.contains( [this]( const Measurement &m ) {
            return m.getLabel() == _trueLabel;
        } );
    }

    const std::string &trueCluster() const
    {
        return _trueLabel;
    }

private:
    std::string _trueLabel;
    Queue _bestMatches;
};


enum class CriteriaEnum
{
    ChiSquared,
    Cosine,
    KullbackLeiblerDiv
};

const std::map<std::string, CriteriaEnum> CriteriaLabels{
        {"chi", CriteriaEnum::ChiSquared},
        {"cos", CriteriaEnum::Cosine},
        {"kl",  CriteriaEnum::KullbackLeiblerDiv}
};

template<typename...>
struct CriteriaList
{
};
using SupportedCriteria = CriteriaList<ChiSquared, Cosine, KullbackLeiblerDivergence>;


#endif //MARKOVIAN_FEATURES_DISTANCES_HPP
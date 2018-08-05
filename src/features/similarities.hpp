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
        Derived &inst = Derived::instance();
        return inst.apply( first1, last1, first2, last2 );
    }

private:
    Criteria() = default;

    friend Derived;
};

struct ChiSquared : public Criteria<ChiSquared>, Cost
{
    static ChiSquared &instance()
    {
        static ChiSquared inst;
        return inst;
    }

    template<typename Iterator>
    static double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum{0};
        for (; first1 != last1; ++first1, ++first2)
        {
            auto m = *first1 - *first2;
            sum += m * m / *first1;
        }
        return sum;
    }

private:
    ChiSquared() = default;
};

struct Cosine : public Criteria<Cosine>, Score
{
    static Cosine &instance()
    {
        static Cosine inst;
        return inst;
    }

    template<typename Iterator>
    static inline double norm2( Iterator first, Iterator last )
    {
        double sum{0};
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


    template<typename Iterator>
    static inline double apply( Iterator firstActual, Iterator lastActual, Iterator firstAprrox, Iterator lastApprox )
    {
        assert( std::distance( firstActual, lastActual ) == std::distance( firstAprrox, lastApprox ));
        double sum{0};
        for (; firstActual != lastActual; ++firstActual, ++firstAprrox)
            sum += (*firstActual) * std::log((*firstActual + eps) / (*firstAprrox + eps));

        return sum;
    }

private:
    KullbackLeiblerDivergence() = default;
};

struct Measurement
{
    std::string id;
    double value;

    bool operator==( const Measurement &other ) const
    {
        return id == other.id && value == other.value;
    }

    bool operator>( const Measurement &other ) const
    {
        return value > other.value;
    }

    bool operator<( const Measurement &other ) const
    {
        return value < other.value;
    }

    bool operator>=( const Measurement &other ) const
    {
        return value >= other.value;
    }

    bool operator<=( const Measurement &other ) const
    {
        return value <= other.value;
    }
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


struct LeaderBoard
{
    virtual std::string bestMatch() const = 0;

    virtual long trueClusterRank() const = 0;

    virtual bool trueClusterFound() const = 0;
};

template<typename Criteria>
struct ClassificationCandidates : LeaderBoard
{
    using Queue = typename MatchSet<Criteria>::Queue;

    explicit ClassificationCandidates( std::string trueCluster, Queue q )
            : _trueCluster( std::move( trueCluster )), _bestMatches( std::move( q ))
    {}

    virtual std::string bestMatch() const override
    {
        return _bestMatches.top().id;
    }

    virtual long trueClusterRank() const override
    {
        return _bestMatches.findRank( [this]( const Measurement &m ) {
            return m.id == _trueCluster;
        } );
    }

    virtual bool trueClusterFound() const override
    {
        return _bestMatches.contains( [this]( const Measurement &m ) {
            return m.id == _trueCluster;
        } );
    }

    const std::string &trueCluster() const
    {
        return _trueCluster;
    }
private:
    std::string _trueCluster;
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
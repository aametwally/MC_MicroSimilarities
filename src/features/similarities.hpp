//
// Created by asem on 03/08/18.
//

#ifndef MARKOVIAN_FEATURES_DISTANCES_HPP
#define MARKOVIAN_FEATURES_DISTANCES_HPP

#include "common.hpp"


static double combineEuclidean( std::vector<double> &&c )
{
    double squares = std::accumulate( std::begin( c ), std::end( c ),
                                      double( 0 ), []( double acc, double val ) {
                return acc + val * val;
            } );
    return std::sqrt( squares );
}


static double combineManhattan( std::vector<double> &&c )
{
    return std::accumulate( std::begin( c ), std::end( c ),
                            double( 0 ));
}

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

    template<typename Container>
    static double combine( Container &&c )
    {
        return Derived::combine( c );
    }

private:
    Criteria() = default;

    friend Derived;
};

struct ChiSquared : public Criteria<ChiSquared>, Score
{
    static constexpr const char *label = "chi";

    template<typename Iterator>
    static double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            auto m = *it1 - *it2;
            sum += m * m / (*it1 + 1.0 / n);
        }
        return std::exp( -sum );
    }

    static constexpr auto combine = combineEuclidean;
private:
    ChiSquared() = default;
};

struct Cosine : public Criteria<Cosine>, Score
{
    static constexpr const char *label = "cos";

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

    static constexpr auto combine = combineManhattan;

private:
    Cosine() = default;
};

struct Intersection : public Criteria<Intersection>, Score
{
    static constexpr const char *label = "intersection";

    template<typename Iterator>
    static inline double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        auto n = std::distance( first1, last1 );

        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
            sum += std::min( *it1, *it2 );
        return sum / n;
    }

    static constexpr auto combine = combineManhattan;

private:
    Intersection() = default;
};

struct Gaussian : public Criteria<Gaussian>, Score
{
    static constexpr const char *label = "gaussian";

    template<typename Iterator>
    static inline double apply( Iterator first1, Iterator last1, Iterator first2, Iterator last2 )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        auto n = std::distance( first1, last1 );

        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            double diff = (*it1 - *it2);
            sum += diff * diff;
        }
        // Variance of ( U[0,1] - U`[0,1] ) = 1/15 - 1/36
        // See: https://stats.stackexchange.com/a/269492
        constexpr double var = 1.0 / 15 - 1.0 / 36;

        return std::exp( -sum / (var * n));
    }

    static constexpr auto combine = combineManhattan;

private:
    Gaussian() = default;
};

struct KullbackLeiblerDivergence : public Criteria<KullbackLeiblerDivergence>, Score
{
    static constexpr const char *label = "kl";

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
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
            sum += (*qIt) * std::log((*qIt + 1.0 / n) / (*pIt + 1.0 / n));

        return -sum;
    }

    static constexpr auto combine = combineEuclidean;
private:
    KullbackLeiblerDivergence() = default;
};

template<uint8_t Alpha>
struct DensityPowerDivergence : public Criteria<DensityPowerDivergence<Alpha>>, Score
{
    /**
     * @brief Density Power Divergence :https://hal.inria.fr/inria-00542337/document
     * @tparam Iterator
     * @param qFirst
     * @param qLast
     * @param pFirst
     * @param pLast
     * @return
     */
    template<typename Iterator, uint8_t Alpha_ = Alpha, typename std::enable_if<(Alpha_ > 0), void>::type * = nullptr>
    static inline double apply( Iterator qFirst, Iterator qLast, Iterator pFirst, Iterator pLast )
    {
        assert( std::distance( qFirst, qLast ) == std::distance( pFirst, pLast ));
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
        {
            double p = *pIt;
            double q = *qIt;

            double t1 = std::pow( q, 1 + Alpha );
            double t2 = (Alpha + 1) * q * std::pow( p, Alpha );
            double t3 = Alpha * std::pow( p, Alpha + 1 );
            sum += (t1 - t2 + t3);
        }
        sum *= 1.f / Alpha;
        return -sum;
    }

    static constexpr auto combine = combineEuclidean;
private:
    DensityPowerDivergence() = default;
};


struct DensityPowerDivergence1 : DensityPowerDivergence<1>
{
    static constexpr const char *label = "dpd1";
};

struct DensityPowerDivergence2 : DensityPowerDivergence<2>
{
    static constexpr const char *label = "dpd2";
};

struct DensityPowerDivergence3 : DensityPowerDivergence<3>
{
    static constexpr const char *label = "dpd3";
};

struct ItakuraSaitu : public Criteria<ItakuraSaitu>, Score
{
    static constexpr const char *label = "itakura-saitu";

    /**
     * @brief Itakura-Saitu Distance: https://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance
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
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
        {
            double u = (*qIt + eps) / (*pIt + eps);
            double v = (*qIt + 0.1 / n) / (*pIt + 0.1 / n);
            sum += u - std::log( v ) - 1;
        }

        return -sum;
    }

    static constexpr auto combine = combineEuclidean;
private:
    ItakuraSaitu() = default;
};


struct Bhattacharyya : public Criteria<Bhattacharyya>, Score
{
    static constexpr const char *label = "bhat";

    /**
     * @brief Bhattacharyya Distance: https://en.wikipedia.org/wiki/Bhattacharyya_distance
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
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
        {
            sum += std::sqrt( *qIt * *pIt + eps );
        }
        double distance = (-std::log( sum + eps ));
        return -distance;
    }

    static constexpr auto combine = combineEuclidean;
private:
    Bhattacharyya() = default;
};


struct Hellinger : public Criteria<Hellinger>, Score
{
    static constexpr const char *label = "hell";

    /**
     * @brief Hellinger Distance: https://en.wikipedia.org/wiki/Hellinger_distance
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
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
        {
            double u = std::sqrt( *qIt ) - std::sqrt( *pIt );
            sum += u * u;
        }
        constexpr double factor = 1.0 / std::sqrt( 2 );

        double distance = factor * std::sqrt( sum );
        return -distance;
    }

    static constexpr auto combine = combineEuclidean;
private:
    Hellinger() = default;
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

    double getValue() const
    {
        return _value;
    }

private:
    const std::string _label;
    const double _value;
};


template<typename T, typename Comp>
struct PriorityQueueFixed
{
    using Queue = std::set<T, Comp>;
    using ConstantIterator = typename Queue::const_iterator;

    explicit PriorityQueueFixed( const Comp &cmp, size_t kTop )
            : _kTop( kTop ), _q( cmp )
    {}

    explicit PriorityQueueFixed( size_t kTop )
            : _kTop( kTop )
    {}

    inline size_t size() const
    {
        return _q.size();
    }

    inline ConstantIterator begin() const
    {
        return _q.begin();
    }

    inline ConstantIterator end() const
    {
        return _q.end();
    }

    inline ConstantIterator cbegin() const
    {
        return _q.cbegin();
    }

    inline ConstantIterator cend() const
    {
        return _q.cend();
    }

    void forTopK( size_t k, const std::function<void( T )> &op )
    {
        auto lastIt = (size() < k) ?
                      std::crend( _q ) :
                      std::next( std::crbegin( _q ), k );

        std::for_each( std::crbegin( _q ), lastIt, op );
    }

    void forTopK( size_t k, const std::function<void( T, size_t )> &op )
    {
        auto lastIt = (size() < k) ?
                      std::crend( _q ) :
                      std::next( std::crbegin( _q ), k );

        size_t index = 0;
        for (auto it = _q.crbegin(); it != lastIt; ++it)
        {
            op( *it, index );
            ++index;
        }
    }

    const std::optional<std::reference_wrapper<const T >> top() const
    {
        if ( !empty())
            return *_q.crbegin();
        else return std::nullopt;
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

    auto insert( const T &val )
    {
        auto res = _q.insert( val );
        if ( _q.size() > _kTop )
            _q.erase( _q.begin());
        return res;
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


template<typename Criteria>
struct ClassificationCandidates
{
    using Queue = typename MatchSet<Criteria>::Queue;

    explicit ClassificationCandidates( std::string trueLabel, Queue q )
            : _trueLabel( std::move( trueLabel )), _bestMatches( std::move( q ))
    {}

    std::optional< std::string > bestMatch() const
    {
        if( auto match = _bestMatches.top() ; match )
            return match.value().get().getLabel();
        else return std::nullopt;
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
    KullbackLeiblerDiv,
    Intersection,
    Gaussian,
    DensityPowerDivergence1,
    DensityPowerDivergence2,
    DensityPowerDivergence3,
    ItakuraSaitu,
    Bhattacharyya,
    Hellinger
};

const std::map<std::string, CriteriaEnum> CriteriaLabels{
        {ChiSquared::label,                CriteriaEnum::ChiSquared},
        {Cosine::label,                    CriteriaEnum::Cosine},
        {KullbackLeiblerDivergence::label, CriteriaEnum::KullbackLeiblerDiv},
        {Intersection::label,              CriteriaEnum::Intersection},
        {Gaussian::label,                  CriteriaEnum::Gaussian},
        {DensityPowerDivergence1::label,   CriteriaEnum::DensityPowerDivergence1},
        {DensityPowerDivergence2::label,   CriteriaEnum::DensityPowerDivergence2},
        {DensityPowerDivergence3::label,   CriteriaEnum::DensityPowerDivergence3},
        {ItakuraSaitu::label,              CriteriaEnum::ItakuraSaitu},
        {Bhattacharyya::label,             CriteriaEnum::Bhattacharyya},
        {Hellinger::label,                 CriteriaEnum::Hellinger}
};

template<typename...>
struct CriteriaList
{
};
using SupportedCriteria = CriteriaList<ChiSquared, Cosine, KullbackLeiblerDivergence, Intersection, Gaussian,
        DensityPowerDivergence1, DensityPowerDivergence2, DensityPowerDivergence3, ItakuraSaitu, Bhattacharyya, Hellinger>;


#endif //MARKOVIAN_FEATURES_DISTANCES_HPP
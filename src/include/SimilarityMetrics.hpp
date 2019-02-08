//
// Created by asem on 03/08/18.
//

#ifndef MARKOVIAN_FEATURES_DISTANCES_HPP
#define MARKOVIAN_FEATURES_DISTANCES_HPP

#include "common.hpp"

static double combineEuclidean( std::vector<double> &&c )
{
    double squares = std::accumulate( std::begin( c ), std::end( c ),
                                      double( 0 ), [](
                    double acc,
                    double val
            ) {
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
    static constexpr bool cost = true;
    static constexpr bool score = !cost;
    static constexpr double worst = std::numeric_limits<double>::infinity();
    static constexpr double best = -worst;
};

struct Score
{
    static constexpr bool cost = false;
    static constexpr bool score = !cost;
    static constexpr double worst = -std::numeric_limits<double>::infinity();
    static constexpr double best = -worst;
};


template<typename Vector>
using MetricFunction = std::function<double(
        const Vector &,
        const Vector &
)>;

template<typename Vector>
using WeightedMetricFunction = std::function<double(
        const Vector &,
        const Vector &,
        const Vector &
)>;

template<typename Container>
struct SimilarityFunctor
{
    explicit SimilarityFunctor(
            const double eps,
            const double inf,
            const double best,
            const double worst,
            const bool cost,
            const bool score,
            const bool weighted,
            const MetricFunction<Container> metricFunction,
            const WeightedMetricFunction<Container> weightedMetricFunction,
            const std::function<bool(
                    double,
                    double
            )> closerThan
    )
            : inf( inf ), eps( eps ),
              best( best ), worst( worst ),
              cost( cost ), score( score ),
              weighted( weighted ),
              weightedMetricFunction( weightedMetricFunction ),
              metricFunction( metricFunction ),
              closerThan( closerThan )
    {}

    const double eps;
    const double inf;
    const double best;
    const double worst;
    const bool cost;
    const bool score;
    const bool weighted;
    const WeightedMetricFunction<Container> weightedMetricFunction;
    const MetricFunction<Container> metricFunction;
    const std::function<bool(
            double,
            double
    )> closerThan;

    inline double operator()(
            const Container &kernel1,
            const Container &kernel2,
            const Container &kernel3
    ) const
    {
        return weightedMetricFunction( kernel1, kernel2, kernel3 );
    }

    inline double operator()(
            const Container &kernel1,
            const Container &kernel2
    ) const
    {
        return metricFunction( kernel1, kernel2 );
    }
};

template<typename Derived, typename MetricKind, bool Weighted = false>
struct Criteria
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    static constexpr double inf = std::numeric_limits<double>::infinity();
    static constexpr bool cost = MetricKind::cost;
    static constexpr bool score = MetricKind::score;
    static constexpr double worst = MetricKind::worst;
    static constexpr double best = MetricKind::best;

    static constexpr bool weighted = Weighted;

    template<typename K = MetricKind, typename std::enable_if<std::is_same<K, Cost>::value, int>::type = 0>
    static inline auto metricCompare()
    {
        return std::less<double>();
    }

    template<typename K = MetricKind, typename std::enable_if<std::is_same<K, Score>::value, int>::type = 0>
    static inline auto metricCompare()
    {
        return std::greater<double>();
    }

    template<typename Container, bool W = Weighted, typename std::enable_if<W, int>::type = 0>
    static inline double measure(
            const Container &kernel1,
            const Container &kernel2,
            const Container &weights
    )
    {
        assert( std::all_of( std::cbegin( kernel1 ), std::cend( kernel1 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( kernel2 ), std::cend( kernel2 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( weights ), std::cend( weights ),
                             []( auto val ) { return !std::isnan( val ); } ));

        return Derived::apply( std::cbegin( kernel1 ), std::cend( kernel1 ),
                               std::cbegin( kernel2 ), std::cend( kernel2 ),
                               std::cbegin( weights ), std::cend( weights ));
    }

    template<typename Container, bool W = Weighted, typename std::enable_if<!W, int>::type = 0>
    static inline double measure(
            const Container &kernel1,
            const Container &kernel2,
            const Container &
    )
    {
        assert( std::all_of( std::cbegin( kernel1 ), std::cend( kernel1 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( kernel2 ), std::cend( kernel2 ),
                             []( auto val ) { return !std::isnan( val ); } ));

        return measure( kernel1, kernel2 );
    }

    template<typename Container>
    static inline double measure(
            const Container &kernel1,
            const Container &kernel2
    )
    {
        assert( std::all_of( std::cbegin( kernel1 ), std::cend( kernel1 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( kernel2 ), std::cend( kernel2 ),
                             []( auto val ) { return !std::isnan( val ); } ));

        return Derived::apply( std::cbegin( kernel1 ), std::cend( kernel1 ),
                               std::cbegin( kernel2 ), std::cend( kernel2 ));
    }

    template<typename Container>
    static inline double measureUnweighted(
            const Container &kernel1,
            const Container &kernel2
    )
    {
        return measure( kernel1, kernel2 );
    }

    template<typename Container>
    static inline double measureWeighted(
            const Container &kernel1,
            const Container &kernel2,
            const Container &kernel3
    )
    {
        return measure( kernel1, kernel2, kernel3 );
    }

    template<typename Container>
    static SimilarityFunctor<Container> similarityFunctor()
    {
        return SimilarityFunctor<Container>( eps, inf, best, worst, cost, score, weighted,
                                             [](
                                                     const Container &kernel1,
                                                     const Container &kernel2
                                             ) -> double {
                                                 return measure( kernel1, kernel2 );
                                             },
                                             [](
                                                     const Container &kernel1,
                                                     const Container &kernel2,
                                                     const Container &kernel3
                                             ) -> double {
                                                 return measure( kernel1, kernel2, kernel3 );
                                             },
                                             metricCompare());
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

struct Euclidean : public Criteria<Euclidean, Cost>
{
    static constexpr const char *label = "euclidean";

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            auto m = *it1 - *it2;
            sum += m * m;
        }
        return sum;
    }

    static constexpr auto combine = combineEuclidean;
private:
    Euclidean() = default;
};

struct Mahalanobis : public Criteria<Mahalanobis, Cost, true>
{
    static constexpr const char *label = "mahalanobis";

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2,
            Iterator weightsFirst,
            Iterator weightsLast
    )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 )
                && std::distance( first1, last1 ) == std::distance( weightsFirst, weightsLast ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2, weIt = weightsFirst; it1 != last1; ++it1, ++it2, ++weIt)
        {
            auto m = (*it1 - *it2) / (*weIt + eps);
            sum += m * m;
        }
        return sum;
    }

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        return Euclidean::apply( first1, last1, first2, last2 );
    }

    static constexpr auto combine = combineEuclidean;
private:
    Mahalanobis() = default;
};


struct Manhattan : public Criteria<Manhattan, Cost>
{
    static constexpr const char *label = "manhattan";

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            auto m = *it1 - *it2;
            sum += (m < 0) ? -m : m;
        }
        return sum;
    }

    static constexpr auto combine = combineEuclidean;
private:
    Manhattan() = default;
};

struct ChiSquared : public Criteria<ChiSquared, Score>
{
    static constexpr const char *label = "chi";

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
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

struct Cosine : public Criteria<Cosine, Score>
{
    static constexpr const char *label = "cos";

    template<typename Iterator>
    static inline double norm2(
            Iterator first,
            Iterator last
    )
    {
        double sum = 0;
        for (auto it = first; it != last; ++it)
        {
            sum += (*it) * (*it);
        }
        return std::sqrt( sum );
    }

    template<typename Iterator>
    static inline double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum = 0;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
        {
            sum += (*it1) * (*it2);
        }
        return sum / (norm2( first1, last1 ) * norm2( first2, last2 ) + eps);
    }

    static constexpr auto combine = combineManhattan;

private:
    Cosine() = default;
};


struct DWCosine : public Criteria<DWCosine, Score, true>
{
    static constexpr const char *label = "dwcos";

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2,
            Iterator weightsFirst,
            Iterator weightsLast
    )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 )
                && std::distance( first1, last1 ) == std::distance( weightsFirst, weightsLast ));

        auto cos = Cosine::apply( first1, last1, first2, last2 );
        auto dist = Mahalanobis::apply( first1, last1, first2, last2, weightsFirst, weightsLast );
        return cos / (cos + dist * dist);
    }

    template<typename Iterator>
    static double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        auto n = std::distance( first1, last1 );
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));

        auto cos = Cosine::apply( first1, last1, first2, last2 );
        auto dist = Mahalanobis::apply( first1, last1, first2, last2 );
        return cos / (cos + dist * dist);
    }

    template<typename Container>
    static double measure(
            const Container &kernel1,
            const Container &kernel2,
            const Container &weights
    )
    {
        assert( std::all_of( std::cbegin( kernel1 ), std::cend( kernel1 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( kernel2 ), std::cend( kernel2 ),
                             []( auto val ) { return !std::isnan( val ); } ));
        assert( std::all_of( std::cbegin( weights ), std::cend( weights ),
                             []( auto val ) { return !std::isnan( val ); } ));

        return apply( std::cbegin( kernel1 ), std::cend( kernel1 ),
                      std::cbegin( kernel2 ), std::cend( kernel2 ),
                      std::cbegin( weights ), std::cend( weights ));

    }

    static constexpr auto combine = combineEuclidean;
private:
    DWCosine() = default;
};


struct Dot : public Criteria<Dot, Score>
{
    static constexpr const char *label = "dot";

    template<typename Iterator>
    static inline double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        double sum{0};
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
            sum += (*it1) * (*it2);
        return sum;
    }

    static constexpr auto combine = combineManhattan;

private:
    Dot() = default;
};


struct Intersection : public Criteria<Intersection, Score>
{
    static constexpr const char *label = "intersection";

    template<typename Iterator>
    static inline double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
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


struct MaxIntersection : public Criteria<MaxIntersection, Score>
{
    static constexpr const char *label = "max_intersection";

    template<typename Iterator>
    static inline double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
    {
        assert( std::distance( first1, last1 ) == std::distance( first2, last2 ));
        auto n = std::distance( first1, last1 );

        double max = -inf;
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2)
            max = std::max( max, std::min( *it1, *it2 ));
        return max;
    }

    static constexpr auto combine = combineManhattan;

private:
    MaxIntersection() = default;
};

struct Gaussian : public Criteria<Gaussian, Score>
{
    static constexpr const char *label = "gaussian";

    template<typename Iterator>
    static inline double apply(
            Iterator first1,
            Iterator last1,
            Iterator first2,
            Iterator last2
    )
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

struct KullbackLeiblerDivergence : public Criteria<KullbackLeiblerDivergence, Score>
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
    static inline double apply(
            Iterator qFirst,
            Iterator qLast,
            Iterator pFirst,
            Iterator pLast
    )
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
struct DensityPowerDivergence : public Criteria<DensityPowerDivergence<Alpha>, Score>
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
    template<typename Iterator, uint8_t Alpha_ = Alpha, typename std::enable_if<(Alpha_ >
                                                                                 0), void>::type * = nullptr>
    static inline double apply(
            Iterator qFirst,
            Iterator qLast,
            Iterator pFirst,
            Iterator pLast
    )
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

struct ItakuraSaitu : public Criteria<ItakuraSaitu, Score>
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
    static inline double apply(
            Iterator qFirst,
            Iterator qLast,
            Iterator pFirst,
            Iterator pLast
    )
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


struct Bhattacharyya : public Criteria<Bhattacharyya, Score>
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
    static inline double apply(
            Iterator qFirst,
            Iterator qLast,
            Iterator pFirst,
            Iterator pLast
    )
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


struct Hellinger : public Criteria<Hellinger, Score>
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
    static inline double apply(
            Iterator qFirst,
            Iterator qLast,
            Iterator pFirst,
            Iterator pLast
    )
    {
        assert( std::distance( qFirst, qLast ) == std::distance( pFirst, pLast ));
        auto n = std::distance( qFirst, qLast );
        double sum = 0;
        for (auto qIt = qFirst, pIt = pFirst; qIt != qLast; ++qIt, ++pIt)
        {
            double u = std::sqrt( *qIt ) - std::sqrt( *pIt );
            sum += u * u;
        }
        static constexpr double factor = 1.0 / std::sqrt( 2 );

        double distance = factor * std::sqrt( sum );
        return -distance;
    }

    static constexpr auto combine = combineEuclidean;
private:
    Hellinger() = default;
};

template<typename Label = std::string_view>
struct ValuedLabel
{
    ValuedLabel(
            Label label,
            double val
    )
            : _label( label ), _value( val )
    {}

    bool operator==( const ValuedLabel &other ) const
    {
        return _label == other._label && _value == other._value;
    }

    bool operator>( const ValuedLabel &other ) const
    {
        return _value > other._value;
    }

    bool operator<( const ValuedLabel &other ) const
    {
        return _value < other._value;
    }

    bool operator>=( const ValuedLabel &other ) const
    {
        return _value >= other._value;
    }

    bool operator<=( const ValuedLabel &other ) const
    {
        return _value <= other._value;
    }

    Label label() const
    {
        return _label;
    }

    double value() const
    {
        return _value;
    }

private:
    Label _label;
    const double _value;
};


template<typename Label, typename Comp>
struct PriorityQueueFixed
{
    using Queue = std::multiset<ValuedLabel<Label>, Comp>;
    using ConstantIterator = typename Queue::const_iterator;
    using ValueType = ValuedLabel<Label>;

    explicit PriorityQueueFixed(
            const Comp &cmp,
            size_t kTop
    )
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


    void forTopK(
            size_t k,
            const std::function<void( const ValueType & )> &op
    ) const
    {
        if ( k == 0 )
            k = _q.size();

        auto lastIt = (size() < k) ?
                      std::crend( _q ) :
                      std::next( std::crbegin( _q ), k );

        std::for_each( std::crbegin( _q ), lastIt, op );
    }

    void forTopK(
            size_t k,
            const std::function<void(
                    const ValueType &,
                    size_t
            )> &op
    ) const
    {
        if ( k == 0 )
            k = _q.size();

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

    std::map<Label, double> toMap() const
    {
        std::map<Label, double> m;
        for (auto &vl : _q)
            m[vl.label()] = vl.value();
        return m;
    }

    const std::optional<std::reference_wrapper<const ValueType >> top() const
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
        auto res = _q.emplace( std::forward<Args>( args )... );
        if ( _q.size() > _kTop )
            _q.erase( _q.begin());
        return res;
    };

    auto insert( const ValueType &val )
    {
        auto res = _q.insert( val );
        if ( _q.size() > _kTop )
            _q.erase( _q.begin());
        return res;
    }

    auto insert( ValueType &&val )
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

    void findOrInsert(
            const Label &label,
            double value = 0
    )
    {
        auto it = std::find_if( _q.crbegin(), _q.crend(), [&]( const ValueType &item ) {
            return item.label() == label;
        } );
        if ( it == _q.crend())
        {
            emplace( label, value );
        }
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
    template<typename Label>
    using Queue = PriorityQueueFixed<Label, std::greater<> >;
};

template<typename T>
struct MatchSet<T, typename std::enable_if<std::is_base_of<Score, T>::value>::type>
{
    template<typename Label>
    using Queue = PriorityQueueFixed<Label, std::less<>>;
};


using ScoredLabels = typename MatchSet<Score>::Queue<std::string_view>;
using ScoredIndices =  typename MatchSet<Score>::Queue<size_t>;
using PenalizedLabels = typename MatchSet<Cost>::Queue<std::string_view>;
using PenalizedIndices = typename MatchSet<Cost>::Queue<size_t>;

template<typename Criteria>
struct ClassificationCandidates
{
    using M = ValuedLabel<std::string_view>;

    using Queue =  typename MatchSet<Criteria>::template Queue<std::string_view>;

    explicit ClassificationCandidates(
            std::string_view trueLabel,
            Queue q
    )
            : _trueLabel( trueLabel ), _bestMatches( std::move( q ))
    {}

    std::optional<std::string_view> bestMatch() const
    {
        if ( auto match = _bestMatches.top(); match )
            return match.value().get().getLabel();
        else return std::nullopt;
    }

    long trueClusterRank() const
    {
        return _bestMatches.findRank( [this]( const M &m ) {
            return m.label() == _trueLabel;
        } );
    }

    bool trueClusterFound() const
    {
        return _bestMatches.contains( [this]( const M &m ) {
            return m.label() == _trueLabel;
        } );
    }

    const std::string_view &trueCluster() const
    {
        return _trueLabel;
    }

private:
    std::string_view _trueLabel;
    Queue _bestMatches;
};


enum class CriteriaEnum
{
    ChiSquared,
    Cosine,
    Euclidean,
    Mahalanobis,
    DWCosine,
    Dot,
    KullbackLeiblerDiv,
    Intersection,
    Gaussian,
    DensityPowerDivergence1,
    DensityPowerDivergence2,
    DensityPowerDivergence3,
    ItakuraSaitu,
    Bhattacharyya,
    Hellinger,
    MaxIntersection
};

const std::map<std::string, CriteriaEnum> CriteriaLabels{
        {ChiSquared::label,                CriteriaEnum::ChiSquared},
        {Cosine::label,                    CriteriaEnum::Cosine},
        {Euclidean::label,                 CriteriaEnum::Euclidean},
        {Mahalanobis::label,               CriteriaEnum::Mahalanobis},
        {DWCosine::label,                  CriteriaEnum::DWCosine},
        {Dot::label,                       CriteriaEnum::Dot},
        {KullbackLeiblerDivergence::label, CriteriaEnum::KullbackLeiblerDiv},
        {Intersection::label,              CriteriaEnum::Intersection},
        {Gaussian::label,                  CriteriaEnum::Gaussian},
        {DensityPowerDivergence1::label,   CriteriaEnum::DensityPowerDivergence1},
        {DensityPowerDivergence2::label,   CriteriaEnum::DensityPowerDivergence2},
        {DensityPowerDivergence3::label,   CriteriaEnum::DensityPowerDivergence3},
        {ItakuraSaitu::label,              CriteriaEnum::ItakuraSaitu},
        {Bhattacharyya::label,             CriteriaEnum::Bhattacharyya},
        {Hellinger::label,                 CriteriaEnum::Hellinger},
        {MaxIntersection::label,           CriteriaEnum::MaxIntersection}
};

#endif //MARKOVIAN_FEATURES_DISTANCES_HPP
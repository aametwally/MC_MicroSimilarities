//
// Created by asem on 03/08/18.
//

#ifndef MARKOVIAN_FEATURES_DISTANCES_HPP
#define MARKOVIAN_FEATURES_DISTANCES_HPP


#include "common.hpp"
#include <mutex>

struct Cost{};
struct Score{};

enum class CriteriaEnum {
    ChiSquared
};

const std::map< std::string , CriteriaEnum  > CriteriaLabels{
        {"chi" , CriteriaEnum ::ChiSquared }
};

template<typename Derived>
struct Criteria
{
    template<typename Iterator>
    static double measure(Iterator first1, Iterator last1, Iterator first2, Iterator last2)
    {
        Derived &inst = Derived::instance();
        return inst.apply(first1, last1, first2, last2);
    }

private:
    Criteria() = default;

    friend Derived;
};

struct ChiSquared : public Criteria<ChiSquared> , Cost
{
    static ChiSquared &instance()
    {
        static ChiSquared inst;
        return inst;
    }

    template<typename Iterator>
    static double apply(Iterator first1, Iterator last1, Iterator first2, Iterator last2)
    {
        assert(std::distance(first1, last1) == std::distance(first2, last2));
        double sum{0};
        for (; first1 != last1; ++first1, ++first2) {
            auto m = *first1 - *first2;
            sum += m * m / *first1;
        }
        return sum;
    }

private:
    ChiSquared() = default;
};


struct Measurement
{
    std::string id;
    double value;
    bool operator>(const Measurement &other) const
    {
        return value > other.value;
    }
};


template<typename T, typename Enable = void>
struct MatchSet;

template<typename T>
struct MatchSet<T, typename std::enable_if<std::is_base_of<Cost,T>::value>::type>
{
    using Queue =std::set< Measurement, std::greater< > >;
};

template<typename T>
struct MatchSet<T, typename std::enable_if<std::is_base_of<Score,T>::value>::type>
{
    using Queue = std::set<Measurement, std::less< >>;
};

//using MatchSetByDistance = ;
//using MatchSetByScore = ;


template<typename Criteria>
struct ClassificationCandidates
{
    using Queue = typename MatchSet< Criteria>::Queue;
    std::string trueCluster;

    Queue bestMatches;

    std::string bestMatch() const
    {
        return bestMatches.crbegin()->id;
    }

    long trueClusterRank() const
    {
        auto trueClusterIt = std::find_if(bestMatches.crbegin(), bestMatches.crend(),
                                          [this](const typename Queue::key_type &m) {
                                              return m.id == trueCluster;
                                          });

        if (trueClusterIt == bestMatches.crend())
            return -1;
        else return std::distance(bestMatches.crbegin(), trueClusterIt);
    }

    bool trueClusterFound() const
    {
        return std::find_if(bestMatches.cbegin(), bestMatches.cend(),
                            [this](const typename Queue::key_type &m) {
                                return m.id == trueCluster;
                            }) != bestMatches.cend();
    }
};


#endif //MARKOVIAN_FEATURES_DISTANCES_HPP
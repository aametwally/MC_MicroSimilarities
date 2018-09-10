//
// Created by asem on 09/09/18.
//

#ifndef MARKOVIAN_FEATURES_HOMCDEFS_HPP
#define MARKOVIAN_FEATURES_HOMCDEFS_HPP

#include "common.hpp"
#include "MarkovChains.hpp"

namespace MC
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double inf = std::numeric_limits<double>::infinity();

    using Selection = std::unordered_map<Order, std::set<HistogramID >>;
    using SelectionFlat = std::unordered_map<Order, std::vector<HistogramID >>;
    using SelectionOrdered = std::map<Order, std::set<HistogramID>>;

}
#endif //MARKOVIAN_FEATURES_HOMCDEFS_HPP

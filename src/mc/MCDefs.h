//
// Created by asem on 14/09/18.
//

#ifndef MARKOVIAN_FEATURES_MCDEFS_H
#define MARKOVIAN_FEATURES_MCDEFS_H

#include "common.hpp"

namespace MC {
using Order = int8_t;
using HistogramID = size_t;

constexpr std::string_view unclassified = std::string_view();

constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double pi = 3.14159265358979323846;


struct KernelIdentifier
{
    explicit KernelIdentifier(
            Order o,
            HistogramID i
    ) : order( o ), id( i )
    {}

    Order order;
    HistogramID id;
};
}

#endif //MARKOVIAN_FEATURES_MCDEFS_H

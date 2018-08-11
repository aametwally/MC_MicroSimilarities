//
// Created by asem on 09/08/18.
//

#ifndef MARKOVIAN_FEATURES_SERIES_HPP
#define MARKOVIAN_FEATURES_SERIES_HPP

#include <optional>
#include <functional>

#include "common.hpp"

template<typename T, typename Derived1, typename Derived2 = int>
class Series
{
public:
    inline void popTerm()
    {
        auto &derived = static_cast<Derived1 &>(*this);
        return derived.next();
    }

    inline T sum() const
    {
        auto &derived = static_cast<const Derived1 &>(*this);
        if ( !isEmpty())
        {
            Derived1 copy = derived;
            T sum = T();
            while (!copy.isEmpty())
            {
                sum += copy.currentTerm().value_or( 0 );
                copy.popTerm();
            }
            return sum;
        } else return T();
    }

    inline T dot( Derived1 other ) const
    {
        assert( length() == other.length());
        auto &derived = static_cast<const Derived1 &>(*this);
        T sum = T( 0 );
        if ( !isEmpty())
        {
            Derived1 copy = derived;
            while (!copy.isEmpty())
                sum += copy.currentTerm().value_or( 0 ) * other.currentTerm().value_or( 0 );

            copy.popTerm();
            other.popTerm();
        }
        return sum;
    }

    template<typename OtherSeries>
    inline T dot( OtherSeries other ) const
    {
        assert( length() == other.length());
        auto &derived = static_cast<const Derived1 &>(*this);
        T sum = T( 0 );
        if ( !isEmpty())
        {
            Derived1 copy = derived;
            while (!copy.isEmpty())
                sum += copy.currentTerm().value_or( 0 ) * other.currentTerm().value_or( 0 );

            copy.popTerm();
            other.popTerm();
        }
        return sum;
    }

    template<typename BinaryOp>
    static double sum( Derived1 s1, Derived1 s2, BinaryOp op,
                       std::optional<T> &&defaultTerm = std::nullopt )
    {
        assert( s1.length() == s2.length());
        double sum = 0;
        if ( !s1.isEmpty())
        {
            while (!s1.isEmpty() && !s2.isEmpty())
            {
                if ( s1.currentTerm())
                {
                    if ( s2.currentTerm())
                        sum += op( s1.currentTerm().value(), s2.currentTerm().value());
                    else if ( defaultTerm )
                        sum += op( s1.currentTerm().value(), s2.currentTerm().value_or( defaultTerm.value()));
                }
                s1.popTerm();
                s2.popTerm();
            }
        }
        return sum;
    }

    template<typename BinaryOp>
    static std::vector<double>
    apply( Derived1 s1, Derived1 s2, BinaryOp op,
           std::optional<T> defaultTerm = std::nullopt )
    {
        assert( s1.length() == s2.length());
        std::vector<double> result;
        if ( !s1.isEmpty())
        {
            while (!s1.isEmpty() && !s2.isEmpty())
            {
                if ( s1.currentTerm())
                {
                    if ( s2.currentTerm())
                        result.push_back( op( s1.currentTerm().value(), s2.currentTerm().value()));
                    else if ( defaultTerm )
                        result.push_back(
                                op( s1.currentTerm().value(), s2.currentTerm().value_or( defaultTerm.value())));
                }
                s1.popTerm();
                s2.popTerm();
            }
        }
        return result;
    }

    template<typename WeightsSeries, typename BinaryOp>
    static double weightedSum( Derived1 s1, Derived1 s2,
                               WeightsSeries w, BinaryOp op,
                               std::optional<T> defaultTerm = std::nullopt )
    {
        assert( s1.length() == s2.length() && s2.length() == w.length());
        double sum = 0;
        if ( !s1.isEmpty())
        {
            while (!s1.isEmpty() && !s2.isEmpty() && !w.isEmpty())
            {
                if ( s1.currentTerm())
                {
                    if ( s2.currentTerm())
                        sum += op( s1.currentTerm().value(), s2.currentTerm().value()) * w.currentTerm().value_or( 1 );
                    else if ( defaultTerm )
                        sum += op( s1.currentTerm().value(), s2.currentTerm().value_or( defaultTerm.value())) *
                               w.currentTerm().value_or( 1 );
                }
                s1.popTerm();
                s2.popTerm();
                w.popTerm();
            }
        }
        return sum;
    }

    inline T product() const
    {
        auto &derived = static_cast<const Derived1 &>(*this);
        if ( !isEmpty())
        {
            auto copy = derived;
            std::function<T( T )> multiplier;
            multiplier = [&]( T prod ) {
                if ( copy.isEmpty()) return prod;
                else
                {
                    T t = copy.currentTerm().value_or( 1 );
                    copy.popTerm();
                    return multiplier( prod * t );
                }
            };
            return multiplier( copy.currentTerm().value_or( 1 ));
        } else return T( 1 );
    }

    inline bool isEmpty() const
    {
        auto &derived = static_cast<const Derived1 &>(*this);
        return derived.isEmpty();
    }

    inline Series &operator++()
    {
        auto &derived = static_cast<Derived1 &>(*this);
        derived.popTerm();
        return derived;
    }

    inline constexpr size_t length() const
    {
        auto &derived = static_cast<const Derived1 &>(*this);
        return derived.length();
    }

    inline std::optional<std::reference_wrapper<const T >> currentTerm() const
    {
        auto &derived = static_cast<const Derived1 &>(*this);
        return derived.currentTerm();
    }

private:

    Series()
    {};
    friend Derived1;
    friend Derived2;
};

#endif //MARKOVIAN_FEATURES_SERIES_HPP

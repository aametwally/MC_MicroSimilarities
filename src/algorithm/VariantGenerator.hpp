//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_VARIANTGENERATOR_HPP
#define MARKOVIAN_FEATURES_VARIANTGENERATOR_HPP

#include <type_traits>
#include <variant>
#include <tuple>
#include <typeinfo>
#include <cxxabi.h>

template<template<typename...> class C, typename ... Ts>
constexpr std::tuple<C<Ts...>> tupleExpand( std::tuple<Ts...> const & );

template<template<typename...> class C, typename ... Ts,
        template<typename...> class C0, typename ... Ls,
        typename ... Cs>
constexpr auto tupleExpand( std::tuple<Ts...> const &, C0<Ls...> const &,
Cs const &... cs )
-> decltype( std::tuple_cat(
        tupleExpand<C>( std::declval<std::tuple<Ts..., Ls>>(), cs... )... ));

template<typename ... Ts>
constexpr std::variant<Ts...> tupleToVariant( std::tuple<Ts...> const & );

template<template<typename...> class C, typename ... Ts>
struct MakeVariant
{
    using type = decltype( tupleToVariant( std::declval<
            decltype( tupleExpand<C>( std::declval<std::tuple<>>(),
                                      std::declval<Ts>()... ))>()));
};

template<template<typename...> class C, typename ... Ts>
using MakeVariantType = typename MakeVariant<C, Ts...>::type;


#endif //MARKOVIAN_FEATURES_VARIANTGENERATOR_HPP

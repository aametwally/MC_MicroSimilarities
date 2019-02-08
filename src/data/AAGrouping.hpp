//
// Created by asem on 04/08/18.
//

#ifndef MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP
#define MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

#include "common.hpp"
#include "LUT.hpp"

enum class AminoAcidGroupingEnum
{
    NoGrouping22,
    OFER15,
    OFER8,
    DIAMOND11
};

const std::map<std::string, AminoAcidGroupingEnum> GroupingLabels{
        {"nogrouping", AminoAcidGroupingEnum::NoGrouping22},
        {"ofer15",     AminoAcidGroupingEnum::OFER15},
        {"ofer8",      AminoAcidGroupingEnum::OFER8},
        {"diamond11",  AminoAcidGroupingEnum::DIAMOND11}
};

template<size_t N>
constexpr std::array<char, N>
reducedAlphabet()
{
    std::array<char, N> newAlphabet{};
    for (auto i = 0; i < newAlphabet.size(); ++i)
        newAlphabet.at( i ) = 'A' + i;
    return newAlphabet;
};

template<size_t N>
constexpr std::array<int16_t, 256>
reducedAlphabetIds( const std::array<const char *, N> &alphabetGrouping )
{
    std::array<int16_t, 256> ids{};
    for (size_t i = 0; i < 256; ++i)
        ids[i] = -1;

    int16_t i = 0;
    for (const auto &aaGroup : alphabetGrouping)
    {
        size_t j = 0;
        while (aaGroup[j] != '\0')
        {
            ids.at( size_t( aaGroup[j++] )) = int16_t( i );
        }
        ++i;
    }
    return ids;
};

constexpr int16_t AA_COUNT20 = 20;
constexpr int16_t AA_COUNT22 = 22;

constexpr std::array<const char *, AA_COUNT22> AAGrouping_NOGROUPING_Array = {
        "A", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "Y"};

constexpr std::array<const char *, 15> AAGrouping_OFER15_Array = {
        "KR", "E", "D", "Q", "N", "C", "G", "H",
        "ILVM", "F", "Y",
        "W", "P", "ST", "A"};

constexpr std::array<const char *, 8> AAGrouping_OFER8_Array = {
        "KRH", "ED", "C", "G", "AILVM", "FYW", "P", "NQST"};

constexpr std::array<const char *, 11> AAGrouping_DIAMOND11_Array = {
        "KREDQN", "C", "G", "H", "ILV", "M", "F", "Y", "W", "P", "STA"};

template<size_t N, const std::array<const char *, N> &GroupingArray>
struct AAGrouping
{
    static constexpr const std::array<const char *, N> &Grouping = GroupingArray;
    static constexpr size_t StatesN = N;
};

constexpr size_t COUNT_NOGROUPING22 = AA_COUNT22;
constexpr size_t COUNT_OFER15 = 15;
constexpr size_t COUNT_OFER8 = 8;
constexpr size_t COUNT_DIAMOND11 = 11;

using AAGrouping_NOGROUPING22 = AAGrouping<COUNT_NOGROUPING22, AAGrouping_NOGROUPING_Array>;
using AAGrouping_OFER15 = AAGrouping<COUNT_OFER15, AAGrouping_OFER15_Array>;
using AAGrouping_OFER8 = AAGrouping<COUNT_OFER8, AAGrouping_OFER8_Array>;
using AAGrouping_DIAMOND11 = AAGrouping<COUNT_DIAMOND11, AAGrouping_DIAMOND11_Array>;

template<typename...>
struct AAGroupingList
{
};
using SupportedAAGrouping  = AAGroupingList<
        AAGrouping_NOGROUPING22,
//        AAGrouping_OFER8 ,
//        AAGrouping_DIAMOND11  ,
        AAGrouping_OFER15
>;

constexpr std::string_view AMINO_ACIDS = "ACDEFGHIKLMNOPQRSTUVWY";
constexpr std::string_view AMINO_ACIDS20 = "ACDEFGHIKLMNPQRSTVWY";
constexpr int8_t POLYMORPHIC_OFFSET = 30;
const std::map<char, std::string> POLYMORPHIC_AA{
        {'X', std::string( AMINO_ACIDS20 )},
        {'B', "ND"},
        {'Z', "EQ"},
        {'J', "LI"}
};

const std::map<char, char> POLYMORPHIC_AA_ENCODE{
        {'X', 'A' + POLYMORPHIC_OFFSET},
        {'B', 'A' + POLYMORPHIC_OFFSET + 1},
        {'Z', 'A' + POLYMORPHIC_OFFSET + 2},
        {'J', 'A' + POLYMORPHIC_OFFSET + 3}
};

const std::map<char, char> POLYMORPHIC_AA_DECODE = []() {
    std::map<char, char> decode;
    for (auto[a, b] : POLYMORPHIC_AA_ENCODE)
        decode[b] = a;
    return decode;
}();

auto AA_ORDER_INDEX( int16_t defaultValue )
{
    return LUT<char, int16_t>::makeLUT( [defaultValue]( char a ) -> int16_t {
        auto it = std::find( AMINO_ACIDS.cbegin(), AMINO_ACIDS.cend(), a );
        if ( it == AMINO_ACIDS.cend())
            return defaultValue;
        return static_cast<int16_t>(std::distance( AMINO_ACIDS.cbegin(), it ));
    } );
};

/**
 * Some physicochemical properties.
 * http://prowl.rockefeller.edu/aainfo/struct.htm
 * Amino Acids Mass **/
/**Transfer energy (Nozaki-Tanford, 1971)**/
/** Hydrophilicity value (Hopp-Woods, 1981) **/
/**Packing density (Tsai et al 1999)**/
/** Nature of the accessible and buried surfaces (Chothia, 1976) **/
/** Shape and surface features (Prabhakaran-Ponnuswamy, 1982) **/
/** Alpha-helix indices (Geisow-Roberts, 1980) **/
/** Helix-coil equilibrium constant (Finkelstein-Ptitsyn, 1977) **/

#endif //MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

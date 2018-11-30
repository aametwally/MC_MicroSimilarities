//
// Created by asem on 04/08/18.
//

#ifndef MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP
#define MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

#include "common.hpp"

enum class AminoAcidGroupingEnum
{
    NoGrouping20,
    OFER15,
    OFER8,
    DIAMOND11
};

const std::map<std::string, AminoAcidGroupingEnum> GroupingLabels{
        { "nogrouping", AminoAcidGroupingEnum::NoGrouping20 },
        { "ofer15",     AminoAcidGroupingEnum::OFER15 },
        { "ofer8",      AminoAcidGroupingEnum::OFER8 },
        { "diamond11",  AminoAcidGroupingEnum::DIAMOND11 }
};

template<size_t N>
constexpr std::array<char, 256> alphabetIds( const std::array<char, N> &alphabet )
{
    std::array<char, 256> ids{};
    int i = 0;
    for ( char aa : alphabet )
        ids.at( size_t( aa )) = char( i++ );
    return ids;
};

template<size_t N>
constexpr std::array<char, N>
reducedAlphabet()
{
    std::array<char, N> newAlphabet{};
    for ( auto i = 0; i < newAlphabet.size(); ++i )
        newAlphabet.at( i ) = 'A' + i;
    return newAlphabet;
};

template<size_t N>
constexpr std::array<char, 256>
reducedAlphabetIds( const std::array<const char *, N> &alphabetGrouping )
{
    std::array<char, 256> ids{};
    int i = 0;
    for ( const auto &aaGroup : alphabetGrouping )
    {
        size_t j = 0;
        while ( aaGroup[j] != '\0' )
        {
            ids.at( size_t( aaGroup[j++] )) = char( i );
        }
        ++i;
    }
    return ids;
};


constexpr std::array<const char *, 20> AAGrouping_NOGROUPING_Array = { "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                                                                       "M", "N", "P", "Q", "R", "S",
                                                                       "T", "V", "W", "Y" };

constexpr std::array<const char *, 15> AAGrouping_OFER15_Array = { "KR", "E", "D", "Q", "N", "C", "G", "H", "ILVM", "F",
                                                                   "Y",
                                                                   "W", "P", "ST", "A" };
constexpr std::array<const char *, 8> AAGrouping_OFER8_Array = { "KRH", "ED", "C", "G", "AILVM", "FYW", "P", "NQST" };

constexpr std::array<const char *, 11> AAGrouping_DIAMOND11_Array = { "KREDQN", "C", "G", "H", "ILV", "M", "F", "Y",
                                                                      "W",
                                                                      "P",
                                                                      "STA" };

template<size_t N, const std::array<const char *, N> &GroupingArray>
struct AAGrouping
{
    static constexpr const std::array<const char *, N> &Grouping = GroupingArray;
    static constexpr size_t StatesN = N;
};

using AAGrouping_NOGROUPING20 = AAGrouping<20, AAGrouping_NOGROUPING_Array>;
using AAGrouping_OFER15 = AAGrouping<15, AAGrouping_OFER15_Array>;
using AAGrouping_OFER8 = AAGrouping<8, AAGrouping_OFER8_Array>;
using AAGrouping_DIAMOND11 = AAGrouping<11, AAGrouping_DIAMOND11_Array>;

template<typename...>
struct AAGroupingList
{
};
using SupportedAAGrouping  = AAGroupingList<
        AAGrouping_NOGROUPING20,
//        AAGrouping_OFER8 ,
//        AAGrouping_DIAMOND11  ,
        AAGrouping_OFER15
>;

/**
 * http://prowl.rockefeller.edu/aainfo/struct.htm
 * Amino Acids Mass **/
const std::map<char, double> AA_MASS = {
        { 'A', 71.079 },
        { 'R', 156.188 },
        { 'N', 114.104 },
        { 'D', 115.089 },
        { 'C', 103.145 },
        { 'Q', 128.131 },
        { 'E', 129.116 },
        { 'G', 57.052 },
        { 'H', 137.141 },
        { 'I', 113.160 },
        { 'L', 113.160 },
        { 'K', 128.17 },
        { 'M', 131.199 },
        { 'F', 147.177 },
        { 'P', 97.117 },
        { 'S', 87.078 },
        { 'T', 101.105 },
        { 'W', 186.213 },
        { 'Y', 163.176 },
        { 'V', 99.133 }
};


/**Transfer energy (Nozaki-Tanford, 1971)**/
const std::map<char, double> AA_TRANSFER_ENERGY = {
        { 'A', 0.5 },
        { 'R', 0 },
        { 'N', 0 },
        { 'D', 0 },
        { 'C', 0 },
        { 'Q', 0 },
        { 'E', 0 },
        { 'G', 0 },
        { 'H', 0.5 },
        { 'I', 1.8 },
        { 'L', 1.8 },
        { 'K', 0 },
        { 'M', 1.3 },
        { 'F', 2.5 },
        { 'P', 0 },
        { 'S', 0 },
        { 'T', 0.4 },
        { 'W', 3.4 },
        { 'Y', 2.3 },
        { 'V', 1.5 }
};


/** Hydrophilicity value (Hopp-Woods, 1981) **/
const std::map<char, double> AA_HYDROPHILICITY = {
        { 'A', -0.5 },
        { 'R', 3 },
        { 'N', 0.2 },
        { 'D', 3 },
        { 'C', -1 },
        { 'Q', 0.2 },
        { 'E', 3 },
        { 'G', 0 },
        { 'H', -0.5 },
        { 'I', -1.8 },
        { 'L', -1.8 },
        { 'K', 3 },
        { 'M', -1.3 },
        { 'F', -2.5 },
        { 'P', 0 },
        { 'S', 0.3 },
        { 'T', -0.4 },
        { 'W', -3.4 },
        { 'Y', -2.3 },
        { 'V', -1.5 }
};

/**Packing density (Tsai et al 1999)**/
const std::map<char, double> AA_PACKING_DENSITY = {
        { 'A', 89.3 },
        { 'R', 190.3 },
        { 'N', 122.4 },
        { 'D', 114.4 },
        { 'C', 102.5 },
        { 'Q', 146.9 },
        { 'E', 138.8 },
        { 'G', 63.8 },
        { 'H', 157.5 },
        { 'I', 163 },
        { 'L', 163.1 },
        { 'K', 165.1 },
        { 'M', 165.8 },
        { 'F', 190.8 },
        { 'P', 121.6 },
        { 'S', 94.2 },
        { 'T', 119.6 },
        { 'W', 226.4 },
        { 'Y', 194.6 },
        { 'V', 138.2 }
};

/** Nature of the accessible and buried surfaces (Chothia, 1976) **/
const std::map<char, double> AA_SURFACE_ACCESSIBILITY = {
        { 'A', 115 },
        { 'R', 225 },
        { 'N', 160 },
        { 'D', 150 },
        { 'C', 135 },
        { 'Q', 180 },
        { 'E', 190 },
        { 'G', 75 },
        { 'H', 195 },
        { 'I', 175 },
        { 'L', 170 },
        { 'K', 200 },
        { 'M', 185 },
        { 'F', 210 },
        { 'P', 145 },
        { 'S', 115 },
        { 'T', 140 },
        { 'W', 255 },
        { 'Y', 230 },
        { 'V', 155 }
};


/** Shape and surface features (Prabhakaran-Ponnuswamy, 1982) **/
const std::map<char, double> AA_SURFACE = {
        { 'A', 0.305 },
        { 'R', 0.227 },
        { 'N', 0.322 },
        { 'D', 0.335 },
        { 'C', 0.339 },
        { 'Q', 0.306 },
        { 'E', 0.282 },
        { 'G', 0.352 },
        { 'H', 0.215 },
        { 'I', 0.278 },
        { 'L', 0.262 },
        { 'K', 0.391 },
        { 'M', 0.28 },
        { 'F', 0.195 },
        { 'P', 0.346 },
        { 'S', 0.326 },
        { 'T', 0.251 },
        { 'W', 0.291 },
        { 'Y', 0.293 },
        { 'V', 0.291 }
};


/** Alpha-helix indices (Geisow-Roberts, 1980) **/
const std::map<char, double> AA_ALPHA_HELIX = {
        { 'A', 1.29 },
        { 'R', 1 },
        { 'N', 0.81 },
        { 'D', 1.1 },
        { 'C', 0.79 },
        { 'Q', 1.07 },
        { 'E', 1.49 },
        { 'G', 0.63 },
        { 'H', 1.33 },
        { 'I', 1.05 },
        { 'L', 1.31 },
        { 'K', 1.33 },
        { 'M', 1.54 },
        { 'F', 1.13 },
        { 'P', 0.63 },
        { 'S', 0.78 },
        { 'T', 0.77 },
        { 'W', 1.18 },
        { 'Y', 0.71 },
        { 'V', 0.81 }
};

/** Helix-coil equilibrium constant (Finkelstein-Ptitsyn, 1977) **/
const std::map<char, double> AA_HELIX_COIL = {
        { 'A', 1.08 },
        { 'R', 1.05 },
        { 'N', 0.85 },
        { 'D', 0.85 },
        { 'C', 0.95 },
        { 'Q', 0.95 },
        { 'E', 1.15 },
        { 'G', 0.55 },
        { 'H', 1 },
        { 'I', 1.05 },
        { 'L', 1.25 },
        { 'K', 1.15 },
        { 'M', 1.15 },
        { 'F', 1.1 },
        { 'P', 0.71 },
        { 'S', 0.75 },
        { 'T', 0.75 },
        { 'W', 1.1 },
        { 'Y', 1.1 },
        { 'V', 0.95 }
};


#endif //MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

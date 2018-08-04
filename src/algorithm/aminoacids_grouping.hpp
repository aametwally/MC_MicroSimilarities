//
// Created by asem on 04/08/18.
//

#ifndef MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP
#define MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

#include "common.hpp"

enum class AminoAcidGroupingEnum {
    OLFER15 ,
    OLFER8 ,
    DIAMOND11
};

const std::map< std::string , AminoAcidGroupingEnum > GroupingLabels {
{"olfer15",AminoAcidGroupingEnum ::OLFER15 } ,
{"olfer8" , AminoAcidGroupingEnum ::OLFER8 } ,
{"diamond11" , AminoAcidGroupingEnum ::DIAMOND11 }
};

template<size_t N>
constexpr std::array<char, 256> alphabetIds(const std::array<char, N> &alphabet)
{
    std::array<char, 256> ids{};
    int i = 0;
    for (char aa : alphabet)
        ids.at(size_t(aa)) = char(i++);
    return ids;
};

template<size_t N>
constexpr std::array<char, N>
reducedAlphabet()
{
    std::array<char, N> newAlphabet{};
    for (auto i = 0; i < newAlphabet.size(); ++i)
        newAlphabet.at(i) = 'A' + i;
    return newAlphabet;
};

template<size_t N>
constexpr std::array<char, 256>
reducedAlphabetIds(const std::array<const char *, N> &alphabetGrouping)
{
    std::array<char, 256> ids{};
    int i = 0;
    for (const auto &aaGroup : alphabetGrouping) {
        size_t j = 0;
        while (aaGroup[j] != '\0')
            ids.at(size_t(aaGroup[j++])) = char(i);
        ++i;
    }
    return ids;
};

constexpr std::array<char, 20> aaSet = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
                                        'T', 'V', 'W', 'Y'};

constexpr std::array<const char *, 15> AAGrouping_OLFER15 = {"KR", "E", "D", "Q", "N", "C", "G", "H", "ILVM", "F", "Y",
                                                             "W", "P", "ST", "A"};
constexpr std::array<const char *, 8> AAGrouping_OLFER8 = {"KRH", "ED", "C", "G", "AILVM", "FYW", "P", "NQST"};

constexpr std::array<const char *, 11> AAGrouping_DIAMOND11 = {"KREDQN", "C", "G", "H", "ILV", "M", "F", "Y", "W", "P",
                                                               "STA"};

#endif //MARKOVIAN_FEATURES_AMINOACIDS_GROUPING_HPP

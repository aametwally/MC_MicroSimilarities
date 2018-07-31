//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_FASTAENTRY_HPP
#define MARKOVIAN_FEATURES_FASTAENTRY_HPP

#include "SequenceEntry.hpp"
#include "common.hpp"


class FastaEntry : SequenceEntry<FastaEntry>
{
public:
    FastaEntry(const std::string &id, const std::string &seq)
            : _id(id), _sequence(seq)
    {}

    size_t sequenceLength() const override
    {
        return _sequence.length();
    }

    std::string clusterNameUniRef() const override
    {
        using io::split;
        using io::join;
        auto tokens = split(getId());
        return join(tokens.cbegin() + 1,
                    std::find_if(tokens.cbegin() + 1, tokens.cend(),
                                 [](const std::string &s) {
                                     return s.find('=') != std::string::npos;
                                 }), " ");

    }

    const std::string &getId() const
    {
        return _id;
    }

    void setId(const std::string &id)
    {
        _id = id;
    }

    const std::string &getSequence() const override
    {
        return _sequence;
    }

    void setSequence(const std::string &sequence)
    {
        _sequence = sequence;
    }

    static std::vector<FastaEntry>
    readFastaFile(const std::string &path,
                  int limit = -1)
    {
        std::ifstream input(path);
        if (!input.good()) {
            std::cerr << "Error opening: " << path << " . You have failed." << std::endl;
            return {};
        }
        std::string line, id, DNA_sequence;
        std::vector<FastaEntry> fasta;

        while (std::getline(input, line) && (limit == -1 || fasta.size() < limit)) {

            // line may be empty so you *must* ignore blank lines
            // or you have a crash waiting to happen with line[0]
            if (line.empty())
                continue;

            if (line.front() == '>') {
                // output previous line before overwriting id
                // but ONLY if id actually contains something
                if (!id.empty())
                    fasta.emplace_back(id, DNA_sequence);

                id = line.substr(1);
                DNA_sequence.clear();
            } else {//  if (line[0] != '>'){ // not needed because implicit
                DNA_sequence += line;
            }
        }

        // output final entry
        // but ONLY if id actually contains something
        if (!id.empty())
            fasta.emplace_back(id, DNA_sequence);
        return fasta;
    }

    static void writeFastaFile(const std::vector<FastaEntry> &fItems,
                               const std::string &filePath)
    {
        std::ofstream ostream(filePath);
        for (auto &item : fItems)
            ostream << '>' << item.getId() << '\n'
                    << item.getSequence() << '\n';
    }

private:
    std::string _id;
    std::string _sequence;

};


#endif //MARKOVIAN_FEATURES_FASTAENTRY_HPP

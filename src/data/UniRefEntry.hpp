//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_UNIREFENTRY_HPP
#define MARKOVIAN_FEATURES_UNIREFENTRY_HPP

#include "SequenceEntry.hpp"
#include "FastaEntry.hpp"
#include "common.hpp"

class UniRefEntry : public SequenceEntry<UniRefEntry>
{
public:
    size_t sequenceLength() const override
    {
        return _sequence.length();
    }

    std::string clusterNameUniRef() const override
    {
        return _clusterName;
    }

    static UniRefEntry from_fasta( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        UniRefEntry uniref;
        auto tokens = split( fEntry.getId());
        uniref._memberId = split( split( tokens.front(), "_" ).front(), "|" ).front();
        uniref._clusterName = join( tokens.cbegin() + 1,
                                    std::find_if( tokens.cbegin() + 1, tokens.cend(),
                                                  []( const std::string &s ) {
                                                      return s.find( '=' ) != std::string::npos;
                                                  } ), " " );
        uniref._sequence = fEntry.getSequence();
        return uniref;
    }

    static UniRefEntry from_fasta_PSORTDB( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        UniRefEntry uniref;
        auto tokens = split( fEntry.getId());
        uniref._memberId = split( tokens.front(), "|" ).front();
        uniref._clusterName =  std::string( tokens.at(1).cbegin() + 1, tokens.at(1).cend() - 1 );
        uniref._sequence = fEntry.getSequence();
        return uniref;
    }

    const std::string &getMemberId() const
    {
        return _memberId;
    }

    void setMemberId( const std::string &memberId )
    {
        _memberId = memberId;
    }

    const std::string &getClusterName() const
    {
        return _clusterName;
    }

    void setClusterName( const std::string &clusterName )
    {
        _clusterName = clusterName;
    }

    const std::string &getSequence() const override
    {
        return _sequence;
    }

    void setSequence( const std::string &sequence )
    {
        _sequence = sequence;
    }

    static std::vector<UniRefEntry>
    fasta2UnirefEntries( std::vector<FastaEntry> fastaEntries )
    {
        std::vector<UniRefEntry> unirefEntries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( unirefEntries ), from_fasta );
        return unirefEntries;
    }

    static std::vector<UniRefEntry>
    fasta2UnirefEntries_PSORTDB( const std::vector<FastaEntry> &fastaEntries )
    {
        std::vector<UniRefEntry> unirefEntries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( unirefEntries ), from_fasta_PSORTDB );
        return unirefEntries;
    }

    static std::vector<UniRefEntry>
    loadEntries( const std::string &input, const std::string &format )
    {
        if( format == "uniref")
            return fasta2UnirefEntries( FastaEntry::readFasta( input ));
        else if( format == "psort" )
            return fasta2UnirefEntries_PSORTDB( FastaEntry::readFasta( input ));
        else throw std::runtime_error(fmt::format("Unexpected fasta file format:{}",format));
    }


protected:
    std::string _memberId;
    std::string _clusterName;
    std::string _sequence;
};

#endif //MARKOVIAN_FEATURES_UNIREFENTRY_HPP

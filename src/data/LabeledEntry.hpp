//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_UNIREFENTRY_HPP
#define MARKOVIAN_FEATURES_UNIREFENTRY_HPP

#include "SequenceEntry.hpp"
#include "FastaEntry.hpp"
#include "common.hpp"


enum class DBFormatProcessor
{
    UniRef,
    TargetP,
    PSORT,
    DeepLoc_Location,
    DeepLoc_Solubility
};
const std::map<std::string, DBFormatProcessor> FormatLabels = {
        {"uniref",      DBFormatProcessor::UniRef},
        {"psort",       DBFormatProcessor::PSORT},
        {"targetp",     DBFormatProcessor::TargetP},
        {"deeploc_loc", DBFormatProcessor::DeepLoc_Location},
        {"deeploc_sol", DBFormatProcessor::DeepLoc_Solubility}
};

class LabeledEntry : public SequenceEntry<LabeledEntry>
{
public:
    size_t sequenceLength() const override
    {
        return _sequence.length();
    }

    static LabeledEntry from_fasta_UNIREF( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        LabeledEntry uniref;
        auto tokens = split( fEntry.getId());
        uniref._memberId = split( split( tokens.front(), "_" ).front(), "|" ).front();
        uniref._label = join( tokens.cbegin() + 1,
                              std::find_if( tokens.cbegin() + 1, tokens.cend(),
                                            []( const std::string &s ) {
                                                return s.find( '=' ) != std::string::npos;
                                            } ), " " );
        uniref._sequence = fEntry.getSequence();
        return uniref;
    }

    static LabeledEntry from_fasta_PSORTDB( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        LabeledEntry entry;
        auto tokens = split( fEntry.getId());
        entry._memberId = split( tokens.front(), "|" ).front();
        entry._label = std::string( tokens.at( 1 ).cbegin() + 1, tokens.at( 1 ).cend() - 1 );
        entry._sequence = fEntry.getSequence();
        return entry;
    }

    static LabeledEntry from_fasta_TARGETP( const FastaEntry &fEntry, const std::string &clusterName )
    {
        using io::split;
        using io::join;

        LabeledEntry entry;
        auto tokens = split( fEntry.getId());
        entry._memberId = split( tokens.front(), ";" ).front();
        entry._label = clusterName;
        entry._sequence = fEntry.getSequence();
        return entry;
    }

    static LabeledEntry from_fasta_DEEPLOC_LOCATION( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        LabeledEntry entry;
        auto tokens = split( fEntry.getId());
        entry._memberId = tokens.front();
        entry._label = split( tokens.at( 1 ), "-" ).front();
        entry._sequence = fEntry.getSequence();
        return entry;
    }

    static LabeledEntry from_fasta_DEEPLOC_SOLUBILITY( const FastaEntry &fEntry )
    {
        using io::split;
        using io::join;

        LabeledEntry entry;
        auto tokens = split( fEntry.getId());
        entry._memberId = tokens.front();
        entry._label = split( tokens.at( 1 ), "-" ).back();
        entry._sequence = fEntry.getSequence();
        return entry;
    }

    const std::string &getMemberId() const
    {
        return _memberId;
    }

    void setMemberId( const std::string &memberId )
    {
        _memberId = memberId;
    }

    const std::string &getLabel() const
    {
        return _label;
    }

    void setLabel( const std::string &clusterName )
    {
        _label = clusterName;
    }

    const std::string &getSequence() const override
    {
        return _sequence;
    }

    void setSequence( const std::string &sequence )
    {
        _sequence = sequence;
    }

    static std::vector<LabeledEntry>
    fasta2LabeledEntries_UNIREF( std::vector<FastaEntry> fastaEntries )
    {
        std::vector<LabeledEntry> entries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( entries ), from_fasta_UNIREF );
        return entries;
    }

    static std::vector<LabeledEntry>
    fasta2LabeledEntries_PSORTDB( const std::vector<FastaEntry> &fastaEntries )
    {
        std::vector<LabeledEntry> entries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( entries ), from_fasta_PSORTDB );
        return entries;
    }

    static std::vector<LabeledEntry>
    fasta2LabeledEntries_TARGETP( const std::map<std::string, std::vector<FastaEntry >> &fastaEntries )
    {
        std::vector<LabeledEntry> entries;
        for (const auto &[clusterName, fEntries] : fastaEntries)
            std::transform( fEntries.cbegin(), fEntries.cend(),
                            std::back_inserter( entries ), [&]( const auto &fi ) {
                        return from_fasta_TARGETP( fi, clusterName );
                    } );
        return entries;
    }

    static std::vector<LabeledEntry>
    fasta2LabeledEntries_DEEPLOC_LOCATION( const std::vector<FastaEntry> &fastaEntries )
    {
        std::vector<LabeledEntry> entries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( entries ), from_fasta_DEEPLOC_LOCATION );
        return entries;
    }

    static std::vector<LabeledEntry>
    fasta2LabeledEntries_DEEPLOC_SOLUBILITY( const std::vector<FastaEntry> &fastaEntries )
    {
        std::vector<LabeledEntry> entries;
        std::transform( fastaEntries.cbegin(), fastaEntries.cend(),
                        std::back_inserter( entries ), from_fasta_DEEPLOC_SOLUBILITY );
        return entries;
    }

    static std::vector<LabeledEntry>
    loadEntries( const std::string &input, const std::string &format )
    {
        switch (DBFormatProcessor formatLabel = FormatLabels.at( format ); formatLabel)
        {
            case DBFormatProcessor::UniRef:
                return fasta2LabeledEntries_UNIREF( FastaEntry::readFasta( input ));
            case DBFormatProcessor::PSORT:
                return fasta2LabeledEntries_PSORTDB( FastaEntry::readFasta( input ));
            case DBFormatProcessor::TargetP:
                return fasta2LabeledEntries_TARGETP( FastaEntry::readFastaGroupByFilename( input ));
            case DBFormatProcessor::DeepLoc_Location:
                return fasta2LabeledEntries_DEEPLOC_LOCATION( FastaEntry::readFasta( input ));
            case DBFormatProcessor::DeepLoc_Solubility:
                return fasta2LabeledEntries_DEEPLOC_SOLUBILITY( FastaEntry::readFasta( input ));
        }
        throw std::runtime_error( fmt::format( "Unexpected fasta file format:{}", format ));
    }


protected:
    std::string _memberId;
    std::string _label;
    std::string _sequence;
};

#endif //MARKOVIAN_FEATURES_UNIREFENTRY_HPP

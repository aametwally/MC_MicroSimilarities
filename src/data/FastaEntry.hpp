//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_FASTAENTRY_HPP
#define MARKOVIAN_FEATURES_FASTAENTRY_HPP

#include "SequenceEntry.hpp"
#include "common.hpp"


class FastaEntry : public SequenceEntry<FastaEntry>
{
public:
    FastaEntry(
            std::string id,
            std::string seq
    )
            : _id( std::move( id )), _sequence( std::move( seq ))
    {}

    size_t length() const override
    {
        return _sequence.length();
    }

    const std::string &getId() const
    {
        return _id;
    }

    void setId( const std::string &id )
    {
        _id = id;
    }

    std::string_view sequence() const override
    {
        return _sequence;
    }

    template<typename Sequence>
    void setSequence( Sequence &&sequence )
    {
        _sequence = std::forward<Sequence>( sequence );
    }

    static std::vector<FastaEntry>
    readFasta( const std::string &pathString )
    {
        namespace fs = std::experimental::filesystem;
        fs::path p( pathString );
        if ( fs::is_directory( p ))
        {
            std::vector<FastaEntry> entries;
            for (const auto &p : fs::directory_iterator( p ))
            {
                auto newEntries = readFastaFile( p.path());
                entries.insert( entries.cend(), std::begin( newEntries ), std::end( newEntries ));
            }

            return entries;
        } else return readFastaFile( pathString );
    }

    static std::map<std::string, std::vector<FastaEntry>>
    readFastaGroupByFilename( const std::string &pathString )
    {
        namespace fs = std::experimental::filesystem;
        fs::path p( pathString );
        if ( fs::is_directory( p ))
        {
            std::map<std::string, std::vector<FastaEntry >> entries;
            for (const auto &p : fs::directory_iterator( p ))
            {
                auto newEntries = readFastaFile( p.path());
                entries.emplace( p.path().filename().string(), std::move( newEntries ));
            }
            return entries;
        } else
        {
            std::map<std::string, std::vector<FastaEntry >> entries =
                    {{p.filename().string(), readFastaFile( pathString )}};
            return entries;
        }
    }

    static std::vector<FastaEntry>
    readFastaFile( const std::string &path )
    {
        std::ifstream input( path );
        if ( !input.good())
        {
            std::cerr << "Error opening: " << path << " . You have failed." << std::endl;
            return {};
        }
        std::string line, id, DNA_sequence;
        std::vector<FastaEntry> fasta;

        while (std::getline( input, line ))
        {

            // line may be empty so you *must* ignore blank lines
            // or you have a crash waiting to happen with line[0]
            if ( line.empty())
                continue;

            if ( line.front() == '>' )
            {
                // output previous line before overwriting id
                // but ONLY if id actually contains something
                if ( !id.empty())
                    fasta.emplace_back( id, DNA_sequence );

                id = line.substr( 1 );
                DNA_sequence.clear();
            } else
            {//  if (line[0] != '>'){ // not needed because implicit
                DNA_sequence += line;
            }
        }

        // output final entry
        // but ONLY if id actually contains something
        if ( !id.empty())
            fasta.emplace_back( id, DNA_sequence );
        return fasta;
    }

    static void writeFastaFile(
            const std::vector<FastaEntry> &fItems,
            const std::string &filePath
    )
    {
        std::ofstream ostream( filePath );
        for (auto &item : fItems)
            ostream << '>' << item.getId() << '\n'
                    << item.sequence() << '\n';
    }

private:
    std::string _id;
    std::string _sequence;

};


#endif //MARKOVIAN_FEATURES_FASTAENTRY_HPP

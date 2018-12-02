//
// Created by asem on 01/12/18.
//

//
// Created by asem on 08/11/18.
//

#include "common.hpp"
#include "clara.hpp"

#include "ProteinTimeSeries.hpp"

int main( int argc , char *argv[] )
{
    using io::join;
    std::string input , output;
    std::string fastaFormat = keys( FormatLabels ).front();

    bool showHelp = false;

    auto cli
            = clara::Arg( input , "input" )
                      ( "UniRef input file" )
              | clara::Opt( output , "output file" )
              ["-o"]["--output-file"]
                      ( "output file" )
              | clara::Opt( fastaFormat , join( keys( FormatLabels ) , "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}" , fastaFormat ))
              | clara::Help( showHelp );


    auto result = cli.parse( clara::Args( argc , argv ));
    if ( !result )
    {
        fmt::print( "Error in command line:{}\n" , result.errorMessage());
        exit( 1 );
    } else if ( showHelp )
    {
        cli.writeToStream( std::cout );
    } else
    {

        fmt::print( "[Args][input:{}]"
                    "[fformat:{}][output:{}]\n" ,
                    input , fastaFormat , output );

        auto proteins = ProteinTimeSeries::createProteinsTimserSeries(
                LabeledEntry::loadEntries( input , fastaFormat ));

        ProteinTimeSeries::print( proteins , output );
    }
    return 0;
}



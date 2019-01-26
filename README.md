## Installation on Linux

#### Prerequisites

- Install [Python-64bits](https://www.python.org/downloads) or run `apt-get install python3 python3-pip`, 32-bits version won't work.

Python is used for Conan package manager installation.

#### Build from source code

- Install conan, c++ package manager, preferably running `pip install conan`. For more information and alternative installation options, please refer to [conan manual page](http://docs.conan.io/en/latest/installation.html).
- Run `conan remote add a-alaa https://api.bintray.com/conan/a-alaa/public-conan`.
- Create `build` folder and, after moving into the new folder, run `conan install .. --build missing`.
- Run `cmake ..`.
- Run `make -j8`.
#!/bin/sh

PROJECT="$1"

cd /tmp
git clone https://github.com/paulhilbert/${PROJECT}.git
mkdir -p ${PROJECT}/build
cd ${PROJECT}/build
git checkout explicit_break
cmake -D CMAKE_INSTALL_PREFIX="/usr" -D CMAKE_BUILD_TYPE="Release" ..
make
make install
cd /tmp
rm -rf ${PROJECT}

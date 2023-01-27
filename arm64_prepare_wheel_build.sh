#!/bin/bash
mkdir -p ~/arm-target/bin
mkdir -p ~/arm-target/brew-cache
export PATH="$HOME/arm-target/bin:$PATH"

cd ~/arm-target
mkdir arm-homebrew && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C arm-homebrew
ln -s ~/arm-target/arm-homebrew/bin/brew ~/arm-target/bin/arm-brew

export HOMEBREW_CACHE=~/arm-target/brew-cache
export HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1
arm-brew fetch --deps --bottle-tag=arm64_big_sur llvm libomp jemalloc xsimd |\
  grep -E "(Downloaded to:|Already downloaded:)" |\
  grep -v pkg-config |\
  awk '{ print $3 }' |\
  xargs -n 1 arm-brew install --ignore-dependencies --force-bottle

# Install host version of pkg-config so we can call it in the build system
arm-brew install pkg-config
ln -s ~/arm-target/arm-homebrew/bin/pkg-config ~/arm-target/bin/arm-pkg-config

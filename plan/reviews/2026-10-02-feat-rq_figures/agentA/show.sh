# usage: show.sh <commit> <paths...>
export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
c=$1; shift; git show --format= -U4 $c -- "$@"

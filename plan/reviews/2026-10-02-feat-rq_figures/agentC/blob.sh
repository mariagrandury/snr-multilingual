#!/bin/bash
# usage: blob.sh <repo-relative path>  -> prints path of a readable copy of the HEAD blob (LFS resolved from local object store)
R=/Users/mariagrandury/Projects/epfl/snr-multilingual
p="$1"; t=$(git -C $R show d309cb72:"$p" | head -c 200)
if [[ "$t" == version\ https://git-lfs* ]]; then
  oid=$(echo "$t" | sed -n 's/^oid sha256://p'); f=$(git -C $R rev-parse --git-common-dir)/lfs/objects/${oid:0:2}/${oid:2:2}/$oid
  [[ "$f" != /* ]] && f=$R/$f
  echo $f
else
  o=/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/agentC/blobs/$(echo "$p" | tr '/' '_'); git -C $R show d309cb72:"$p" > $o; echo $o
fi

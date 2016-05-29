#!/usr/bin/env bash
cd /home/chenl/drive/flickr/flickr_40k
for name in portrait*
do
    newname="n$name"
    echo Old name: $name, new name: $newname
    mv "$name" "$newname"
done

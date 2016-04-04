for name in vggpool3*
do
    newname="$(echo "$name" | cut -c10-)"
    echo Old name: $name, new name: $newname
    mv "$name" "$newname"
done

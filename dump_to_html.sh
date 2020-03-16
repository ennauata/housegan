#!/bin/bash
###########################################################
# Usage: imgDetails.sh
#
# Script will save to /private/tmp/imgDetails_(PID).html 
# with the details of images it came across in the current 
# directory which the script was run. When complete, 
# the script will open the tmp file in textEdit.
############################################################
PREFIX=$1

declare -r OUT=./all.html
declare -r CMD="sips -g pixelWidth -g pixelHeight"
declare -a PROPS=()
declare -ar ALLOWED=(
$PREFIX/*.jpg *.JPG $PREFIX/*.svg
*.GIF *.gif
*.png *.PNG
)

let COUNT=0

for ITEM in ${ALLOWED[@]}; do
  if [ -f $ITEM ]; then
    pos=0
    for PROP in $($CMD "$ITEM"|tail -2|sed 's/ //g'|awk -F':' '{print $2}')
    do
      echo $PROP | egrep '[0-9]+'>/dev/null 2>&1
      if [ $? == 0 ]; then
        PROPS[$pos]=$PROP
        pos=$((pos+1))
      fi
    done
    if [ -n ${PROPS[0]} -a -n ${PROPS[1]} ]; then
      echo "<object data=\"${ITEM}\" type=\"image/svg+xml\"></object>" | tee -a $OUT
      COUNT=$((COUNT+1))
    fi      
    if [ $COUNT -ge 900 ]; then
      break
    fi
  fi
done

echo -e "\nAttempted to process (${COUNT}) files."

[ -f $OUT ] && open -e $OUT

exit 0

#
PYTHON="/Users/xyli1905/anaconda3/envs/deepinfo/bin/python"
# PYTHON=echo | which python

# ------------------------- results for threshold ------------------------ #
range_slope="0.5 1.0 2.0"

for slope in $range_slope
do

if [ "$slope" = "0.5" ]
then
    range_dispX="-2.5 -1.5 -1.0 -0.5 0.0 0.5 1.5"
elif [ "$slope" = "1.0" ]
then
    range_dispX="-1.5 -0.5 0.0 0.5 1.5"
elif [ "$slope" = "2.0" ]
then
    range_dispX="-1.0 -0.25 0.0 0.25 0.5 0.75 1.5"
fi

for dispX in $range_dispX
do

echo "run for: slope = $slope, dispX = $dispX"

$PYTHON IBnet.py \
--experiment_name test-tanhx-slope1_0-dispX0_0 \
--max_epoch 8000 \
--activation tanhx \
--Vmax 1.0 \
--Vmin 0.0 \
--slope $slope \
--dispX $dispX \

done

done

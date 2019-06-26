#
PYTHON="/Users/xyli1905/anaconda3/envs/deepinfo/bin/python"

# ------------------------- results for threshold ------------------------ #
range_slope="1.0"
range_dispX="0.0"

for slope in $range_slope
do

for dispX in $range_dispX
do

echo "run for: slope = $slope, dispX = $dispX"
$PYTHON IBnet.py \
--experiment_name test-tanhx-slope1_0-dispX0_0 \
--max_epoch 100 \
--activation tanhx \
--Vmax 1.0 \
--Vmin 0.0 \
--slope $slope \
--dispX $dispX \

done

done
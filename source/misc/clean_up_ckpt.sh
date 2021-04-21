#!/bin/bash
outputdir=/shared/lyuqing/probing_for_event/output_model_dir
cd $outputdir
for model_name in *; do
    cd $outputdir/$model_name
    echo $model_name
    rm -rf checkpoint*
done
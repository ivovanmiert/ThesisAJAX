#!/bin/bash

Array of hyperparameters to choose from
# learning_rates=(0.0001 0.001 0.01 0.1)
# batch_sizes=(32 64)
# dropout_rates=(0.5 0.3)
# layers=(2 3 4)
# hidden_units=(64 128 256)
#hidden: 64, layers: 4, dropout: 0.3, batch_size: 64, learning rate: 0.001
# learning_rates=(0.0001)
# batch_sizes=(64)
# dropout_rates=(0.5)
# layers=(2)
# hidden_units=(64)

#Top down
# learning_rates=(0.001)
# batch_sizes=(128)
# dropout_rates=(0.3)
# layers=(2)
# hidden_units=(32)

#FINAL:
learning_rates=(0.001)
batch_sizes=(64)
hidden_units=(64)
layers=(3)
dropout=(0.5)

#Bottom up
# learning_rates=(0.001)
# batch_sizes=(64)
# dropout_rates=(0.5)
# layers=(2)
# hidden_units=(256)

#classification_sort=('shot_body_part' 'duel')
classification_sort=('primary')
#classification_sort=('shot_body_part' 'duel' 'normal_pass_or_cross' 'accuracy_cross' 'cross_direction' 'cross_flank' 'pass_accurate' 'pass_direction' 'pass_distance' 'pass_progressive' 'pass_through' 'shot_on_target')
hpe_sort=('basic' 'top_down' 'bottom_up')
#hpe_sort=('basic')
#hpe_sort=('basic' 'top_down')
#classification_sort=('normal_pass_or_cross' 'pass_accurate' 'pass_distance')

#Basic
learning_rates=(0.001)
batch_sizes=(64)
dropout_rates=(0.3)
layers=(4)
hidden_units=(64)


# Specify the number of random combinations to test
n=1 # Change this to however many combinations you want

for hpe_sort_value in "${hpe_sort[@]}"; do
  for classification_sort_value in "${classification_sort[@]}"; do
    lr=${learning_rates[RANDOM % ${#learning_rates[@]}]}
    batch=${batch_sizes[RANDOM % ${#batch_sizes[@]}]}
    dropout=${dropout_rates[RANDOM % ${#dropout_rates[@]}]}
    layer=${layers[RANDOM % ${#layers[@]}]}
    hidden=${hidden_units[RANDOM % ${#hidden_units[@]}]}

    # Submit the job with random parameters and the current classification_sort
    sbatch job_run_classification_parameters.sh $hidden $layer $dropout $batch $lr $classification_sort_value $hpe_sort_value
  done
done

# # Loop to generate random combinations and submit jobs
# for ((i = 0; i < n; i++)); do
#   lr=${learning_rates[RANDOM % ${#learning_rates[@]}]}
#   batch=${batch_sizes[RANDOM % ${#batch_sizes[@]}]}
#   dropout=${dropout_rates[RANDOM % ${#dropout_rates[@]}]}
#   layer=${layers[RANDOM % ${#layers[@]}]}
#   hidden=${hidden_units[RANDOM % ${#hidden_units[@]}]}
#   classification_sort=${classification_sort[RANDOM % ${#classification_sort[@]}]}

#   sbatch job_run_classification_parameters.sh $hidden $layer $dropout $batch $lr $classification_sort
# done
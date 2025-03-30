#!/usr/bin/env sh

seq 0 12 | parallel -j 3 python3 main_three_layers_delays_dist_sense_actuate_lili.py delays_stats_subang_v9_lili_sense_actuate_gaussian gaussian {} &
seq 0 12 | parallel -j 3 python3 main_three_layers_delays_dist_sense_actuate_lili.py delays_stats_subang_v9_lili_sense_actuate_poisson poisson {} &

wait

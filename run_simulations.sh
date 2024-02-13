#!/usr/bin/env sh

time seq 0 60 | parallel -j 20 python main_three_layers_nodist_parallel.py stats_subang_v9_nodist {}

time seq 0 67 |  parallel -j 4 python main_three_layers_delays_dist_actuate_parallel.py delays_stats_subang_v9_actuate_poisson poisson {} &
time seq 0 67 |  parallel -j 4 python main_three_layers_delays_dist_actuate_parallel.py delays_stats_subang_v9_actuate_gaussian gaussian {} &
time seq 0 120 | parallel -j 7 python main_three_layers_delays_dist_sense_parallel.py delays_stats_subang_v9_sense_poisson poisson {} &
time seq 0 120 | parallel -j 7 python main_three_layers_delays_dist_sense_parallel.py delays_stats_subang_v9_sense_gaussian gaussian {} &

wait

time seq 0 500 | parallel -j 10 python main_three_layers_delays_dist_sense_actuate_parallel.py delays_stats_subang_v9_sense_actuation_poisson poisson {} &
time seq 0 500 | parallel -j 10 python main_three_layers_delays_dist_sense_actuate_parallel.py delays_stats_subang_v9_sense_actuation_gaussian gaussian {} &

wait

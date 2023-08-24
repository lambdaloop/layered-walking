#!/usr/bin/env bash

cd vids

## -- control forward walking
# python3 ../tile_videos_horizontal.py simulated_fly_8_0_0.mp4 simulated_fly_10_0_0.mp4 simulated_fly_8_10.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_12_0_0.mp4 simulated_fly_14_0_0.mp4 simulated_fly_12_14.mp4
# python3 ../tile_videos_vertical.py simulated_fly_8_10.mp4 simulated_fly_12_14.mp4 simulated_fly_control.mp4
# python3 ../add_text_video.py simulated_fly_control.mp4 simulated_fly_control_text.mp4 \
    # "8 mm/s forward" "10 mm/s forward" "12 mm/s forward" "14 mm/s forward" "Simulated forward walking"

## -- control rotation walking
# python3 ../tile_videos_horizontal.py simulated_fly_12_0_-4.mp4 simulated_fly_12_0_4.mp4 simulated_fly_side.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_12_-8_0.mp4 simulated_fly_12_8_0.mp4 simulated_fly_rotation.mp4
# python3 ../tile_videos_vertical.py simulated_fly_side.mp4 simulated_fly_rotation.mp4 simulated_fly_side_rotation.mp4
# python3 ../add_text_video.py simulated_fly_side_rotation.mp4 simulated_fly_side_rotation_text.mp4 \
    # "-4 mm/s side" "4 mm/s side" "-8 mm/s rotation" "8 mm/s rotation" "Simulated walking with nonzero side or rotation speed" "(each with 12 mm/s forward)"

## -- impulse perturbation, actuation delay
# python3 ../tile_videos_horizontal.py simulated_fly_actdelay_impulse_0ms.mp4 simulated_fly_actdelay_impulse_15ms.mp4 simulated_actdelay_impulse_0_15.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_actdelay_impulse_30ms.mp4 simulated_fly_actdelay_impulse_45ms.mp4 simulated_actdelay_impulse_30_45.mp4
# python3 ../tile_videos_vertical.py simulated_actdelay_impulse_0_15.mp4 simulated_actdelay_impulse_30_45.mp4 simulated_actdelay_impulse.mp4
# python3 ../add_text_video.py simulated_actdelay_impulse.mp4 simulated_actdelay_impulse_text.mp4 \
#     "0 ms actuation delay" "15 ms actuation delay" "30 ms actuation delay" "45 ms actuation delay" \
#     "Simulated walking with different actuation delays" "Impulse perturbations, walking 12 mm/s forward"
# python3 ../add_stim_indicator.py simulated_actdelay_impulse_text.mp4 simulated_actdelay_impulse_text_stim.mp4 300 330


## -- poisson perturbation, actuation delay
# python3 ../tile_videos_horizontal.py simulated_fly_actdelay_poisson_0ms.mp4 simulated_fly_actdelay_poisson_15ms.mp4 simulated_actdelay_poisson_0_15.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_actdelay_poisson_30ms.mp4 simulated_fly_actdelay_poisson_45ms.mp4 simulated_actdelay_poisson_30_45.mp4
# python3 ../tile_videos_vertical.py simulated_actdelay_poisson_0_15.mp4 simulated_actdelay_poisson_30_45.mp4 simulated_actdelay_poisson.mp4
# python3 ../add_text_video.py simulated_actdelay_poisson.mp4 simulated_actdelay_poisson_text.mp4 \
#     "0 ms actuation delay" "15 ms actuation delay" "30 ms actuation delay" "45 ms actuation delay" \
#     "Simulated walking with different actuation delays" "Persistent stochastic perturbations, walking 12 mm/s forward"
# python3 ../add_stim_indicator.py simulated_actdelay_poisson_text.mp4 simulated_actdelay_poisson_text_stim.mp4 300 600


## -- impulse perturbation, sensory delay
# python3 ../tile_videos_horizontal.py simulated_fly_sensedelay_impulse_0ms.mp4 simulated_fly_sensedelay_impulse_5ms.mp4 simulated_sensedelay_impulse_0_5.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_sensedelay_impulse_10ms.mp4 simulated_fly_sensedelay_impulse_15ms.mp4 simulated_sensedelay_impulse_10_15.mp4
# python3 ../tile_videos_vertical.py simulated_sensedelay_impulse_0_5.mp4 simulated_sensedelay_impulse_10_15.mp4 simulated_sensedelay_impulse.mp4
# python3 ../add_text_video.py simulated_sensedelay_impulse.mp4 simulated_sensedelay_impulse_text.mp4 \
#     "0 ms sensory delay" "5 ms sensory delay" "10 ms sensory delay" "15 ms sensory delay" \
#     "Simulated walking with different sensory delays" "Impulse perturbations, walking 12 mm/s forward" "30 ms actuation delay"
# python3 ../add_stim_indicator.py simulated_sensedelay_impulse_text.mp4 simulated_sensedelay_impulse_text_stim.mp4 300 330


## -- poisson perturbation, actuation delay
# python3 ../tile_videos_horizontal.py simulated_fly_sensedelay_poisson_0ms.mp4 simulated_fly_sensedelay_poisson_5ms.mp4 simulated_sensedelay_poisson_0_5.mp4
# python3 ../tile_videos_horizontal.py simulated_fly_sensedelay_poisson_10ms.mp4 simulated_fly_sensedelay_poisson_15ms.mp4 simulated_sensedelay_poisson_10_15.mp4
# python3 ../tile_videos_vertical.py simulated_sensedelay_poisson_0_5.mp4 simulated_sensedelay_poisson_10_15.mp4 simulated_sensedelay_poisson.mp4
# python3 ../add_text_video.py simulated_sensedelay_poisson.mp4 simulated_sensedelay_poisson_text.mp4 \
#     "0 ms sensory delay" "5 ms sensory delay" "10 ms sensory delay" "15 ms sensory delay" \
#     "Simulated walking with different sensory delays" "Persistent stochastic perturbations, walking 12 mm/s forward" "30 ms actuation delay"
# python3 ../add_stim_indicator.py simulated_sensedelay_poisson_text.mp4 simulated_sensedelay_poisson_text_stim.mp4 300 330

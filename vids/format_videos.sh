#!/usr/bin/env bash


## -- control forward walking
python3 tile_videos_horizontal.py simulated_fly_8_0_0.mp4 simulated_fly_10_0_0.mp4 simulated_fly_12_0_0.mp4 simulated_fly_14_0_0.mp4 simulated_fly_forward.mp4
python3 tile_videos_horizontal.py real_fly_8_0_0.mp4 real_fly_10_0_0.mp4 real_fly_12_0_0.mp4 real_fly_14_0_0.mp4 real_fly_forward.mp4

python3 trim_video.py real_fly_forward.mp4 real_fly_forward_trimmed.mp4 200 0 0 0
python3 tile_videos_vertical.py simulated_fly_forward.mp4 real_fly_forward_trimmed.mp4 fly_forward.mp4
python3 add_text_video.py fly_forward.mp4 fly_forward_text.mp4 \
    "Forward walking" \
    "Simulated 8 mm/s" "10 mm/s" "12 mm/s" "14 mm/s" \
    "Real 8 mm/s" "10 mm/s" "12 mm/s" "14 mm/s"


## -- control rotation walking

python3 tile_videos_horizontal.py simulated_fly_12_0_-4.mp4 simulated_fly_12_0_4.mp4  simulated_fly_12_-8_0.mp4 simulated_fly_12_8_0.mp4 simulated_fly_rotation.mp4
python3 tile_videos_horizontal.py real_fly_12_0_-4.mp4 real_fly_12_0_4.mp4  real_fly_12_-8_0.mp4 real_fly_12_8_0.mp4 real_fly_rotation.mp4
python3 trim_video.py real_fly_rotation.mp4 real_fly_rotation_trimmed.mp4 200 0 0 0
python3 tile_videos_vertical.py simulated_fly_rotation.mp4 real_fly_rotation_trimmed.mp4 fly_rotation.mp4

python3 add_text_video.py fly_rotation.mp4 fly_rotation_text.mp4 \
    "Walking with nonzero side or rotation speed|(each with 12 mm/s forward)" \
    "Simulated -4 mm/s side" "4 mm/s side" "-8 mm/s rotation" "8 mm/s rotation" \
    "Real -4 mm/s side" "4 mm/s side" "-8 mm/s rotation" "8 mm/s rotation"

## -- impulse perturbation, actuation delay
ffmpeg -y -v warning -stats -i simulated_fly_actdelay_impulse_0ms.mp4 -ss 5 -to 15 simulated_fly_actdelay_impulse_0ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_actdelay_impulse_15ms.mp4 -ss 5 -to 15 simulated_fly_actdelay_impulse_15ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_actdelay_impulse_30ms.mp4 -ss 5 -to 15 simulated_fly_actdelay_impulse_30ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_actdelay_impulse_45ms.mp4 -ss 5 -to 15 simulated_fly_actdelay_impulse_45ms_cropped.mp4

python3 tile_videos_horizontal.py simulated_fly_actdelay_impulse_0ms_cropped.mp4 simulated_fly_actdelay_impulse_15ms_cropped.mp4 \
    simulated_fly_actdelay_impulse_30ms_cropped.mp4 simulated_fly_actdelay_impulse_45ms_cropped.mp4 simulated_actdelay_impulse.mp4

python3 add_stim_indicator.py simulated_actdelay_impulse.mp4 simulated_actdelay_impulse_stim.mp4 150 180

python3 add_text_video.py simulated_actdelay_impulse_stim.mp4 simulated_actdelay_impulse_stim_text.mp4 \
    "Simulated walking with different actuation delays|Impulse perturbations, 12 mm/s forward, 10 ms sensory delay" \
    "0 ms actuation delay" "15 ms actuation delay" "30 ms actuation delay" "45 ms actuation delay" \
    "Kinematic Similarity at perturbation = -1.18" "-1.36" "-1.94" "-1.84"

ffmpeg -y -v warning -stats -i simulated_actdelay_impulse_stim.mp4 \
    -filter_complex "[0:v]trim=start=4.5:end=8,setpts=PTS*3[outv]" -map "[outv]" simulated_actdelay_impulse_stim_slow.mp4

python3 add_text_video.py simulated_actdelay_impulse_stim_slow.mp4 simulated_actdelay_impulse_stim_slow_text.mp4 \
    "Replay|Slowed down 30x" \
    "0 ms actuation delay" "15 ms actuation delay" "30 ms actuation delay" "45 ms actuation delay" \
    "Kinematic Similarity at perturbation = -1.18" "-1.36" "-1.94" "-1.84"

ffmpeg -y -v warning -stats \
    -i simulated_actdelay_impulse_stim_text.mp4 -i simulated_actdelay_impulse_stim_slow_text.mp4 \
    -filter_complex "[0:v:0][1:v:0]concat=n=2:v=1[outv]" -map "[outv]" simulated_actdelay_impulse_final.mp4

## -- poisson perturbation, actuation delay
python3 tile_videos_horizontal.py simulated_fly_actdelay_poisson_0ms.mp4 simulated_fly_actdelay_poisson_15ms.mp4 \
    simulated_fly_actdelay_poisson_30ms.mp4 simulated_fly_actdelay_poisson_45ms.mp4 simulated_actdelay_poisson.mp4

python3 add_text_video.py simulated_actdelay_poisson.mp4 simulated_actdelay_poisson_text.mp4 \
    "Simulated walking with different actuation delays|Persistent stochastic perturbations, 12 mm/s forward, 10 ms sensory delay" \
    "0 ms actuation delay" "15 ms actuation delay" "30 ms actuation delay" "45 ms actuation delay" \
    "Kinematic Similarity at perturbation = -1.12" "-1.23" "-1.51" "-1.95"

python3 add_stim_indicator.py simulated_actdelay_poisson_text.mp4 simulated_actdelay_poisson_text_stim.mp4 300 600

## -- impulse perturbation, sensory delay
ffmpeg -y -v warning -stats -i simulated_fly_sensedelay_impulse_0ms.mp4 -ss 5 -to 15 simulated_fly_sensedelay_impulse_0ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_sensedelay_impulse_5ms.mp4 -ss 5 -to 15 simulated_fly_sensedelay_impulse_5ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_sensedelay_impulse_10ms.mp4 -ss 5 -to 15 simulated_fly_sensedelay_impulse_10ms_cropped.mp4
ffmpeg -y -v warning -stats -i simulated_fly_sensedelay_impulse_15ms.mp4 -ss 5 -to 15 simulated_fly_sensedelay_impulse_15ms_cropped.mp4

python3 tile_videos_horizontal.py simulated_fly_sensedelay_impulse_0ms_cropped.mp4 simulated_fly_sensedelay_impulse_5ms_cropped.mp4 \
    simulated_fly_sensedelay_impulse_10ms_cropped.mp4 simulated_fly_sensedelay_impulse_15ms_cropped.mp4 simulated_sensedelay_impulse.mp4

python3 add_stim_indicator.py simulated_sensedelay_impulse.mp4 simulated_sensedelay_impulse_stim.mp4 150 180

python3 add_text_video.py simulated_sensedelay_impulse_stim.mp4 simulated_sensedelay_impulse_stim_text.mp4 \
    "Simulated walking with different sensory delays|Impulse perturbations, 12 mm/s forward, 30 ms actuation delay" \
    "0 ms sensory delay" "5 ms sensory delay" "10 ms sensory delay" "15 ms sensory delay" \
    "Kinematic Similarity at perturbation = -1.32" "-1.55" "-1.94" "-1.98"

ffmpeg -y -v warning -stats -i simulated_sensedelay_impulse_stim.mp4 \
    -filter_complex "[0:v]trim=start=4.5:end=8,setpts=PTS*3[outv]" -map "[outv]" simulated_sensedelay_impulse_stim_slow.mp4

python3 add_text_video.py simulated_sensedelay_impulse_stim_slow.mp4 simulated_sensedelay_impulse_stim_slow_text.mp4 \
    "Replay|Slowed down 30x" \
    "0 ms sensory delay" "5 ms sensory delay" "10 ms sensory delay" "15 ms sensory delay" \
    "Kinematic Similarity at perturbation = -1.32" "-1.55" "-1.94" "-1.98"

ffmpeg -y -v warning -stats \
    -i simulated_sensedelay_impulse_stim_text.mp4 -i simulated_sensedelay_impulse_stim_slow_text.mp4 \
    -filter_complex "[0:v:0][1:v:0]concat=n=2:v=1[outv]" -map "[outv]" simulated_sensedelay_impulse_final.mp4

## -- poisson perturbation, sensory delay
python3 tile_videos_horizontal.py simulated_fly_sensedelay_poisson_0ms.mp4 simulated_fly_sensedelay_poisson_5ms.mp4 \
    simulated_fly_sensedelay_poisson_10ms.mp4 simulated_fly_sensedelay_poisson_15ms.mp4 simulated_sensedelay_poisson.mp4

python3 add_text_video.py simulated_sensedelay_poisson.mp4 simulated_sensedelay_poisson_text.mp4 \
    "Simulated walking with different sensory delays|Persistent stochastic perturbations, 12 mm/s forward, 30 ms actuation delay" \
    "0 ms sensory delay" "5 ms sensory delay" "10 ms sensory delay" "15 ms sensory delay" \
    "Kinematic Similarity at perturbation = -1.35" "-1.74" "-1.51" "-2.02"

python3 add_stim_indicator.py simulated_sensedelay_poisson_text.mp4 simulated_sensedelay_poisson_text_stim.mp4 300 600

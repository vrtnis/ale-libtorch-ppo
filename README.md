# ALE-libtorch-PPO

A small C++ PPO trainer for Atari Breakout, based on the original
`ALE-libtorch-PPO` project by `cemlyn007`.

It uses LibTorch for the neural network, Arcade Learning Environment for Atari,
Bazel for builds, and FFmpeg for MP4 output. It talks to ALE directly rather
than going through Gymnasium.

You can train from scratch, save checkpoints, resume training, record clean
eval videos, and render annotated videos with policy/value graphs.

## Original Project

This repo is based on the original `ALE-libtorch-PPO` project by `cemlyn007`:

https://github.com/cemlyn007/ALE-libtorch-PPO

The original project provided the C++/LibTorch PPO implementation, ALE
environment integration, Bazel build, and Breakout training setup. This version
builds on that work with checkpointing, checkpoint eval, annotated view videos,
run scripts, and documentation updates.

## Sample Output

This is a tiny smoke-test video from the annotated view mode. It is useful for
checking that rendering works; it is not meant to represent a trained agent.

<video src="docs/assets/view_smoke.mp4" controls width="460"></video>

[Open the annotated smoke video](docs/assets/view_smoke.mp4)

## Setup

Install:

- Bazel or Bazelisk
- FFmpeg
- A Breakout ROM

The ROM is not included. Put it here:

```shell
mkdir -p roms
cp /path/to/breakout.bin roms/breakout.bin
```

## Build

```shell
bazel build --config=opt //src/bin:train
```

## Quick Smoke Run

This runs a small debug config so you can check that the build, ROM, ALE, and
FFmpeg are wired up:

```shell
./scripts/run_debug.sh
```

The smoke run writes logs and any configured videos under:

```text
logs/debug/
videos/debug/
```

## Train

For the main v0 config:

```shell
./scripts/train_v0.sh
```

The script is just a wrapper around:

```shell
bazel run --config=opt //src/bin:train -- \
  roms/breakout.bin \
  logs/train \
  videos/train \
  breakout-v0 \
  configs/v0.yaml
```

You can override paths without editing the script:

```shell
ROM_PATH=roms/breakout.bin CONFIG=configs/v1.yaml RUN_NAME=breakout-v1 ./scripts/train_v0.sh
```

## Checkpoints

Checkpointing is controlled in the YAML config:

```yaml
checkpoint_dir: "checkpoints/train"
checkpoint_every: 100
resume_checkpoint: ""
```

The trainer writes numbered checkpoints and keeps `latest.pt` updated in the
same directory.

To resume training, point `resume_checkpoint` at a saved checkpoint:

```yaml
resume_checkpoint: "checkpoints/train/latest.pt"
```

Checkpoints include the model, optimizer state, next rollout index, return
tracking state, and basic architecture metadata. They do not include live ALE
emulator state, so resumed runs restore training state but create fresh emulator
instances.

## Record A Clean Eval Video

Use eval mode when you want to see the current agent without training:

```shell
./scripts/eval_video.sh
```

The script wraps:

```shell
bazel run --config=opt //src/bin:train -- \
  eval \
  roms/breakout.bin \
  checkpoints/train/latest.pt \
  videos/eval \
  configs/v0.yaml \
  3
```

This writes:

```text
videos/eval/episode_1.mp4
videos/eval/episode_2.mp4
videos/eval/episode_3.mp4
```

## Record An Annotated View Video

Use view mode when you want a presentation-style video with graphs on the side:

```shell
./scripts/view_checkpoint.sh
```

The script wraps:

```shell
bazel run --config=opt //src/bin:train -- \
  view \
  roms/breakout.bin \
  checkpoints/train/latest.pt \
  videos/view \
  configs/v0.yaml \
  3
```

The side panel shows:

- critic value `V(s)`
- policy probabilities for each action
- selected action
- episode return
- step count

The bars are policy probabilities, not Q-values. This is a PPO actor-critic
model, so it learns a policy and a state-value function rather than a Q-function.

## Output Locations

Training logs:

```text
logs/
```

Recorded videos:

```text
videos/
```

Saved checkpoints:

```text
checkpoints/
```

The exact checkpoint path comes from `checkpoint_dir` in the YAML config.

## Troubleshooting

If the ROM is missing, put `breakout.bin` under `roms/` or set `ROM_PATH` when
running a script.

If video recording fails, check that `ffmpeg` is installed and available on
`PATH`.

If eval or view mode says the checkpoint is missing, train with checkpointing
enabled first, or set `CHECKPOINT=/path/to/latest.pt`.

If a checkpoint does not load, make sure the config matches the checkpoint. The
hidden size, action size, and frame stack need to match.

If a full training run feels too slow for iteration, start with
`./scripts/run_debug.sh` before moving to `configs/v0.yaml` or `configs/v1.yaml`.

## Useful Environment Overrides

The scripts use simple defaults, but you can override them from the shell:

```shell
ROM_PATH=roms/breakout.bin ./scripts/run_debug.sh
CHECKPOINT=checkpoints/train/latest.pt EPISODES=1 ./scripts/eval_video.sh
VIDEO_DIR=videos/view CONFIG=configs/v0.yaml ./scripts/view_checkpoint.sh
```

## Acknowledgements

- Original project: https://github.com/cemlyn007/ALE-libtorch-PPO
- [LibTorch](https://pytorch.org/cppdocs/)
- [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [Bazel](https://bazel.build/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl), which served as a useful PPO baseline

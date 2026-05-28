load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "train",
    srcs = ["train.cc"],
    deps = [
        "//src:ai",
        "//src/ai/environment",
        "//src/ai/environment:episode_life",
        "//src/ai/environment:episode_observation_recorder",
        "//src/ai/environment:episode_recorder",
        "//src/ai/environment:fire_reset",
        "//src/ai/environment:max_and_skip",
        "//src/ai/environment:noop_reset",
        "//src/ai/environment:resize",
        "//src/ai/environment:truncate_on_episode_return",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
        "@yaml-cpp//:yaml-cpp",
    ],
)

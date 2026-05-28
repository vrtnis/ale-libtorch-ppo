#include "ai/ppo/train.h"
#include "ai/environment/environment.h"
#include "ai/environment/episode_life.h"
#include "ai/environment/episode_observation_recorder.h"
#include "ai/environment/episode_recorder.h"
#include "ai/environment/fire_reset.h"
#include "ai/environment/max_and_skip.h"
#include "ai/environment/noop_reset.h"
#include "ai/environment/resize.h"
#include "ai/environment/truncate_on_episode_return.h"
#include "ai/ppo/losses.h"
#include "ai/rollout.h"
#include "ai/vision.h"
#include "tensorboard_logger.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <torch/nn.h>
#include <torch/torch.h>
#include <vector>
#include <yaml-cpp/yaml.h>

const bool ANNEAL_ENTROPY_COEFFICIENT = false;

// This is being used for annealing the entropy coefficient
//  based on the average return.
static double sum = 0;
static double count = 1;

double get_average_return() { return sum / count; }

double get_annealed_entropy_coef(double entropy_coef) {
  // This is a simple annealing function that decreases the entropy
  // coefficient as the average return increases.
  // 864 is the maximum episodic return for playing breakout.
  if (!ANNEAL_ENTROPY_COEFFICIENT) {
    return entropy_coef;
  }
  return entropy_coef * (864.0 - get_average_return()) / 864.0;
}

struct Config {
  size_t total_environments;
  size_t hidden_size;
  const size_t action_size = 4;
  size_t horizon;
  size_t max_steps;
  size_t frame_stack;
  double learning_rate;
  float clip_param;
  float value_loss_coef;
  float entropy_coef;
  long num_epochs;
  long mini_batch_size;
  long num_mini_batches;
  float gae_discount;
  float gae_lambda;
  float max_gradient_norm;
  size_t num_rollouts;
  size_t num_workers;
  size_t worker_batch_size;
  size_t frame_skip;
  // Some games like breakout have a maximum return
  // which should be used to reset the environment.
  float max_return;
  // It is faster to record using the observation.
  // However the observation may be in grayscale.
  bool record_observation;
  bool record_video;
  bool cuda_graph;
  bool deterministic;
  std::filesystem::path checkpoint_dir;
  size_t checkpoint_every;
  std::filesystem::path resume_checkpoint;
};

google::protobuf::Value get_value(double value) {
  google::protobuf::Value val;
  val.set_number_value(value);
  return val;
}

google::protobuf::Value get_bool_value(bool value) {
  google::protobuf::Value val;
  val.set_bool_value(value);
  return val;
}

std::map<std::string, google::protobuf::Value>
get_parameters(const Config &config) {
  std::map<std::string, google::protobuf::Value> hparams;
  hparams["total_environments"] = get_value(config.total_environments);
  hparams["hidden_size"] = get_value(config.hidden_size);
  hparams["action_size"] = get_value(config.action_size);
  hparams["horizon"] = get_value(config.horizon);
  hparams["max_steps"] = get_value(config.max_steps);
  hparams["frame_stack"] = get_value(config.frame_stack);
  hparams["learning_rate"] = get_value(config.learning_rate);
  hparams["clip_param"] = get_value(config.clip_param);
  hparams["value_loss_coef"] = get_value(config.value_loss_coef);
  hparams["entropy_coef"] = get_value(config.entropy_coef);
  hparams["num_epochs"] = get_value(config.num_epochs);
  hparams["mini_batch_size"] = get_value(config.mini_batch_size);
  hparams["num_mini_batches"] = get_value(config.num_mini_batches);
  hparams["gae_discount"] = get_value(config.gae_discount);
  hparams["gae_lambda"] = get_value(config.gae_lambda);
  hparams["max_gradient_norm"] = get_value(config.max_gradient_norm);
  hparams["num_rollouts"] = get_value(config.num_rollouts);
  hparams["num_workers"] = get_value(config.num_workers);
  hparams["worker_batch_size"] = get_value(config.worker_batch_size);
  hparams["frame_skip"] = get_value(config.frame_skip);
  hparams["max_return"] = get_value(config.max_return);
  hparams["record_observation"] = get_bool_value(config.record_observation);
  hparams["record_video"] = get_bool_value(config.record_video);
  hparams["cuda_graph"] = get_bool_value(config.cuda_graph);
  hparams["deterministic"] = get_bool_value(config.deterministic);
  hparams["checkpoint_every"] = get_value(config.checkpoint_every);
  return hparams;
}

Config load_config(const std::filesystem::path &path) {
  Config config;
  YAML::Node node = YAML::LoadFile(path.string());
  config.total_environments = node["total_environments"].as<size_t>(512);
  config.hidden_size = node["hidden_size"].as<size_t>(512);
  config.horizon = node["horizon"].as<size_t>(128);
  config.max_steps = node["max_steps"].as<size_t>(108000);
  config.frame_stack = node["frame_stack"].as<size_t>(4);
  config.learning_rate = node["learning_rate"].as<double>(2.5e-4);
  config.clip_param = node["clip_param"].as<float>(0.1f);
  config.value_loss_coef = node["value_loss_coef"].as<float>(0.5f);
  config.entropy_coef = node["entropy_coef"].as<float>(0.01f);
  config.num_epochs = node["num_epochs"].as<long>(1);
  config.mini_batch_size = node["mini_batch_size"].as<long>(2048);
  config.num_mini_batches = node["num_mini_batches"].as<long>(32);
  config.gae_discount = node["gae_discount"].as<float>(0.99f);
  config.gae_lambda = node["gae_lambda"].as<float>(0.95f);
  config.max_gradient_norm = node["max_gradient_norm"].as<float>(0.5f);
  config.num_rollouts = node["num_rollouts"].as<size_t>(7000);
  config.num_workers = node["num_workers"].as<size_t>(16);
  config.worker_batch_size = node["worker_batch_size"].as<size_t>(32);
  config.frame_skip = node["frame_skip"].as<size_t>(4);
  config.max_return = node["max_return"].as<float>(-1.0f);
  config.record_observation = node["record_observation"].as<bool>(false);
  config.record_video = node["record_video"].as<bool>(false);
  config.cuda_graph = node["cuda_graph"].as<bool>(false);
  config.deterministic = node["deterministic"].as<bool>(false);
  config.checkpoint_dir = node["checkpoint_dir"].as<std::string>("");
  config.checkpoint_every = node["checkpoint_every"].as<size_t>(0);
  config.resume_checkpoint = node["resume_checkpoint"].as<std::string>("");
  return config;
}

bool checkpoint_enabled(const Config &config) {
  return !config.checkpoint_dir.empty() && config.checkpoint_every > 0;
}

torch::Tensor scalar_int64(int64_t value) {
  return torch::tensor(value, torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor scalar_float64(double value) {
  return torch::tensor(value, torch::TensorOptions().dtype(torch::kFloat64));
}

int64_t read_int64(torch::serialize::InputArchive &archive,
                   const std::string &key) {
  torch::Tensor value;
  archive.read(key, value, true);
  return value.item<int64_t>();
}

double read_float64(torch::serialize::InputArchive &archive,
                    const std::string &key) {
  torch::Tensor value;
  archive.read(key, value, true);
  return value.item<double>();
}

std::filesystem::path checkpoint_path(const std::filesystem::path &dir,
                                      size_t next_rollout_index) {
  std::ostringstream filename;
  filename << "checkpoint_rollout_" << std::setw(8) << std::setfill('0')
           << next_rollout_index << ".pt";
  return dir / filename.str();
}

struct CheckpointState {
  size_t next_rollout_index;
  double return_sum;
  double return_count;
};

template <typename T> float mean(const std::vector<T> &values) {
  if (values.empty())
    throw std::invalid_argument("Values vector is empty.");
  return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

float mean(const torch::Tensor &tensor, const torch::Tensor &mask) {
  auto masked_tensor = tensor.masked_select(mask);
  return masked_tensor.mean().item<float>();
}

std::vector<float> gather(const torch::Tensor &tensor,
                          const torch::Tensor &mask) {
  auto t =
      tensor.masked_select(mask).contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

std::vector<float> to_vector(const torch::Tensor &tensor) {
  auto t = tensor.contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const ai::ppo::train::Metrics &metrics, double lr) {
  if (!log.episode_returns.empty()) {
    logger.add_scalar("mean_episode_return", log.steps,
                      mean(log.episode_returns));
    logger.add_scalar("mean_episode_length", log.steps,
                      mean(log.episode_lengths));
    logger.add_histogram("episode_returns", log.steps, log.episode_returns);
    logger.add_histogram("episode_lengths", log.steps, log.episode_lengths);

    if (!log.game_returns.empty()) {
      logger.add_scalar("mean_game_return", log.steps, mean(log.game_returns));
      logger.add_scalar("mean_game_length", log.steps, mean(log.game_lengths));
      logger.add_histogram("game_returns", log.steps, log.game_returns);
      logger.add_histogram("game_lengths", log.steps, log.game_lengths);
    }
  }
  logger.add_scalar("mean_clipped_gradient", log.steps,
                    metrics.clipped_gradients.mean().item<float>());
  logger.add_scalar("mean_loss", log.steps, metrics.loss.mean().item<float>());
  logger.add_scalar("mean_clipped_loss", log.steps,
                    mean(metrics.clipped_losses, metrics.masks));
  logger.add_scalar("mean_value_loss", log.steps,
                    mean(metrics.value_losses, metrics.masks));
  logger.add_scalar("mean_entropy", log.steps,
                    mean(metrics.entropies, metrics.masks));
  logger.add_scalar("mean_ratio", log.steps,
                    mean(metrics.ratio, metrics.masks));
  if (metrics.clipped_gradients.numel() > 1)
    logger.add_histogram("clipped_gradients", log.steps,
                         to_vector(metrics.clipped_gradients));
  logger.add_histogram("losses", log.steps,
                       gather(metrics.total_losses, metrics.masks));
  logger.add_histogram("clipped_losses", log.steps,
                       gather(metrics.clipped_losses, metrics.masks));
  logger.add_histogram("value_losses", log.steps,
                       gather(metrics.value_losses, metrics.masks));
  logger.add_histogram("entropies", log.steps,
                       gather(metrics.entropies, metrics.masks));
  logger.add_histogram("ratios", log.steps,
                       gather(metrics.ratio, metrics.masks));
  logger.add_histogram("advantages", log.steps,
                       gather(metrics.advantages, metrics.masks));
  logger.add_histogram("returns", log.steps,
                       gather(metrics.returns, metrics.masks));

  logger.add_scalar("learning_rate", log.steps, lr);
}

torch::nn::Conv2d layer_init(torch::nn::Conv2d layer,
                             double std = std::sqrt(2.0), double bias = 0.0) {
  torch::nn::init::orthogonal_(layer->weight, std);
  if (layer->bias.defined()) {
    torch::nn::init::constant_(layer->bias, bias);
  }
  return layer;
}

torch::nn::Linear layer_init(torch::nn::Linear layer,
                             double std = std::sqrt(2.0), double bias = 0.0) {
  torch::nn::init::orthogonal_(layer->weight, std);
  if (layer->bias.defined()) {
    torch::nn::init::constant_(layer->bias, bias);
  }
  return layer;
}

struct NetworkImpl : torch::nn::Module {
  NetworkImpl(size_t hidden_size, size_t action_size)
      : sequential(layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(4, 32, 8).stride(4))),
                   torch::nn::ReLU(),
                   layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(32, 64, 4).stride(2))),
                   torch::nn::ReLU(),
                   layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(64, 64, 3).stride(1))),
                   torch::nn::ReLU(), torch::nn::Flatten(),
                   layer_init(torch::nn::Linear(64 * 7 * 7, hidden_size))),
        action_head(
            layer_init(torch::nn::Linear(hidden_size, action_size), 0.01)),
        value_head(layer_init(torch::nn::Linear(hidden_size, 1), 1)) {
    register_module("sequential", sequential);
    register_module("action_head", action_head);
    register_module("value_head", value_head);
  }

  struct OutputType {
    torch::Tensor logits;
    torch::Tensor value;
  };

  OutputType forward(torch::Tensor x) {
    {
      torch::NoGradGuard no_grad;
      x = x.to(torch::kFloat32);
      x.divide_(255.0);
    }
    x = sequential->forward(x);
    auto logits = action_head->forward(x);
    auto value = value_head->forward(x).squeeze(-1);
    return {logits, value};
  }

  torch::nn::Sequential sequential;
  torch::nn::Linear action_head, value_head;
};
TORCH_MODULE(Network);

template <typename Optimizer>
void save_checkpoint(const std::filesystem::path &path, Network &network,
                     Optimizer &optimizer, const Config &config,
                     size_t next_rollout_index) {
  if (!std::filesystem::exists(path.parent_path())) {
    std::filesystem::create_directories(path.parent_path());
  }

  torch::serialize::OutputArchive archive;
  torch::serialize::OutputArchive model_archive;
  torch::serialize::OutputArchive optimizer_archive;

  network->save(model_archive);
  optimizer.save(optimizer_archive);

  archive.write("checkpoint_version", scalar_int64(1), true);
  archive.write("next_rollout_index",
                scalar_int64(static_cast<int64_t>(next_rollout_index)), true);
  archive.write("num_rollouts",
                scalar_int64(static_cast<int64_t>(config.num_rollouts)), true);
  archive.write("hidden_size",
                scalar_int64(static_cast<int64_t>(config.hidden_size)), true);
  archive.write("action_size",
                scalar_int64(static_cast<int64_t>(config.action_size)), true);
  archive.write("frame_stack",
                scalar_int64(static_cast<int64_t>(config.frame_stack)), true);
  archive.write("return_sum", scalar_float64(sum), true);
  archive.write("return_count", scalar_float64(count), true);
  archive.write("model", model_archive);
  archive.write("optimizer", optimizer_archive);

  const auto temporary_path = path.string() + ".tmp";
  archive.save_to(temporary_path);
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
  std::filesystem::rename(temporary_path, path);
}

template <typename Optimizer>
CheckpointState load_checkpoint(const std::filesystem::path &path,
                                Network &network, Optimizer &optimizer,
                                const Config &config,
                                const torch::Device &device) {
  if (!std::filesystem::exists(path)) {
    throw std::invalid_argument("Checkpoint file does not exist: " +
                                path.string());
  }

  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), device);

  const auto version = read_int64(archive, "checkpoint_version");
  if (version != 1) {
    throw std::runtime_error("Unsupported checkpoint version: " +
                             std::to_string(version));
  }

  const auto hidden_size = read_int64(archive, "hidden_size");
  const auto action_size = read_int64(archive, "action_size");
  const auto frame_stack = read_int64(archive, "frame_stack");
  if (hidden_size != static_cast<int64_t>(config.hidden_size) ||
      action_size != static_cast<int64_t>(config.action_size) ||
      frame_stack != static_cast<int64_t>(config.frame_stack)) {
    throw std::runtime_error(
        "Checkpoint architecture does not match the loaded config.");
  }

  torch::serialize::InputArchive model_archive;
  torch::serialize::InputArchive optimizer_archive;
  archive.read("model", model_archive);
  archive.read("optimizer", optimizer_archive);
  network->load(model_archive);
  network->to(device);
  optimizer.load(optimizer_archive);

  return {.next_rollout_index =
              static_cast<size_t>(read_int64(archive, "next_rollout_index")),
          .return_sum = read_float64(archive, "return_sum"),
          .return_count = read_float64(archive, "return_count")};
}

ai::ppo::train::Batch prepare_batch(ai::buffer::Batch &batch) {
  auto observations = batch.observations.flatten(0, 1);
  auto actions = batch.actions.ravel();
  auto advantages = batch.advantages.ravel();
  auto logits = batch.logits.view({-1, batch.logits.size(2)});
  auto returns = batch.returns.ravel();
  auto masks = batch.masks.ravel();
  auto log_probabilities = ai::ppo::losses::normalize_logits(logits);
  ai::ppo::train::Batch other_batch = {observations, actions, log_probabilities,
                                       advantages,   returns, masks};
  return other_batch;
}

ai::ppo::train::Hyperparameters prepare_hyperparameters(const Config &config) {
  ai::ppo::train::Hyperparameters hp = {
      config.clip_param, config.value_loss_coef,
      static_cast<float>(get_annealed_entropy_coef(config.entropy_coef)),
      config.max_gradient_norm};
  return hp;
}

void enable_torch_determinism(uint64_t seed) {
  // As per the logged warning by LibTorch: "Warning: Deterministic behavior was
  // enabled with either `torch.use_deterministic_algorithms(True)` or
  // `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
  // deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
  // deterministic behavior in this case, you must set an environment variable
  // before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
  // CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
  // https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  // (function alertCuBLASConfigNotDeterministic)"
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);

  torch::manual_seed(seed);

  // Enable deterministic algorithms, throw errors for non-deterministic
  // operations
  torch::globalContext().setDeterministicAlgorithms(true, true);

  // If using CUDA, ensure CuDNN is deterministic
  if (torch::cuda::is_available()) {
    torch::globalContext().setDeterministicCuDNN(true);
  }

  // Optionally, enable filling uninitialized memory for additional determinism
  torch::globalContext().setDeterministicFillUninitializedMemory(true);
}

void print_usage(const char *executable) {
  std::cerr << "Usage:\n"
            << "  " << executable
            << " <rom_path> <logger_path> <video_path> <group_name>"
               " <config_path> [profile_path]\n"
            << "  " << executable
            << " eval <rom_path> <checkpoint_path> <video_path>"
               " <config_path> [episodes]\n";
}

torch::Device select_device(const std::string &operation) {
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! " << operation << " on GPU."
              << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "CUDA is not available! " << operation << " on CPU."
              << std::endl;
  }
#ifdef __APPLE__
  std::cout << "Using MPS backend on macOS." << std::endl;
  device = torch::Device(torch::kMPS);
#endif
  return device;
}

std::unique_ptr<ai::environment::VirtualEnvironment> create_eval_environment(
    const std::filesystem::path &rom_path, const Config &config,
    const std::filesystem::path &video_path, size_t seed) {
  std::unique_ptr<ai::environment::VirtualEnvironment> environment =
      std::make_unique<ai::environment::Environment>(
          rom_path, config.max_steps, true, static_cast<int>(seed));

  if (config.max_return > 0.0f) {
    environment =
        std::make_unique<ai::environment::TruncateOnEpisodeReturnEnvironment>(
            std::move(environment), config.max_return);
  }

  environment = std::make_unique<ai::environment::ResizeEnvironment>(
      std::move(environment), 84, 84);

  if (config.record_observation) {
    environment =
        std::make_unique<ai::environment::EpisodeObservationRecorder>(
            std::move(environment), video_path, 1, 84, 84);
  } else {
    environment = std::make_unique<ai::environment::EpisodeRecorder>(
        std::move(environment), video_path, false);
  }

  environment = std::make_unique<ai::environment::NoopResetEnvironment>(
      std::move(environment), 30, seed);
  environment = std::make_unique<ai::environment::MaxAndSkipEnvironment>(
      std::move(environment), config.frame_skip);
  environment =
      std::make_unique<ai::environment::EpisodeLife>(std::move(environment));
  environment =
      std::make_unique<ai::environment::FireReset>(std::move(environment));
  return environment;
}

void reset_frame_stack(
    std::vector<ai::environment::ScreenBuffer> &frame_stack,
    const ai::environment::ScreenBuffer &observation) {
  std::fill(frame_stack.begin(), frame_stack.end(), observation);
}

void push_frame(std::vector<ai::environment::ScreenBuffer> &frame_stack,
                const ai::environment::ScreenBuffer &observation) {
  if (frame_stack.empty()) {
    throw std::runtime_error("Frame stack must not be empty.");
  }
  for (size_t frame_index = frame_stack.size() - 1; frame_index > 0;
       --frame_index) {
    frame_stack[frame_index] = frame_stack[frame_index - 1];
  }
  frame_stack[0] = observation;
}

torch::Tensor make_eval_observation_tensor(
    const std::vector<ai::environment::ScreenBuffer> &frame_stack,
    const torch::Device &device) {
  if (frame_stack.empty()) {
    throw std::runtime_error("Frame stack must not be empty.");
  }

  std::vector<torch::Tensor> frame_tensors;
  frame_tensors.reserve(frame_stack.size());
  for (const auto &frame : frame_stack) {
    if (frame.size() != 84 * 84) {
      throw std::runtime_error("Unexpected eval frame size.");
    }
    frame_tensors.push_back(
        torch::from_blob(const_cast<unsigned char *>(frame.data()), {84, 84},
                         torch::TensorOptions().dtype(torch::kByte))
            .clone());
  }
  return torch::stack(frame_tensors, 0).unsqueeze(0).to(device);
}

int run_eval(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    print_usage(argv[0]);
    return 1;
  }

  const auto rom_path = std::filesystem::path(argv[2]);
  const auto checkpoint_path = std::filesystem::path(argv[3]);
  const auto video_path = std::filesystem::path(argv[4]);
  const auto config = load_config(std::filesystem::path(argv[5]));
  size_t episodes = 3;
  if (argc == 7) {
    episodes = static_cast<size_t>(std::stoull(argv[6]));
  }
  if (episodes == 0) {
    throw std::invalid_argument("Eval episodes must be greater than 0.");
  }
  if (config.frame_stack == 0) {
    throw std::invalid_argument("Frame stack must be greater than 0.");
  }

  if (!std::filesystem::exists(video_path)) {
    std::filesystem::create_directories(video_path);
  }

  if (config.deterministic) {
    enable_torch_determinism(42);
  }

  torch::Device device = select_device("Evaluating");
  Network network(config.hidden_size, config.action_size);
  network->to(device);
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(config.learning_rate).eps(1e-5));

  const auto checkpoint_state =
      load_checkpoint(checkpoint_path, network, optimizer, config, device);
  network->eval();

  auto environment = create_eval_environment(rom_path, config, video_path, 0);
  const auto action_set = environment->get_interface().getMinimalActionSet();
  if (action_set.size() != config.action_size) {
    throw std::runtime_error(
        "Config action size does not match the ALE minimal action set.");
  }

  std::cout << "Loaded checkpoint " << checkpoint_path << " at rollout "
            << checkpoint_state.next_rollout_index << std::endl;
  std::cout << "Recording " << episodes
            << " deterministic eval episode(s) to " << video_path
            << std::endl;

  std::vector<ai::environment::ScreenBuffer> frame_stack(config.frame_stack);
  for (size_t episode_index = 0; episode_index < episodes; ++episode_index) {
    auto observation = environment->reset();
    reset_frame_stack(frame_stack, observation);

    float episode_return = 0.0f;
    size_t episode_steps = 0;
    while (true) {
      torch::NoGradGuard no_grad;
      const auto observation_tensor =
          make_eval_observation_tensor(frame_stack, device);
      const auto output = network->forward(observation_tensor);
      const auto action_index = output.logits.argmax(-1).item<int64_t>();
      if (action_index < 0 ||
          action_index >= static_cast<int64_t>(action_set.size())) {
        throw std::out_of_range("Policy selected an invalid action index.");
      }

      const auto step =
          environment->step(action_set[static_cast<size_t>(action_index)]);
      episode_return += step.reward;
      episode_steps++;

      if (step.game_over) {
        break;
      }

      if (step.terminated || step.truncated) {
        observation = environment->reset();
        reset_frame_stack(frame_stack, observation);
      } else {
        push_frame(frame_stack, step.observation);
      }
    }

    std::cout << "Eval episode " << episode_index + 1 << " return "
              << episode_return << " length " << episode_steps << std::endl;
  }

  std::cout << "Success" << std::endl;
  return 0;
}

int run_train(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    print_usage(argv[0]);
    return 1;
  }

  const auto start_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const auto rom_path = std::filesystem::path(argv[1]);
  const auto logger_path = std::filesystem::path(argv[2]).replace_extension(
      "tfevents." + std::to_string(start_time));
  const auto config = load_config(std::filesystem::path(argv[5]));
  const std::optional<std::filesystem::path> video_path =
      config.record_video
          ? std::optional<std::filesystem::path>(std::filesystem::path(argv[3]))
          : std::nullopt;
  const std::string group_name = argv[4];
  std::filesystem::path profile_path;
  if (argc == 7) {
    profile_path = std::filesystem::path(argv[6]);
  }
  torch::Device device = select_device("Training");

  if (!std::filesystem::exists(logger_path.parent_path())) {
    std::filesystem::create_directories(logger_path.parent_path());
  }
  if (video_path.has_value() && !std::filesystem::exists(video_path.value())) {
    std::filesystem::create_directories(video_path.value());
  }
  if (checkpoint_enabled(config) &&
      !std::filesystem::exists(config.checkpoint_dir)) {
    std::filesystem::create_directories(config.checkpoint_dir);
  }

  if (config.deterministic)
    enable_torch_determinism(42);

  TensorBoardLogger logger(logger_path);
  Network network(config.hidden_size, config.action_size);
  network->to(device);
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(config.learning_rate).eps(1e-5));

  size_t start_rollout_index = 0;
  if (!config.resume_checkpoint.empty()) {
    const auto checkpoint_state =
        load_checkpoint(config.resume_checkpoint, network, optimizer, config,
                        device);
    start_rollout_index = checkpoint_state.next_rollout_index;
    sum = checkpoint_state.return_sum;
    count = checkpoint_state.return_count;
    std::cout << "Resumed checkpoint " << config.resume_checkpoint
              << " at rollout " << start_rollout_index << std::endl;
    if (start_rollout_index >= config.num_rollouts) {
      std::cout << "Checkpoint has already reached the configured rollout "
                   "count."
                << std::endl;
      std::cout << "Success" << std::endl;
      return 0;
    }
  }

  ai::rollout::Rollout rollout(
      rom_path, config.total_environments, config.horizon, config.max_steps,
      config.frame_stack, true,
      [&network, &device, action_size = config.action_size](
          const torch::Tensor &obs) -> ai::rollout::ActionResult {
        network->eval();
        torch::NoGradGuard no_grad;
        auto observations = device.is_cuda() ? obs.to(torch::kFloat32) : obs;
        auto output = network->forward(observations.to(device));
        auto logits = output.logits;
        auto probabilities = torch::nn::functional::softmax(logits, -1);
        auto actions = torch::multinomial(probabilities, 1, true);
        return {actions.ravel(),
                logits.reshape({-1, static_cast<long>(action_size)}),
                output.value.ravel()};
      },
      config.gae_discount, config.gae_lambda, device, 0, config.num_workers,
      config.worker_batch_size, config.frame_skip, config.max_return,
      video_path, config.record_observation);
  torch::Tensor indices =
      torch::empty(config.mini_batch_size * config.num_mini_batches,
                   torch::TensorOptions().dtype(torch::kLong).device(device));
  ai::ppo::train::Metrics metrics(config.num_epochs, config.num_mini_batches,
                                  config.mini_batch_size, device);

  logger.add_hparams(get_parameters(config), group_name, start_time);

  ai::buffer::Batch b;
  {
    torch::NoGradGuard no_grad;
    b = rollout.rollout().batch;
  }
  ai::ppo::train::Batch batch = prepare_batch(b);
  ai::rollout::RolloutResult result;

#if defined(__linux__) && defined(AI_ENABLE_CUDA_GRAPHS)
  at::cuda::CUDAGraph graph;
  network->train();
  if (config.cuda_graph) {
    auto hp = prepare_hyperparameters(config);
    ai::ppo::train::capture_train_cuda_graph(graph, network, optimizer, metrics,
                                             indices, batch, config.num_epochs,
                                             config.num_mini_batches, hp, 10);
  }
#endif
  if (!profile_path.empty()) {
    torch::autograd::profiler::ProfilerConfig profiler_config =
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::KINETO);
    auto activities = {torch::autograd::profiler::ActivityType::CUDA,
                       torch::autograd::profiler::ActivityType::CPU};
    torch::autograd::profiler::prepareProfiler(profiler_config, activities);
    torch::autograd::profiler::enableProfiler(
        profiler_config, activities,
        {torch::RecordScope::FUNCTION, torch::RecordScope::USER_SCOPE});
  }
  for (size_t rollout_index = start_rollout_index;
       rollout_index < config.num_rollouts; ++rollout_index) {
    std::cout << "Rollout " << rollout_index + 1 << " of "
              << config.num_rollouts << std::endl;
    auto lr = config.learning_rate *
              (1.0 - rollout_index / static_cast<double>(config.num_rollouts));
    static_cast<torch::optim::AdamOptions &>(
        optimizer.param_groups()[0].options())
        .lr(lr);

    {
      torch::NoGradGuard no_grad;
      result = rollout.rollout();
    }
    if (config.cuda_graph) {
#if defined(__linux__) && defined(AI_ENABLE_CUDA_GRAPHS)
      auto b = prepare_batch(result.batch);
      batch.copy_(b);
      ai::ppo::train::train_cuda_graph(graph);
#else
      TORCH_CHECK(false, "cuda_graph support was not enabled for this build. "
                         "Set cuda_graph=false or compile with "
                         "AI_ENABLE_CUDA_GRAPHS and CUDA toolkit headers.");
#endif

    } else {
      batch = prepare_batch(result.batch);
      auto hp = prepare_hyperparameters(config);
      ai::ppo::train::train(network, optimizer, metrics, indices, batch,
                            config.num_epochs, config.num_mini_batches, hp);
    }

    log_data(logger, result.log, metrics,
             static_cast<torch::optim::AdamOptions &>(
                 optimizer.param_groups()[0].options())
                 .lr());
    sum += std::accumulate(result.log.episode_returns.begin(),
                           result.log.episode_returns.end(), 0.0f);
    count += result.log.episode_returns.size();

    const auto next_rollout_index = rollout_index + 1;
    if (checkpoint_enabled(config) &&
        (next_rollout_index % config.checkpoint_every == 0 ||
         next_rollout_index == config.num_rollouts)) {
      const auto numbered_path =
          checkpoint_path(config.checkpoint_dir, next_rollout_index);
      const auto latest_path = config.checkpoint_dir / "latest.pt";
      save_checkpoint(numbered_path, network, optimizer, config,
                      next_rollout_index);
      std::filesystem::copy_file(
          numbered_path, latest_path,
          std::filesystem::copy_options::overwrite_existing);
      std::cout << "Saved checkpoint " << numbered_path << std::endl;
    }
  }
  if (!profile_path.empty()) {
    auto profiler_result = torch::autograd::profiler::disableProfiler();
    profiler_result->save(profile_path);
  }
  std::cout << "Success" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "eval") {
    return run_eval(argc, argv);
  }
  return run_train(argc, argv);
}

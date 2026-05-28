#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ai::view_renderer {

struct ViewFrame {
  const std::vector<unsigned char> &game_rgb;
  size_t game_width;
  size_t game_height;
  float value;
  std::vector<float> action_probabilities;
  size_t selected_action;
  std::vector<std::string> action_labels;
  float episode_return;
  size_t episode_step;
};

class ViewRenderer {
public:
  ViewRenderer(size_t game_width, size_t game_height, size_t panel_width);

  void reset();
  std::vector<unsigned char> render(const ViewFrame &frame);

  size_t width() const;
  size_t height() const;

private:
  size_t game_width_;
  size_t game_height_;
  size_t panel_width_;
  size_t width_;
  size_t height_;
  std::vector<float> value_history_;
};

} // namespace ai::view_renderer

#include "ai/view_renderer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ai::view_renderer {
namespace {

struct Color {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

using Glyph = std::array<const char *, 7>;

Glyph glyph_for(char ch) {
  switch (static_cast<char>(std::toupper(static_cast<unsigned char>(ch)))) {
  case 'A':
    return {" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"};
  case 'B':
    return {"#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "};
  case 'C':
    return {" ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####"};
  case 'D':
    return {"#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "};
  case 'E':
    return {"#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"};
  case 'F':
    return {"#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "};
  case 'G':
    return {" ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ####"};
  case 'H':
    return {"#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"};
  case 'I':
    return {"#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"};
  case 'J':
    return {"#####", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "};
  case 'K':
    return {"#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"};
  case 'L':
    return {"#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"};
  case 'M':
    return {"#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"};
  case 'N':
    return {"#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"};
  case 'O':
    return {" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "};
  case 'P':
    return {"#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "};
  case 'Q':
    return {" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"};
  case 'R':
    return {"#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"};
  case 'S':
    return {" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "};
  case 'T':
    return {"#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "};
  case 'U':
    return {"#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "};
  case 'V':
    return {"#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "};
  case 'W':
    return {"#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"};
  case 'X':
    return {"#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"};
  case 'Y':
    return {"#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "};
  case 'Z':
    return {"#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"};
  case '0':
    return {" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "};
  case '1':
    return {"  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"};
  case '2':
    return {" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"};
  case '3':
    return {"#### ", "    #", "    #", " ### ", "    #", "    #", "#### "};
  case '4':
    return {"#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"};
  case '5':
    return {"#####", "#    ", "#    ", "#### ", "    #", "    #", "#### "};
  case '6':
    return {" ####", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "};
  case '7':
    return {"#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "};
  case '8':
    return {" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "};
  case '9':
    return {" ### ", "#   #", "#   #", " ####", "    #", "    #", "#### "};
  case '.':
    return {"     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "};
  case '-':
    return {"     ", "     ", "     ", "#####", "     ", "     ", "     "};
  case ':':
    return {"     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "};
  case '(':
    return {"   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "};
  case ')':
    return {" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "};
  case '|':
    return {"  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "};
  case '/':
    return {"    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "};
  case '%':
    return {"##  #", "## # ", "  #  ", " #   ", "# ## ", "# ## ", "     "};
  default:
    return {"     ", "     ", "     ", "     ", "     ", "     ", "     "};
  }
}

std::string format_float(float value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string trim_label(const std::string &label, size_t max_chars) {
  if (label.size() <= max_chars) {
    return label;
  }
  return label.substr(0, max_chars);
}

class Canvas {
public:
  Canvas(size_t width, size_t height, Color fill)
      : width_(width), height_(height), data_(width * height * 3) {
    clear(fill);
  }

  void clear(Color color) {
    for (size_t i = 0; i < width_ * height_; ++i) {
      data_[i * 3] = color.r;
      data_[i * 3 + 1] = color.g;
      data_[i * 3 + 2] = color.b;
    }
  }

  void pixel(int x, int y, Color color) {
    if (x < 0 || y < 0 || x >= static_cast<int>(width_) ||
        y >= static_cast<int>(height_)) {
      return;
    }
    const size_t index = (static_cast<size_t>(y) * width_ +
                          static_cast<size_t>(x)) *
                         3;
    data_[index] = color.r;
    data_[index + 1] = color.g;
    data_[index + 2] = color.b;
  }

  void fill_rect(int x, int y, int width, int height, Color color) {
    for (int py = y; py < y + height; ++py) {
      for (int px = x; px < x + width; ++px) {
        pixel(px, py, color);
      }
    }
  }

  void stroke_rect(int x, int y, int width, int height, Color color) {
    line(x, y, x + width, y, color);
    line(x, y + height, x + width, y + height, color);
    line(x, y, x, y + height, color);
    line(x + width, y, x + width, y + height, color);
  }

  void line(int x0, int y0, int x1, int y1, Color color) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
      pixel(x0, y0, color);
      if (x0 == x1 && y0 == y1) {
        break;
      }
      const int error2 = 2 * error;
      if (error2 >= dy) {
        error += dy;
        x0 += sx;
      }
      if (error2 <= dx) {
        error += dx;
        y0 += sy;
      }
    }
  }

  void text(int x, int y, const std::string &text, Color color,
            int scale = 1) {
    int cursor = x;
    for (char ch : text) {
      const Glyph glyph = glyph_for(ch);
      for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
          if (glyph[static_cast<size_t>(row)][static_cast<size_t>(col)] !=
              ' ') {
            fill_rect(cursor + col * scale, y + row * scale, scale, scale,
                      color);
          }
        }
      }
      cursor += 6 * scale;
    }
  }

  void blit_rgb(size_t x, size_t y, size_t width, size_t height,
                const std::vector<unsigned char> &rgb) {
    if (rgb.size() != width * height * 3) {
      return;
    }
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        const size_t src = (row * width + col) * 3;
        const size_t dst = ((y + row) * width_ + x + col) * 3;
        data_[dst] = rgb[src];
        data_[dst + 1] = rgb[src + 1];
        data_[dst + 2] = rgb[src + 2];
      }
    }
  }

  std::vector<unsigned char> take() { return std::move(data_); }

private:
  size_t width_;
  size_t height_;
  std::vector<unsigned char> data_;
};

int scaled_y(float value, float min_value, float max_value, int y, int height) {
  const float range = std::max(max_value - min_value, 1e-6f);
  const float unit = std::clamp((value - min_value) / range, 0.0f, 1.0f);
  return y + height - static_cast<int>(std::round(unit * height));
}

} // namespace

ViewRenderer::ViewRenderer(size_t game_width, size_t game_height,
                           size_t panel_width)
    : game_width_(game_width), game_height_(game_height),
      panel_width_(panel_width), width_(game_width + panel_width),
      height_(game_height) {
  if (game_width_ == 0 || game_height_ == 0 || panel_width_ < 220) {
    throw std::invalid_argument("Invalid view renderer dimensions.");
  }
}

void ViewRenderer::reset() { value_history_.clear(); }

size_t ViewRenderer::width() const { return width_; }

size_t ViewRenderer::height() const { return height_; }

std::vector<unsigned char> ViewRenderer::render(const ViewFrame &frame) {
  constexpr Color black{0, 0, 0};
  constexpr Color white{248, 250, 252};
  constexpr Color panel{242, 246, 251};
  constexpr Color grid{210, 218, 230};
  constexpr Color ink{30, 41, 59};
  constexpr Color muted{100, 116, 139};
  constexpr Color blue{37, 99, 235};
  constexpr Color red{224, 74, 58};
  constexpr Color green{96, 132, 64};
  constexpr Color purple{105, 82, 240};
  constexpr Color magenta{197, 68, 225};
  constexpr Color gold{245, 158, 11};

  Canvas canvas(width_, height_, black);
  canvas.blit_rgb(0, 0, game_width_, game_height_, frame.game_rgb);
  canvas.fill_rect(static_cast<int>(game_width_), 0, static_cast<int>(panel_width_),
                   static_cast<int>(height_), panel);
  canvas.line(static_cast<int>(game_width_), 0, static_cast<int>(game_width_),
              static_cast<int>(height_ - 1), white);

  value_history_.push_back(frame.value);
  constexpr size_t max_history = 180;
  if (value_history_.size() > max_history) {
    value_history_.erase(value_history_.begin(),
                         value_history_.begin() +
                             static_cast<long>(value_history_.size() -
                                               max_history));
  }

  const int panel_x = static_cast<int>(game_width_);
  const int chart_x = panel_x + 16;
  const int chart_y = 24;
  const int chart_w = static_cast<int>(panel_width_) - 32;
  const int chart_h = 58;
  canvas.text(panel_x + 16, 8, "STATE VALUE V", ink);
  canvas.stroke_rect(chart_x, chart_y, chart_w, chart_h, grid);
  for (int i = 1; i < 4; ++i) {
    const int gy = chart_y + i * chart_h / 4;
    canvas.line(chart_x, gy, chart_x + chart_w, gy, grid);
  }

  auto [min_it, max_it] =
      std::minmax_element(value_history_.begin(), value_history_.end());
  float min_value = *min_it;
  float max_value = *max_it;
  if (std::abs(max_value - min_value) < 1.0f) {
    min_value -= 0.5f;
    max_value += 0.5f;
  }
  canvas.text(chart_x + 4, chart_y + 4, format_float(max_value, 1), muted);
  canvas.text(chart_x + 4, chart_y + chart_h - 11, format_float(min_value, 1),
              muted);
  canvas.text(chart_x + chart_w - 72, chart_y + 4,
              "V " + format_float(frame.value, 2), ink);

  if (value_history_.size() > 1) {
    for (size_t i = 1; i < value_history_.size(); ++i) {
      const int x0 = chart_x + static_cast<int>((i - 1) * chart_w /
                                                (value_history_.size() - 1));
      const int x1 =
          chart_x + static_cast<int>(i * chart_w / (value_history_.size() - 1));
      const int y0 =
          scaled_y(value_history_[i - 1], min_value, max_value, chart_y, chart_h);
      const int y1 =
          scaled_y(value_history_[i], min_value, max_value, chart_y, chart_h);
      canvas.line(x0, y0, x1, y1, blue);
    }
  }

  const int stats_y = 90;
  canvas.text(panel_x + 16, stats_y,
              "RETURN " + format_float(frame.episode_return, 1), ink);
  canvas.text(panel_x + 150, stats_y,
              "STEP " + std::to_string(frame.episode_step), ink);

  const int bars_title_y = 108;
  const int bars_y = 130;
  const int bars_h = 48;
  canvas.text(panel_x + 16, bars_title_y, "POLICY PI(A|S)", ink);

  const size_t action_count =
      std::min(frame.action_probabilities.size(), frame.action_labels.size());
  if (action_count > 0) {
    const std::array<Color, 6> palette{red, green, purple, magenta, gold, blue};
    const int gap = 8;
    const int available_w = static_cast<int>(panel_width_) - 32;
    const int bar_w =
        std::max(8, (available_w - gap * (static_cast<int>(action_count) - 1)) /
                        static_cast<int>(action_count));

    for (size_t i = 0; i < action_count; ++i) {
      const int x = panel_x + 16 + static_cast<int>(i) * (bar_w + gap);
      const float probability =
          std::clamp(frame.action_probabilities[i], 0.0f, 1.0f);
      const int h = std::max(1, static_cast<int>(std::round(probability * bars_h)));
      const int y = bars_y + bars_h - h;
      canvas.fill_rect(x, y, bar_w, h, palette[i % palette.size()]);
      canvas.stroke_rect(x, bars_y, bar_w, bars_h,
                         i == frame.selected_action ? black : grid);
      canvas.text(x, bars_y + bars_h + 6, trim_label(frame.action_labels[i], 5),
                  ink);
    }
  }

  if (frame.selected_action < frame.action_labels.size()) {
    canvas.text(panel_x + 16, static_cast<int>(height_) - 14,
                "ACTION " + trim_label(frame.action_labels[frame.selected_action], 12),
                ink);
  }

  return canvas.take();
}

} // namespace ai::view_renderer

#include <iostream>

void showProgressBar(float progress, std::string message = "") {
  const int barWidth = 70;
  int progress_percent = int(progress * 100.0);
  std::cout << message << progress_percent << "% [";
  int pos = static_cast<int>(barWidth * progress);
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) {
      std::cout << "=";
    } else if (i == pos) {
      std::cout << ">";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "] " << progress_percent << " %";
  if (progress_percent == 100) {
    std::cout << std::endl;
  } else {
    std::cout << "\r";
  }
  std::cout.flush();
}
#include "binary_utils.h"

#include <iomanip>
#include <iostream>

// Function to print the binary array in hexadecimal format
void PrintBinaryArrayInHex(const unsigned char *data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl; // Switch back to decimal for any future output
}
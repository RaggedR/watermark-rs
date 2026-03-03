//
//  detect_blind.cpp
//  Blind Watermark Detection (no original image required)
//
//  Based on detect.cpp by Nicholas Meinhold
//  Modified for blind detection - works without the original image
//
//  The key insight: Legendre sequences are pseudorandom and should be
//  uncorrelated with natural image content. When we correlate the frequency
//  domain of the marked image with the Legendre sequence, the watermark
//  produces a peak while the image content averages to ~zero.
//

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <fstream>

#include "WatermarkDetection.hpp"
#include "Utilities.hpp"

int main(int argc, const char* argv[]) {
  // Start total timer
  auto totalStart = std::chrono::high_resolution_clock::now();

  // Check args - only need the marked image (not the original!)
  // args are: file path for marked image, [optional threshold]
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: detect_blind <marked_image_path> [threshold]" << std::endl;
    std::cout << std::endl;
    std::cout << "BLIND DETECTION - no original image required." << std::endl;
    std::cout << std::endl;
    std::cout << "Default threshold: 4.0 (lower than non-blind due to image noise)" << std::endl;
    std::cout << "Try threshold 3.0-5.0 depending on embedding strength." << std::endl;
    return -1;
  }

  std::string markedFilePath = argv[1];

  // Default threshold is lower for blind detection
  // (image content adds noise to the correlation)
  double threshold = 4.0;
  if (argc == 3) {
    threshold = atof(argv[2]);
  }

  std::cout << "========================================" << std::endl;
  std::cout << "     BLIND WATERMARK DETECTION" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;
  std::cout << "Image: " << markedFilePath << std::endl;
  std::cout << "Threshold: " << threshold << std::endl;
  std::cout << std::endl;

  int p, k, maxX, maxY, imgRows, imgCols;
  double ms, peak2rmsVal, maxVal;
  std::vector<int> shifts;
  std::vector<double> confidences;

  // Time image loading
  auto loadStart = std::chrono::high_resolution_clock::now();

  // Read in the marked image only - NO ORIGINAL NEEDED
  cv::Mat marked = cv::imread(markedFilePath, cv::IMREAD_COLOR);

  if (marked.empty()) {
    std::cout << "ERROR: Could not read image: " << markedFilePath << std::endl;
    return -1;
  }

  auto loadEnd = std::chrono::high_resolution_clock::now();
  double timeImageLoad = std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();

  std::cout << "Image loaded: " << marked.cols << "x" << marked.rows << std::endl;

  imgRows = marked.rows;
  imgCols = marked.cols;

  // Calculate the largest prime for this image
  p = largestPrimeFor(marked);
  std::cout << "Largest prime: " << p << std::endl;
  std::cout << "Watermark coefficients: " << p << " x " << p << " = " << (p*p) << std::endl;
  std::cout << std::endl;

  // Convert image to HSV
  cv::Mat hsvMarked;
  cvtColor(marked, hsvMarked, cv::COLOR_BGR2HSV);

  // Create 1d array for luma values
  // KEY DIFFERENCE FROM NON-BLIND: We use the marked image directly!
  // No subtraction of original image.
  double* lumaArray = new double[imgCols * imgRows];

  for (int y = 0; y < imgRows; y++) {
    for (int x = 0; x < imgCols; x++) {
      // BLIND: Just use the marked image's luma directly
      lumaArray[y * imgCols + x] = hsvMarked.at<cv::Vec3b>(y, x).val[2] / 255.0;
    }
  }

  // Time extraction phase
  auto extractStart = std::chrono::high_resolution_clock::now();

  // Extract from frequency domain (DFT)
  std::cout << "Extracting frequency coefficients (DFT)..." << std::endl;
  double* extractedMark = new double[p * p];
  extractMark(imgRows, imgCols, p, p, lumaArray, extractedMark);

  auto extractEnd = std::chrono::high_resolution_clock::now();
  double timeExtraction = std::chrono::duration<double, std::milli>(extractEnd - extractStart).count();

  double* wmArray = new double[p * p];
  double* correlationVals = new double[p * p];

  // Time correlation phase
  auto corrStart = std::chrono::high_resolution_clock::now();

  std::cout << "Correlating with Legendre sequences..." << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(6) << "Seq"
            << std::setw(14) << "Peak/RMS"
            << std::setw(12) << "Status"
            << std::setw(10) << "Shift" << std::endl;
  std::cout << std::string(42, '-') << std::endl;

  // Perform detection for each family of arrays
  k = 1;
  int maxK = 100;  // Safety limit

  while (k <= maxK) {
    // Generate Legendre array for this k value
    generateArray(p, k, wmArray);

    // Correlate extracted coefficients with Legendre sequence
    fastCorrelation(p, p, extractedMark, wmArray, correlationVals);

    // Find peak value and position
    maxVal = 0.0;
    maxX = -1;
    maxY = -1;
    for (int y = 0; y < p; y++) {
      for (int x = 0; x < p; x++) {
        if (correlationVals[y * p + x] > maxVal) {
          maxVal = correlationVals[y * p + x];
          maxY = y;
          maxX = x;
        }
      }
    }

    // Calculate peak-to-RMS ratio
    ms = 0;
    for (int i = 0; i < p * p; i++) {
      ms += (correlationVals[i] * correlationVals[i]) / (p * p);
    }
    double rmsVal = sqrt(ms);
    peak2rmsVal = maxVal / rmsVal;

    // Print result for this sequence
    std::cout << std::setw(6) << k
              << std::setw(14) << std::fixed << std::setprecision(2) << peak2rmsVal;

    if (peak2rmsVal > threshold) {
      int shift = maxY * p + maxX;
      std::cout << std::setw(12) << "DETECTED"
                << std::setw(10) << shift << std::endl;
      shifts.push_back(shift);
      confidences.push_back(peak2rmsVal);
      k++;
    } else {
      std::cout << std::setw(12) << "below" << std::endl;
      break;  // Stop when signal drops below threshold
    }
  }

  auto corrEnd = std::chrono::high_resolution_clock::now();
  double timeCorrelation = std::chrono::duration<double, std::milli>(corrEnd - corrStart).count();

  std::cout << std::endl;

  // Decode the message
  std::string message = "";
  double minConfidence = 0.0;
  bool detected = false;

  if (shifts.size() > 0) {
    message = getASCII(shifts, p * p);
    detected = true;

    // Find minimum confidence (weakest link in the chain)
    minConfidence = confidences[0];
    for (double c : confidences) {
      if (c < minConfidence) minConfidence = c;
    }
  }

  // Calculate total time
  auto totalEnd = std::chrono::high_resolution_clock::now();
  double timeTotal = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

  // Print results
  std::cout << "========================================" << std::endl;
  std::cout << "              RESULTS" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;
  std::cout << "Watermark detected: " << (detected ? "YES" : "NO") << std::endl;

  if (detected) {
    std::cout << "Message: \"" << message << "\"" << std::endl;
    std::cout << "Message length: " << message.length() << " characters" << std::endl;
    std::cout << "Sequences decoded: " << shifts.size() << std::endl;
    std::cout << "Minimum confidence: " << std::fixed << std::setprecision(2) << minConfidence;
    std::cout << " (threshold was " << threshold << ")" << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "No watermark found above threshold " << threshold << std::endl;
    std::cout << "Try lowering the threshold if you expect a watermark." << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Timing:" << std::endl;
  std::cout << "  Image load:    " << std::fixed << std::setprecision(0) << timeImageLoad << " ms" << std::endl;
  std::cout << "  DFT extract:   " << timeExtraction << " ms" << std::endl;
  std::cout << "  Correlation:   " << timeCorrelation << " ms" << std::endl;
  std::cout << "  Total:         " << timeTotal << " ms" << std::endl;
  std::cout << std::endl;

  // Clean up
  delete[] lumaArray;
  delete[] extractedMark;
  delete[] wmArray;
  delete[] correlationVals;

  return detected ? 0 : 1;
}

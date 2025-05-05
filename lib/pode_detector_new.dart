import 'dart:collection';
import 'dart:math' as math;

import 'package:ai_detection/coordinates_translator.dart'
    as coordinates_translator;
import 'package:ai_detection/uitils/calculate.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class JumpDetector {
  // Threshold values - adjust based on testing
  static const double jumpHeightThreshold = 0.1; // Relative height increase
  static const double feetSeparationThreshold = 0.05; // Distance between feet
  static const double kneeBendThreshold =
      160.0; // Angle in degrees (straighter legs)

  // Moving average window size for smoothing
  static const int windowSize = 5;
  final Queue<double> heightHistory = Queue<double>();

  // Jump counting
  int jumpCount = 0;
  bool wasJumping = false;

  bool isJumping = false;
  double jumpThreshold = 30.0; // customize based on test
  double handSpeedThreshold = 15.0;
  int lastJumpTime = 0;

  double? lastLeftY, lastRightY;
  double? lastLeftWristX, lastRightWristX;

  // Cooldown to prevent multiple counts for a single jump
  int jumpCooldownFrames = 0;
  static const int jumpCooldownThreshold =
      10; // Number of frames to wait before detecting another jump

  // Frame boundary variables - default values, will be set based on screen size
  Rect frameBoundary = Rect.zero;
  bool isFrameActive = true; // Whether to use the frame for detection

  // Set frame boundary based on camera preview size
  // Takes both screen size and camera preview size to calculate accurate frame
  void setFrameBoundary(Size screenSize, {Size? previewSize}) {
    // If preview size is provided, use it to calculate the frame boundary that
    // accurately represents the visible camera area
    Size actualSize = previewSize ?? screenSize;

    double frameWidth = actualSize.width;
    double frameHeight = actualSize.height * 0.9;
    double offsetX = 0;
    double offsetY = 0;

    frameBoundary = Rect.fromLTWH(offsetX, offsetY, frameWidth, frameHeight);
    print("Frame boundary set to: $frameBoundary with size: $actualSize");
  }

  // Toggle frame activation
  void toggleFrameActivation() {
    isFrameActive = !isFrameActive;
  }

  // Check if a person is within the frame
  bool isPersonInFrame(Pose pose, Size screenSize, Size imageSize,
      InputImageRotation rotation, CameraLensDirection cameraLensDirection) {
    if (!isFrameActive) {
      return true; // If frame is not active, always return true
    }

    final landmarks = pose.landmarks;
    if (landmarks.isEmpty) return false;

    // Use nose, shoulders, and hips to determine if person is in frame
    final nose = pose.landmarks[PoseLandmarkType.nose];
    final leftShoulder = pose.landmarks[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose.landmarks[PoseLandmarkType.rightShoulder];
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final rightHip = pose.landmarks[PoseLandmarkType.rightHip];

    if (nose == null ||
        leftShoulder == null ||
        rightShoulder == null ||
        leftHip == null ||
        rightHip == null) return false;

    // Check if the key body points are within the frame
    bool noseInFrame = _isPointInFrame(
        nose, screenSize, imageSize, rotation, cameraLensDirection);
    bool shouldersInFrame = _isPointInFrame(leftShoulder, screenSize, imageSize,
            rotation, cameraLensDirection) &&
        _isPointInFrame(rightShoulder, screenSize, imageSize, rotation,
            cameraLensDirection);
    bool hipsInFrame = _isPointInFrame(
            leftHip, screenSize, imageSize, rotation, cameraLensDirection) &&
        _isPointInFrame(
            rightHip, screenSize, imageSize, rotation, cameraLensDirection);

    // Person is in frame if at least nose and one pair (shoulders or hips) are in frame
    return noseInFrame && (shouldersInFrame || hipsInFrame);
  }

  // Helper method to check if a landmark is within the frame boundary
  bool _isPointInFrame(PoseLandmark landmark, Size screenSize, Size imageSize,
      InputImageRotation rotation, CameraLensDirection cameraLensDirection) {
    // Import the coordinate translator functions here
    double translatedX = coordinates_translator.translateX(
        landmark.x, screenSize, imageSize, rotation, cameraLensDirection);
    double translatedY = coordinates_translator.translateY(
        landmark.y, screenSize, imageSize, rotation, cameraLensDirection);
    print(Offset(translatedX, translatedY));

    return frameBoundary.contains(Offset(translatedX, translatedY));
  }

  /// Detects if a person is jumping and counts jumps
  /// Returns the current jump count
  int detectAndCountJump(Pose pose,
      {Size? screenSize,
      Size? imageSize,
      InputImageRotation? rotation,
      CameraLensDirection? cameraLensDirection}) {
    // If screen size parameters are provided, check if person is in frame
    bool personInFrame = true;
    if (screenSize != null &&
        rotation != null &&
        cameraLensDirection != null &&
        imageSize != null) {
      personInFrame = isPersonInFrame(
          pose, screenSize, imageSize, rotation, cameraLensDirection);
    }

    // Only detect jumps if the person is in the frame or if frame is not active
    if (personInFrame) {
      bool isJumping = _detectJump(pose);

      // Handle jump counting with cooldown
      if (jumpCooldownFrames > 0) {
        jumpCooldownFrames--;
      }

      // Jump starts (transition from not jumping to jumping)
      if (isJumping && !wasJumping && jumpCooldownFrames == 0) {
        jumpCount++;
        jumpCooldownFrames = jumpCooldownThreshold;
      }

      // Update state
      wasJumping = isJumping;
    }

    return jumpCount;
  }

  bool _detectJump(Pose pose) {
    final landmarks = pose.landmarks;
    if (landmarks.isEmpty) {
      return false;
    }

    // Get key body points
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];
    final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee];
    final rightKnee = pose.landmarks[PoseLandmarkType.rightKnee];
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final rightHip = pose.landmarks[PoseLandmarkType.rightHip];
    final nose = pose.landmarks[PoseLandmarkType.nose];

    // Ensure we have all required landmarks
    if (leftAnkle == null ||
        rightAnkle == null ||
        leftKnee == null ||
        rightKnee == null ||
        leftHip == null ||
        rightHip == null ||
        nose == null) {
      return false;
    }

    // 1. Check feet off ground - compare y-coordinates
    // In image coordinates, smaller y means higher position (top of image)
    double avgAnkleY = (leftAnkle.y + rightAnkle.y) / 2;
    double hipY = (leftHip.y + rightHip.y) / 2;

    // Track relative body height (distance from hips to ankles)
    double bodyHeight = hipY - avgAnkleY;
    _updateHeightHistory(bodyHeight);

    // 2. Check for straight legs during jump (knees should be extended)
    double leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
    double rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
    bool legsExtended = (leftKneeAngle > kneeBendThreshold &&
        rightKneeAngle > kneeBendThreshold);

    // 3. Check feet separation - during jump, feet often move apart
    double feetDistance = _calculateDistance(leftAnkle, rightAnkle);
    double hipDistance = _calculateDistance(leftHip, rightHip);
    bool feetSeparated = (feetDistance / hipDistance) > feetSeparationThreshold;

    // 4. Check for significant height increase compared to baseline
    bool heightIncreased = _isHeightIncreased();

    // Combine conditions to detect jump
    // Core condition is height increase, with at least one supporting condition
    return heightIncreased && (legsExtended || feetSeparated);
  }

  /// Calculate distance between two landmarks
  double _calculateDistance(PoseLandmark first, PoseLandmark second) {
    return math.sqrt(
        math.pow(first.x - second.x, 2) + math.pow(first.y - second.y, 2));
  }

  bool _detectRopeSkipping(Pose pose) {
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

    final leftWrist = pose.landmarks[PoseLandmarkType.leftWrist];
    final rightWrist = pose.landmarks[PoseLandmarkType.rightWrist];

    if (leftAnkle == null ||
        rightAnkle == null ||
        leftWrist == null ||
        rightWrist == null) return false;

    final currentTime = DateTime.now().millisecondsSinceEpoch;

    // Calculate ankle jump height difference
    final avgAnkleY = (leftAnkle.y + rightAnkle.y) / 2;

    // Simple baseline check
    if (lastLeftY != null && lastRightY != null) {
      final jumpDelta = ((lastLeftY! + lastRightY!) / 2) - avgAnkleY;

      if (jumpDelta > jumpThreshold &&
          !isJumping &&
          (currentTime - lastJumpTime) > 800) {
        // Optionally: check wrist swing too
        if (_isWristMovingFast(leftWrist.x, rightWrist.x)) {
          isJumping = true;
          lastJumpTime = currentTime;
        }
      } else if (jumpDelta < 5) {
        isJumping = false; // Reset on landing
      }
    }

    // Update ankle Y positions
    lastLeftY = leftAnkle.y;
    lastRightY = rightAnkle.y;

    // Update wrist X for swing detection
    lastLeftWristX = leftWrist.x;
    lastRightWristX = rightWrist.x;

    return isJumping;
  }

  bool _isWristMovingFast(double leftX, double rightX) {
    if (lastLeftWristX == null || lastRightWristX == null) return false;

    final leftSpeed = (leftX - lastLeftWristX!).abs();
    final rightSpeed = (rightX - lastRightWristX!).abs();

    return leftSpeed > handSpeedThreshold && rightSpeed > handSpeedThreshold;
  }

  /// Update height history with moving average
  void _updateHeightHistory(double currentHeight) {
    heightHistory.add(currentHeight);
    if (heightHistory.length > windowSize) {
      heightHistory.removeFirst();
    }
  }

  /// Determine if current height shows significant increase
  bool _isHeightIncreased() {
    if (heightHistory.length < windowSize) {
      return false; // Need more data points
    }

    // Use the first few entries as baseline
    double baselineHeight = 0;
    int baselineCount = windowSize - 2;

    List<double> heightList = heightHistory.toList();
    for (int i = 0; i < baselineCount; i++) {
      baselineHeight += heightList[i];
    }
    baselineHeight /= baselineCount;

    // Check if current height shows significant increase
    double currentHeight = heightList.last;
    return currentHeight > baselineHeight * (1 + jumpHeightThreshold);
  }

  /// Reset the jump counter
  void resetJumpCount() {
    jumpCount = 0;
  }
}

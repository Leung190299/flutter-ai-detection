import 'dart:collection';
import 'dart:math' as math;
import 'package:ai_detection/uitils/calculate.dart';
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

  // Cooldown to prevent multiple counts for a single jump
  int jumpCooldownFrames = 0;
  static const int jumpCooldownThreshold =
      10; // Number of frames to wait before detecting another jump

  /// Detects if a person is jumping and counts jumps
  /// Returns the current jump count
  int detectAndCountJump(Pose pose) {
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

    return jumpCount;
  }

  /// Main method to determine if a person is jumping based on pose landmarks
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

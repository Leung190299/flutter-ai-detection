import 'dart:ui';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class RopeSkippingPoseDetector {
  double? filteredNoseY;
  double? previousY;
  String direction = 'down'; // current movement direction
  int jumpCount = 0;

// Parameters
  double alpha = 0.3; // low-pass filter factor
  double jumpThreshold = 5;
  double footLiftThreshold = 5;
  double? groundLevelY;
  void detectPose(Pose pose,
      {Size screenSize = Size.zero, Size imageSize = Size.zero}) {
    final landmarks = pose.landmarks;

    if (landmarks.isEmpty) {
      return;
    }

    final leftShoulder = landmarks[PoseLandmarkType.leftShoulder];
    final leftHip = landmarks[PoseLandmarkType.leftHip];

    final leftAnkle = landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = landmarks[PoseLandmarkType.rightAnkle];
    final nose = landmarks[PoseLandmarkType.nose];

    if (leftShoulder == null ||
        leftHip == null ||
        leftAnkle == null ||
        rightAnkle == null ||
        nose == null) {
      return;
    }
    double leftFootY = leftAnkle.y;
    double rightFootY = rightAnkle.y;

    if (groundLevelY == null) {
      groundLevelY = (leftFootY + rightFootY) / 2;
      return; // wait until next frame
    }
    bool feetLifted = (groundLevelY! - leftFootY > footLiftThreshold) ||
        (groundLevelY! - rightFootY > footLiftThreshold);

    double rawY = nose.y;
    double currentY = _getFilteredNoseY(rawY);

    if (previousY != null) {
      double deltaY = currentY - previousY!;

      if (deltaY < -jumpThreshold && direction != 'up' && feetLifted) {
        direction = 'up'; // nose is rising
      } else if (deltaY > jumpThreshold && direction == 'up' && feetLifted) {
        direction = 'down'; // nose has fallen back down
        jumpCount++;
        print('Jump Count: $jumpCount');
      }
    }

    previousY = currentY;
  }

  double _getFilteredNoseY(double currentY) {
    if (filteredNoseY == null) {
      filteredNoseY = currentY;
    } else {
      filteredNoseY = alpha * currentY + (1 - alpha) * filteredNoseY!;
    }
    return filteredNoseY!;
  }
}

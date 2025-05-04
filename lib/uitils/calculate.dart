import 'dart:math';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

calculateAngle(PoseLandmark a, PoseLandmark b, PoseLandmark c) {
  final radians =
      (atan2(c.y - a.y, c.x - a.x) - atan2(b.y - a.y, b.x - a.x)) * 180 / pi;

  double angle = radians.abs();
  if (angle > 180) {
    angle = 360 - angle;
  }
  return angle;
}

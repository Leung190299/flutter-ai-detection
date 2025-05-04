import 'package:ai_detection/pode_detector_new.dart';
import 'package:ai_detection/pose_painter.dart';
import 'package:ai_detection/uitils/calculate.dart';
import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'detector_view.dart';

class PoseDetectorView extends StatefulWidget {
  const PoseDetectorView({super.key});

  @override
  State<StatefulWidget> createState() => _PoseDetectorViewState();
}

class _PoseDetectorViewState extends State<PoseDetectorView> {
  final PoseDetector _poseDetector =
      PoseDetector(options: PoseDetectorOptions());
  bool _canProcess = true;
  bool _isBusy = false;
  CustomPaint? _customPaint;
  String? _text;
  var _cameraLensDirection = CameraLensDirection.back;
  final JumpDetector _jumpDetector = JumpDetector();

  // Rope skipping state variables
  int jumpCount = 0;
  String previousState = "on_ground";
  double previousY = 0.0;
  double jumpThreshold = 30.0; // adjust based on your environment

  @override
  void dispose() async {
    _canProcess = false;
    _poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DetectorView(
      title: 'Pose Detector',
      customPaint: _customPaint,
      text: _text,
      onImage: _processImage,
      initialCameraLensDirection: _cameraLensDirection,
      onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
    );
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_canProcess) return;
    if (_isBusy) return;
    _isBusy = true;
    setState(() {
      _text = '';
    });
    final poses = await _poseDetector.processImage(inputImage);
    if (poses.isNotEmpty) {
      final test = _jumpDetector.detectAndCountJump(poses.first);
      print("Jump Count: $test");
    }
    if (inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      final painter = PosePainter(
        poses,
        inputImage.metadata!.size,
        inputImage.metadata!.rotation,
        _cameraLensDirection,
      );
      _customPaint = CustomPaint(painter: painter);
    } else {
      _text = 'Poses found: ${poses.length}\n\n';
      // TODO: set _customPaint to draw landmarks on top of image
      _customPaint = null;
    }
    _isBusy = false;
    if (mounted) {
      setState(() {});
    }
  }

  // Define threshold for jump detection
  double angleThreshold = 9.0; // Angle threshold for detecting a jump

  // function count rope skipping with multiple body points
  int countRopeSkipping(List<Pose> poses) {
    int localCount = 0;
    for (var pose in poses) {
      // Check if all required landmarks are detected for both legs
      if (pose.landmarks[PoseLandmarkType.leftKnee] != null &&
          pose.landmarks[PoseLandmarkType.leftAnkle] != null &&
          pose.landmarks[PoseLandmarkType.leftFootIndex] != null &&
          pose.landmarks[PoseLandmarkType.rightKnee] != null &&
          pose.landmarks[PoseLandmarkType.rightAnkle] != null &&
          pose.landmarks[PoseLandmarkType.rightFootIndex] != null) {
        // Calculate angles for both legs
        double leftLegAngle = calculateAngle(
            pose.landmarks[PoseLandmarkType.leftKnee]!,
            pose.landmarks[PoseLandmarkType.leftAnkle]!,
            pose.landmarks[PoseLandmarkType.leftFootIndex]!);

        double rightLegAngle = calculateAngle(
            pose.landmarks[PoseLandmarkType.rightKnee]!,
            pose.landmarks[PoseLandmarkType.rightAnkle]!,
            pose.landmarks[PoseLandmarkType.rightFootIndex]!);

        // Use average angle of both legs
        double averageAngle = (leftLegAngle + rightLegAngle) / 2;

        switch (previousState) {
          case "on_ground":
            // When the angle is greater than threshold, person is likely in jumping position
            if (averageAngle > angleThreshold) {
              previousState = "in_air";
              localCount += 1;
              print(
                  "Jump Count: ${jumpCount + localCount} (Angle: $averageAngle)");
            }
            break;
          case "in_air":
            // When the angle becomes less than threshold, person has landed
            if (averageAngle < angleThreshold - 20) {
              // Add hysteresis to prevent false counts
              previousState = "on_ground";
            }
            break;
        }
      } else {
        // Handle the case where landmarks are not detected
        print('Required leg landmarks not detected');
      }
    }
    return localCount;
  }
}

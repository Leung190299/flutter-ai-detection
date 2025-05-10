import 'dart:ui';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class PoseImprovement {
  final Map<String, dynamic> thresholds = {
    'buffer_time': 50,
    'dy_ratio': 0.3,
    'up_ratio': 0.55,
    'down_ratio': 0.35,
    'flag_low': 150.0,
    'flag_high': 250.0,
  };
  late Map<String, BufferList> buffers = {
    'center_y': BufferList(thresholds['buffer_time']),
    'center_y_up': BufferList(thresholds['buffer_time']),
    'center_y_down': BufferList(thresholds['buffer_time']),
    'center_y_flip': BufferList(thresholds['buffer_time']),
    'center_y_pref_flip': BufferList(thresholds['buffer_time']),
  };

  // State variables
  double cyMax = 100.0;
  double cyMin = 100.0;
  double flipFlag = 250.0; // Starting with flag_high
  int count = 0;

  processPose(Pose pose,
      {Size screenSize = Size.zero, Size imageSize = Size.zero}) {
    final landmarks = pose.landmarks;

    if (landmarks.isNotEmpty && imageSize != Size.zero) {
      final hipPoints = _extractLandmarks(landmarks,
          [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip], imageSize);
      final shoulderPoints = _extractLandmarks(
          landmarks,
          [PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
          imageSize);

      if (hipPoints.isNotEmpty && shoulderPoints.isNotEmpty) {
        final cx = _calculateMeanX(hipPoints);
        final CyResult result = _calculateCenterY(hipPoints, shoulderPoints);
        final cy = result.cy;
        final cyShoulderHip = result.cyShoulderHip;

        _updateBuffers(cy, cyShoulderHip);
        _updateCounters(cy, cyShoulderHip);
      }
    }
  }

  CyResult _calculateCenterY(
      List<Offset> hipPoints, List<Offset> shoulderPoints) {
    double cyHip = 0;
    double cyShoulder = 0;

    for (final point in hipPoints) {
      cyHip += point.dy;
    }
    cyHip = cyHip / hipPoints.length;

    for (final point in shoulderPoints) {
      cyShoulder += point.dy;
    }
    cyShoulder = cyShoulder / shoulderPoints.length;

    return CyResult(cyHip, cyHip - cyShoulder);
  }

  double _calculateMeanX(List<Offset> points) {
    double sum = 0;
    for (final point in points) {
      sum += point.dx;
    }
    return sum / points.length;
  }

  List<Offset> _extractLandmarks(Map<PoseLandmarkType, PoseLandmark> landmarks,
      List<PoseLandmarkType> landmarkIndices, Size imageSize) {
    final result = <Offset>[];

    for (final type in landmarkIndices) {
      final landmark = landmarks[type];
      if (landmark != null) {
        result.add(Offset(
          landmark.x * imageSize.width,
          landmark.y * imageSize.height,
        ));
      }
    }

    return result;
  }

  void _updateBuffers(double cy, double cyShoulderHip) {
    buffers['center_y']!.push(cy);
    cyMax = buffers['center_y']!.smoothUpdate(cyMax, buffers['center_y']!.max);
    buffers['center_y_up']!.push(cyMax);
    cyMin = buffers['center_y']!.smoothUpdate(cyMin, buffers['center_y']!.min);
    buffers['center_y_down']!.push(cyMin);
  }

  void _updateCounters(double cy, double cyShoulderHip) {
    final double prevFlipFlag = flipFlag;
    flipFlag = _updateFlipFlag(cy, cyShoulderHip, cyMax, cyMin, flipFlag);

    buffers['center_y_flip']!.push(flipFlag);
    buffers['center_y_pref_flip']!.push(prevFlipFlag);

    if (prevFlipFlag < flipFlag) {
      count += 1;
      print('Jump Count: $count');
    }
  }

  double _updateFlipFlag(double cy, double cyShoulderHip, double cyMax,
      double cyMin, double currentFlag) {
    final dy = cyMax - cyMin;

    if (dy > thresholds['dy_ratio'] * cyShoulderHip) {
      if (cy > cyMax - thresholds['up_ratio'] * dy &&
          currentFlag == thresholds['flag_low']) {
        return thresholds['flag_high'];
      } else if (cy < cyMin + thresholds['down_ratio'] * dy &&
          currentFlag == thresholds['flag_high']) {
        return thresholds['flag_low'];
      }
    }

    return currentFlag;
  }
}

class BufferList {
  final int size;
  final List<double> _buffer;
  int _currentIndex = 0;

  BufferList(this.size)
      : _buffer = List<double>.filled(size, 0, growable: true);

  void push(double value) {
    if (_buffer.length < size) {
      _buffer.add(value);
    } else {
      _buffer[_currentIndex] = value;
      _currentIndex = (_currentIndex + 1) % size;
    }
  }

  double get max => _buffer.reduce((a, b) => a > b ? a : b);
  double get min => _buffer.reduce((a, b) => a < b ? a : b);
  double smoothUpdate(double oldValue, double newValue, {double alpha = 0.5}) {
    return alpha * newValue + (1 - alpha) * oldValue;
  }
}

class CyResult {
  final double cy;
  final double cyShoulderHip;

  CyResult(this.cy, this.cyShoulderHip);
}

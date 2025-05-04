import 'dart:io';
import 'dart:typed_data';
import 'dart:math' show min, max;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:developer' as developer;

class PoseDetectorService {
  static const int width = 192;
  static const int height = 256;
  late final OrtSession _session;
  bool _isInitialized = false;
  
  // RTMPose keypoint indices
  static const List<String> keypointNames = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
  ];

  // Skeleton connections for visualization
  static const List<List<int>> skeletonConnections = [
    [0, 1], [0, 2], [1, 3], [2, 4],  // Face
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // Arms
    [5, 11], [6, 12], [11, 12],  // Body
    [11, 13], [13, 15], [12, 14], [14, 16]  // Legs
  ];

  // Mean and std values for normalization (from RTMPose)
  static const List<double> mean = [123.675, 116.28, 103.53];
  static const List<double> std = [58.395, 57.12, 57.375];
  
  Future<void> initModel() async {
    try {
      final modelBytes = await rootBundle.load('assets/models/rtmpose-s-end2end.onnx');
      final modelBuffer = modelBytes.buffer;
      final sessionOptions = OrtSessionOptions();
      _session = await OrtSession.fromBuffer(modelBuffer.asUint8List(), sessionOptions);
      _isInitialized = true;
      developer.log('Model initialized successfully');
    } catch (e, stack) {
      developer.log('Error initializing pose detection model', error: e, stackTrace: stack);
      rethrow;
    }
  }

  Future<List<List<double>>> detectPose(String imagePath) async {
    if (!_isInitialized) {
      throw Exception('Model not initialized');
    }

    // Load and preprocess image
    final imageBytes = await rootBundle.load(imagePath);
    final image = img.decodeImage(imageBytes.buffer.asUint8List());
    if (image == null) throw Exception('Failed to load image');
    
    developer.log('Image loaded successfully: ${image.width}x${image.height}');

    // Calculate scale factor to preserve aspect ratio
    final scaleFactor = min(
      width / image.width.toDouble(),
      height / image.height.toDouble()
    );
    final newWidth = (image.width * scaleFactor).round();
    final newHeight = (image.height * scaleFactor).round();

    developer.log('Scale factor: $scaleFactor');
    developer.log('New dimensions: ${newWidth}x${newHeight}');

    // Resize image preserving aspect ratio
    final resized = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear
    );

    // Create padded image with gray background (114, 114, 114)
    final padded = img.Image(width: width, height: height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        padded.setPixel(x, y, img.ColorRgb8(114, 114, 114));
      }
    }

    // Copy resized image to center of padded image
    final offsetX = ((width - newWidth) / 2).round();
    final offsetY = ((height - newHeight) / 2).round();
    for (var y = 0; y < newHeight; y++) {
      for (var x = 0; x < newWidth; x++) {
        final pixel = resized.getPixel(x, y);
        padded.setPixel(x + offsetX, y + offsetY, pixel);
      }
    }

    // Convert to float32 and normalize with mean/std
    final inputData = Float32List(width * height * 3);
    var pixelIndex = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final pixel = padded.getPixel(x, y);
        // CHW format
        inputData[pixelIndex] = (pixel.r.toDouble() - mean[0]) / std[0];
        inputData[pixelIndex + width * height] = (pixel.g.toDouble() - mean[1]) / std[1];
        inputData[pixelIndex + 2 * width * height] = (pixel.b.toDouble() - mean[2]) / std[2];
        pixelIndex++;
      }
    }

    // Log input tensor info
    developer.log('Input tensor shape: [1, 3, $height, $width]');
    developer.log('Input tensor range: ${inputData.reduce(min)} to ${inputData.reduce(max)}');

    // Run inference
    final runOptions = OrtRunOptions();
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, height, width],
    );
    
    final inputs = {'input': inputTensor};
    final outputs = await _session.runAsync(runOptions, inputs);
    
    if (outputs == null || outputs.isEmpty) {
      throw Exception('No output from model');
    }

    // Process outputs
    final outputTensor = outputs[0];
    if (outputTensor == null) {
      throw Exception('Invalid output tensor');
    }
    
    // Get the raw tensor data and reshape it
    final tensorValue = outputTensor.value;
    developer.log('Output tensor type: ${tensorValue.runtimeType}');
    
    List<List<double>> keypoints = [];
    
    try {
      if (tensorValue is List) {
        developer.log('Processing tensor as List, first element type: ${tensorValue.isNotEmpty ? tensorValue[0].runtimeType : "empty"}');
        developer.log('Tensor value length: ${tensorValue.length}');
        
        if (tensorValue.isEmpty) {
          throw Exception('Empty tensor value');
        }

        // RTMPose outputs [1, 17, 3] tensor where:
        // - First dimension is batch size (1)
        // - Second dimension is number of keypoints (17)
        // - Third dimension is [x, y, confidence] for each keypoint
        if (tensorValue[0] is List) {
          developer.log('Processing as nested List');
          final points = tensorValue[0] as List;
          for (var i = 0; i < points.length; i++) {
            final point = points[i] as List;
            if (point.length >= 3) {
              // Model outputs coordinates directly in pixel space
              final x = (point[0] as num).toDouble();
              final y = (point[1] as num).toDouble();
              final confidence = (point[2] as num).toDouble();
              
              // Scale coordinates back to original image resolution
              final scaledX = (x - offsetX) / scaleFactor;
              final scaledY = (y - offsetY) / scaleFactor;
              
              keypoints.add([scaledX, scaledY, confidence]);
              
              developer.log('Keypoint ${keypointNames[i]}: [$scaledX, $scaledY, $confidence]');
            }
          }
        }
      }
      
      developer.log('Processed ${keypoints.length} keypoints');
      if (keypoints.isNotEmpty) {
        developer.log('First keypoint (nose): ${keypoints[0]}');
      } else {
        developer.log('No keypoints were processed from tensor');
      }
      
    } catch (e, stack) {
      developer.log('Error processing tensor output', error: e, stackTrace: stack);
      developer.log('Tensor value type: ${tensorValue.runtimeType}');
      if (tensorValue is List && tensorValue.isNotEmpty) {
        developer.log('First element type: ${tensorValue[0].runtimeType}');
        developer.log('First element: ${tensorValue[0]}');
      }
      rethrow;
    }

    // Clean up
    inputTensor.release();
    runOptions.release();
    if (outputs != null) {
      for (final output in outputs) {
        output?.release();
      }
    }

    return keypoints;
  }

  void dispose() {
    if (_isInitialized) {
      _session.release();
    }
  }
} 
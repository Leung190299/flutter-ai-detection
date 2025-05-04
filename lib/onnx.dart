import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class OnnxModel {
  final OrtSessionOptions sessionOptions = OrtSessionOptions();
  final String assetFileName = 'assets/models/end2end.onnx';
  late OrtSession? _session;
  final int height = 192; // Model expects height of 256
  final int width = 256; // Model expects width of 192
  final int channels = 3;

  OnnxModel() {
    _loadModel();
  }
  void _loadModel() async {
    try {
      // Load the model from the asset file
      final ByteData data = await rootBundle.load(assetFileName);
      final Uint8List modelData = data.buffer.asUint8List();

      // Create a session with the loaded model
      _session = OrtSession.fromBuffer(modelData, sessionOptions);
      print("Model loaded successfully");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<Uint8List> _loadImage(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List();
  }

  clearModel() {
    _session?.release();
    OrtEnv.instance.release();
  }
  // Function to run inference on the model

  Future<List<Offset>> runInference(CameraImage cameraImage) async {
    if (_session == null) {
      print("Session is not initialized");
      return const [];
    }

    try {
      // Convert CameraImage to Image format more efficiently
      final img.Image? image = await convertCameraImageToImage(cameraImage);
      if (image == null) {
        print("Failed to convert camera image");
        return const [];
      }

      // Resize image using computed parameters
      final resizedImage = img.copyResize(
        image,
        width: width,
        height: height,
        interpolation: img.Interpolation.linear,
      );

      // Convert to float32 data
      final inputData = imageToFloat32List(resizedImage);

      // Create input tensor with optimal memory usage
      final inputShape = [1, channels, width, height];
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        inputData,
        inputShape,
      );

      // Run model inference with error handling
      final outputs = await Future(() => _session!.run(
            OrtRunOptions(),
            {'input': inputTensor},
          ));

      // Validate outputs
      if (outputs.length < 2) {
        print("Invalid model output");
        return const [];
      }
      inputTensor.release();
      final simccX = outputs[0] as OrtValueTensor;
      final simccY = outputs[1] as OrtValueTensor;

      return _postProcess(simccX, simccY, image.width, image.height);
    } catch (e, stackTrace) {
      print("Error during inference: $e");
      print("Stack trace: $stackTrace");
      return const [];
    } finally {
      // Clean up resources
      // inputTensor?.release();
    }
  }

  // Helper method to convert CameraImage to Image format
  Future<img.Image?> convertCameraImageToImage(CameraImage cameraImage) async {
    try {
      // Handle BGRA8888 format
      final int width = cameraImage.width;
      final int height = cameraImage.height;

      // Create image buffer
      final img.Image image = img.Image(width: width, height: height);

      // Get the raw bytes from first plane
      final Uint8List bytes = cameraImage.planes[0].bytes;

      // Convert BGRA to RGBA
      int inputIndex = 0;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int b = bytes[inputIndex];
          final int g = bytes[inputIndex + 1];
          final int r = bytes[inputIndex + 2];
          final int a = bytes[inputIndex + 3];

          // Set pixel in the image
          image.setPixelRgba(x, y, r, g, b, a);

          inputIndex += 4; // Move to next pixel (4 bytes per pixel)
        }
      }

      return image;
    } catch (e) {
      print("Error converting camera image: $e");
      return null;
    }
  }

  convertPoints(List<Offset> outputs, int widthNormal, int heightNormal) {
    final List<Offset> newPonits = [];

    for (var i = 0; i < outputs.length; i++) {
      final x = outputs[i].dx;
      final y = outputs[i].dy;
      if (x > 0 && y > 0) {
        final newX = x * widthNormal / width;
        final newY = y * heightNormal / height;

        newPonits.add(Offset(newX, newY));
      }
    }

    return newPonits;
  }

  List<Offset> _postProcess(OrtValueTensor simccX, OrtValueTensor simccY,
      int widthNormal, int heightNormal) {
    final keypoints = <Offset>[];
    final simccXPoints = simccX.value as List<List<List<double>>>;
    final simccYPoints = simccY.value as List<List<List<double>>>;
    final simccx = simccXPoints[0];
    final simccy = simccYPoints[0];

    for (var [itemX, itemY] in zip(simccx, simccy)) {
      final indexX = itemX.argMax();
      final indexY = itemY.argMax();

      final x = indexX / itemX.length * width;
      final y = indexY / itemY.length * height;

      final xConf = itemX[indexX];
      final yConf = itemY[indexY];

      if (xConf > 0.3 && yConf > 0.3) {
        keypoints.add(Offset(x, y));
      } else {
        keypoints.add(Offset.zero);
      }
    }
    return convertPoints(keypoints, widthNormal, heightNormal);
    // return convertPoints(keypoints, widthNormal, heightNormal);
  }

  Float32List imageToFloat32List(img.Image image) {
    int pixelCount = width * height;

    Float32List inputData = Float32List(3 * pixelCount);

    // Các giá trị chuẩn hóa - điều chỉnh theo mô hình của bạn nếu cần
    final List<double> mean = [0.485, 0.456, 0.406];
    final List<double> std = [0.229, 0.224, 0.225];

    final List<int> rgbBytes = image.getBytes();

    for (int i = 0; i < pixelCount; i++) {
      final int r = rgbBytes[i * 3];
      final int g = rgbBytes[i * 3 + 1];
      final int b = rgbBytes[i * 3 + 2];

      // Chuẩn hóa và chuyển sang từng kênh màu
      inputData[i] = ((r / 255.0) - mean[0]) / std[0]; // R channel
      inputData[pixelCount + i] = ((g / 255.0) - mean[1]) / std[1]; // G channel
      inputData[2 * pixelCount + i] =
          ((b / 255.0) - mean[2]) / std[2]; // B channel
    }

    return inputData;
  }

  List<List<T>> zip<T>(List<T> a, List<T> b) {
    int length = a.length < b.length ? a.length : b.length;
    List<List<T>> result = [];
    for (int i = 0; i < length; i++) {
      result.add([a[i], b[i]]);
    }
    return result;
  }
}

extension ArgMaxExtension on List<num> {
  int argMax() {
    if (isEmpty) {
      throw StateError('Cannot find argMax of an empty list.');
    }

    int maxIndex = 0;
    num maxValue = this[0];

    for (int i = 1; i < length; i++) {
      if (this[i] > maxValue) {
        maxValue = this[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }
}

import 'package:ai_detection/pose_detector_rope_skipping.dart';
import 'package:ai_detection/pose_painter.dart';
import 'package:camera/camera.dart';
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
  final RopeSkippingPoseDetector _jumpDetector = RopeSkippingPoseDetector();
  final GlobalKey _cameraViewKey = GlobalKey();

  // Rope skipping state variables
  int jumpCount = 0;
  String previousState = "on_ground";
  double previousY = 0.0;
  double jumpThreshold = 30.0;

  Size screenSize = Size.zero;

  @override
  void dispose() async {
    _canProcess = false;
    _poseDetector.close();
    super.dispose();
  }

  Size? _cameraPreviewSize;

  @override
  void initState() {
    super.initState();
    // We'll initialize the frame boundary when the layout is built
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Get the screen size to set the frame boundaries
      final size = MediaQuery.of(context).size;
      // Initially set with screen size, will be updated when camera is ready
      setState(() {
        screenSize = getWidgetSize() ?? size;
        // Set initial frame boundary
      });
    });
  }

  Size? getWidgetSize() {
    if (_cameraViewKey.currentContext == null) return null;

    final RenderBox renderBox =
        _cameraViewKey.currentContext!.findRenderObject() as RenderBox;
    return renderBox.size;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          DetectorView(
            key: _cameraViewKey,
            title: 'Pose Detector',
            customPaint: _customPaint,
            text: _text,
            onImage: _processImage,
            initialCameraLensDirection: _cameraLensDirection,
            onCameraLensDirectionChanged: (value) =>
                _cameraLensDirection = value,
          ),
          if (_cameraPreviewSize != null)
            Positioned(
              top: 100,
              left: 10,
              child: Container(
                padding: const EdgeInsets.all(5),
                color: Colors.black45,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Preview: ${_cameraPreviewSize!.width.toStringAsFixed(0)}x${_cameraPreviewSize!.height.toStringAsFixed(0)}',
                      style: const TextStyle(color: Colors.white),
                    ),
                    TextButton(
                      onPressed: () {
                        if (_cameraPreviewSize != null) {
                          // _jumpDetector.setFrameBoundary(
                          //   getWidgetSize()!,
                          // );
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text('Frame boundary recalibrated')),
                          );
                        }
                      },
                      child: const Text('Recalibrate Frame',
                          style: TextStyle(color: Colors.lightBlue)),
                    ),
                  ],
                ),
              ),
            ),
          // Add a button to toggle the frame
          Positioned(
            bottom: 190,
            right: 20,
            child: FloatingActionButton(
              heroTag: 'toggle_frame',
              onPressed: () {
                // setState(() {
                //   _jumpDetector.toggleFrameActivation();
                // });
              },
              backgroundColor: Colors.blue.withOpacity(0.7),
              child: const Icon(
                Icons.crop_din,
                color: Colors.white,
              ),
            ),
          ),
          // Add a button to adjust frame size (smaller)
          Positioned(
            bottom: 120,
            right: 20,
            child: FloatingActionButton(
              heroTag: 'adjust_frame',
              onPressed: () {
                if (_cameraPreviewSize != null) {
                  // setState(() {
                  //   // Adjust frame size (make it smaller)
                  //   _jumpDetector.setFrameBoundary(
                  //     _cameraPreviewSize!,
                  //   );
                  // });
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                        content: Text('Frame size adjusted to smaller')),
                  );
                }
              },
              backgroundColor: Colors.green.withOpacity(0.7),
              child: const Icon(
                Icons.fit_screen,
                color: Colors.white,
              ),
            ),
          ),
          // Add a button to reset jump count
          Positioned(
            bottom: 50,
            right: 20,
            child: FloatingActionButton(
              heroTag: 'reset_count',
              onPressed: () {
                // setState(() {
                //   _jumpDetector.resetJumpCount();
                //   _text = 'Jump Count: 0';
                // });
              },
              backgroundColor: Colors.red.withOpacity(0.7),
              child: const Icon(
                Icons.refresh,
                color: Colors.white,
              ),
            ),
          ),
          // Display jump count prominently
          if (_text != null && _text!.isNotEmpty)
            Positioned(
              top: 50,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black45,
                child: Text(
                  _text!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
        ],
      ),
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
    if (poses.isNotEmpty &&
        inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      // Use the camera preview size if available, or fall back to image size

      // // If the frame boundary is still at default (Rect.zero), set it now
      // if (_jumpDetector.frameBoundary == Rect.zero) {
      //   _jumpDetector.setFrameBoundary(screenSize);
      // }

      // Pass screen size, rotation, and camera direction to check if person is in frame
      _jumpDetector.detectPose(poses.first, screenSize: screenSize);

      // Update the UI with jump count and person-in-frame status
      // final inFrame = _jumpDetector.isPersonInFrame(
      //     poses.first,
      //     screenSize,
      //     inputImage.metadata!.size,
      //     inputImage.metadata!.rotation,
      //     _cameraLensDirection);

      // print("Jump Count: $jumpCount, In frame: $inFrame");
    }
    if (inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      final painter = PosePainter(
        poses,
        inputImage.metadata!.size,
        inputImage.metadata!.rotation,
        _cameraLensDirection,
        // Pass jump detector to draw frame
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
}

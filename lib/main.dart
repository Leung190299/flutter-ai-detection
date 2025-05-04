import 'dart:ui' as ui;

import 'package:ai_detection/onnx.dart';
import 'package:ai_detection/pose_detector.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

late List<CameraDescription> _cameras;
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const PoseDetectorView(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late OnnxModel _onnxModel;
  List<Offset> _outputs = [];
  ui.Image? backgroundImage;
  late CameraController controller;

  @override
  void initState() {
    _onnxModel = OnnxModel();
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      if (_cameras.isEmpty) {
        return;
      }

      controller = CameraController(
        _cameras[1],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.bgra8888,
        fps: 5,
      );
      await controller.initialize().then((value) {
        controller.startImageStream(_processCameraImage);
      });
    } on CameraException catch (e) {
      print('Camera error: ${e.code}, ${e.description}');
    } catch (e) {
      print('Error: $e');
    }
  }

  _processCameraImage(CameraImage image) async {
    final outputs = await _onnxModel.runInference(image);
    if (outputs.isNotEmpty) {
      setState(() {
        _outputs = outputs;
      });
    }
  }

  Future<ui.Image> getBackgroundImage(String path) async {
    final ByteData data = await rootBundle.load(path);
    final Uint8List bytes = data.buffer.asUint8List();
    final codec = await ui.instantiateImageCodec(bytes);
    final frameInfo = await codec.getNextFrame();
    return frameInfo.image;
  }

  // Future<void> _runModel() async {
  //   const String assetPath = 'assets/images/output_onnxruntime.jpg';
  //   final List<Offset> outputs = await _onnxModel.runInference(assetPath);
  //   final background = await getBackgroundImage(assetPath);

  //   if (outputs.isEmpty) {
  //     throw Exception('No outputs from the model');
  //   }
  //   // Process the outputs as needed
  //   if (outputs.isEmpty) {
  //     throw Exception('No outputs from the model');
  //   }
  //   setState(() {
  //     _outputs = outputs;
  //     backgroundImage = background;
  //   });
  // }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      body: Stack(
        children: [
          OverflowBox(
            maxHeight: size.height,
            maxWidth: size.width,
            child: CameraPreview(controller),
          ),
          if (_outputs.isNotEmpty)
            OverflowBox(
              maxHeight: size.height,
              maxWidth: size.width,
              child: CustomPaint(
                size: Size(controller.value.previewSize?.width ?? size.width,
                    controller.value.previewSize?.height ?? size.height),
                painter: SkeletonPainter(
                  keypoints: _outputs,
                  background: backgroundImage,
                ),
              ),
            )
        ],
      ),

      // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}

class SkeletonPainter extends CustomPainter {
  final List<Offset> keypoints;
  final ui.Image? background;

  SkeletonPainter({required this.keypoints, this.background});

  Map<String, List<List<int>>> keypointNames = {
    'face': [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 4]
    ],
    'upper_body': [
      [0, 5],
      [0, 6],
      [5, 6]
    ],
    'lower_body': [
      [5, 11],
      [6, 12],
      [11, 12]
    ],
    'left_limbs': [
      [5, 7],
      [7, 9],
      [11, 13],
      [13, 15]
    ],
    'right_limbs': [
      [6, 8],
      [8, 10],
      [12, 14],
      [14, 16]
    ]
  };

  @override
  void paint(Canvas canvas, Size size) {
    if (background != null) {
      paintImage(
          canvas: canvas,
          rect: Rect.fromLTWH(0, 0, size.width, size.height),
          image: background!);
    }

    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 2;

    _drawSkeleton(canvas, paint, size);
    _drawKeypoints(canvas, paint);
  }

  void _drawSkeleton(Canvas canvas, Paint paint, Size size) {
    // Draw lines first
    for (var connections in keypointNames.values) {
      for (final pair in connections) {
        final startIndex = pair[0];
        final endIndex = pair[1];

        if (_areValidIndices(startIndex, endIndex)) {
          final pt1 = keypoints[startIndex];
          final pt2 = keypoints[endIndex];
          if (pt1.dx != 0 && pt1.dy != 0 && pt2.dx != 0 && pt2.dy != 0) {
            _drawBodyLine(canvas, paint, startIndex, endIndex);
          }
        }
      }
    }

    // Define keypoint names
    final List<String> pointNames = [
      'mũi',
      'mắt_trái',
      'mắt_phải',
      'tai_trái',
      'tai_phải',
      'vai_trái',
      'vai_phải',
      'khuỷu_tay_trái',
      'khuỷu_tay_phải',
      'cổ_tay_trái',
      'cổ_tay_phải',
      'hông_trái',
      'hông_phải',
      'đầu_gối_trái',
      'đầu_gối_phải',
      'mắt_cá_chân_trái',
      'mắt_cá_chân_phải'
    ];

    // Draw text for each keypoint

    final textStyle = TextStyle(
      color: Colors.white,
      fontSize: 12,
      background: Paint()..color = Colors.black.withOpacity(0.5),
    );

    for (var i = 0; i < keypoints.length; i++) {
      if (i < pointNames.length) {
        final textSpan = TextSpan(
          text: pointNames[i],
          style: textStyle,
        );
        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
        textPainter.paint(
          canvas,
          Offset(keypoints[i].dx + 10, keypoints[i].dy - 10),
        );
      }
    }
  }

  bool _areValidIndices(int start, int end) {
    return start < keypoints.length && end < keypoints.length;
  }

  void _drawBodyLine(Canvas canvas, Paint paint, int startIndex, int endIndex) {
    canvas.drawLine(
      keypoints[startIndex],
      keypoints[endIndex],
      paint,
    );
  }

  void _drawKeypoints(Canvas canvas, Paint paint) {
    for (var point in keypoints) {
      canvas.drawCircle(point, 5, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

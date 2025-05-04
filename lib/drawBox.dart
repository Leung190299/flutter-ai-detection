import 'package:flutter/widgets.dart';

class BndBox extends StatelessWidget {
  final List<Offset> results;

  const BndBox(this.results, {super.key});

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;

    List<Widget> renderKeypoints() {
      return results.map((re) {
        return Positioned(
          left: re.dx,
          top: re.dy,
          width: 100,
          height: 12,
          child: Container(
            child: Text(
              "● ${["part"]}",
              style: const TextStyle(
                color: Color.fromRGBO(37, 213, 253, 1.0),
                fontSize: 12.0,
              ),
            ),
          ),
        );
      }).toList();
    }

    return SizedBox(
      width: size.width,
      height: size.height,
      child: Stack(
        children: renderKeypoints(),
      ),
    );
  }
}

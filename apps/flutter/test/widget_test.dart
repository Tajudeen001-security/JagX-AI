import 'package:flutter_test/flutter_test.dart';
import 'package:jagx_ai/main.dart';

void main() {
  testWidgets('JagX home renders core workspace', (tester) async {
    await tester.pumpWidget(const JagXApp());
    expect(find.text('JagX AI'), findsOneWidget);
    expect(find.text('Your unified AI workspace.'), findsOneWidget);
    expect(find.text('Code'), findsOneWidget);
  });
}

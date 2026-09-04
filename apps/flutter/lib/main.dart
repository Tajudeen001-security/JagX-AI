import 'package:flutter/material.dart';

void main() => runApp(const JagXApp());

class JagXApp extends StatelessWidget {
  const JagXApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'JagX AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF09090B),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF7C5CFF),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const JagXHome(),
    );
  }
}

class JagXHome extends StatelessWidget {
  const JagXHome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('JagX AI')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: const [
          Text('Your AI workspace', style: TextStyle(fontSize: 30, fontWeight: FontWeight.w800)),
          SizedBox(height: 8),
          Text('Chat, create, research, code and manage projects from one app.'),
          SizedBox(height: 24),
          _FeatureCard('Chat', 'Reasoning, conversations and multimodal input.'),
          _FeatureCard('Create', 'Images, audio, video and documents.'),
          _FeatureCard('JagX Code', 'Projects, files, AI assistance and safe execution.'),
          _FeatureCard('Research', 'Structured research with sources and reports.'),
          _FeatureCard('Memory', 'Control saved knowledge and project context.'),
        ],
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final String title;
  final String description;
  const _FeatureCard(this.title, this.description);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(description, style: TextStyle(color: Colors.grey.shade400)),
        ]),
      ),
    );
  }
}

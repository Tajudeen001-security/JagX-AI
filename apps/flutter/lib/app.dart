import 'package:flutter/material.dart';
import 'theme.dart';

class JagXApp extends StatelessWidget {
  const JagXApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'JagX AI', debugShowCheckedModeBanner: false,
    theme: JagXTheme.dark(), home: const JagXHome(),
  );
}

class JagXHome extends StatefulWidget {
  const JagXHome({super.key});
  @override State<JagXHome> createState() => _JagXHomeState();
}

class _JagXHomeState extends State<JagXHome> {
  int index = 0;
  static const items = [
    ('Home', Icons.home_outlined), ('Chat', Icons.chat_bubble_outline),
    ('Create', Icons.auto_awesome_outlined), ('Code', Icons.code),
    ('Research', Icons.travel_explore), ('Memory', Icons.psychology_outlined),
    ('Projects', Icons.folder_outlined),
  ];

  @override Widget build(BuildContext context) {
    final selected = items[index].$1;
    return Scaffold(
      appBar: AppBar(title: const Text('JagX AI', style: TextStyle(fontWeight: FontWeight.w800)), actions: [
        IconButton(onPressed: () {}, icon: const Icon(Icons.settings_outlined)),
      ]),
      body: LayoutBuilder(builder: (context, constraints) {
        final wide = constraints.maxWidth >= 800;
        final content = _Workspace(selected: selected);
        if (!wide) return content;
        return Row(children: [
          NavigationRail(selectedIndex: index, onDestinationSelected: (v) => setState(() => index = v), labelType: NavigationRailLabelType.all,
            destinations: [for (final item in items) NavigationRailDestination(icon: Icon(item.$2), label: Text(item.$1))]),
          const VerticalDivider(width: 1), Expanded(child: content),
        ]);
      }),
      bottomNavigationBar: MediaQuery.sizeOf(context).width < 800 ? NavigationBar(selectedIndex: index, onDestinationSelected: (v) => setState(() => index = v), destinations: [
        for (final item in items.take(5)) NavigationDestination(icon: Icon(item.$2), label: item.$1),
      ]) : null,
    );
  }
}

class _Workspace extends StatelessWidget {
  final String selected;
  const _Workspace({required this.selected});
  @override Widget build(BuildContext context) => ListView(padding: const EdgeInsets.fromLTRB(20, 12, 20, 32), children: [
    Text(selected, style: const TextStyle(fontSize: 32, fontWeight: FontWeight.w800)),
    const SizedBox(height: 8),
    Text(_subtitle(selected), style: const TextStyle(color: JagXTheme.muted, fontSize: 16)),
    const SizedBox(height: 24),
    if (selected == 'Home') ...[
      const _PromptBox(), const SizedBox(height: 24),
      Wrap(spacing: 12, runSpacing: 12, children: const [
        _Card('Chat', 'Reason, ask questions and work with files.', Icons.chat_bubble_outline),
        _Card('Create', 'Generate and transform content.', Icons.auto_awesome_outlined),
        _Card('Code', 'Build software with JagX Code.', Icons.code),
        _Card('Research', 'Explore information and produce reports.', Icons.travel_explore),
      ]),
    ] else const _ComingSoon(),
  ]);

  static String _subtitle(String value) => switch (value) {
    'Chat' => 'A focused workspace for conversations and multimodal work.',
    'Create' => 'Create documents, media and other content.',
    'Code' => 'Build, inspect and run projects safely.',
    'Research' => 'Organize research into evidence-backed work.',
    'Memory' => 'Manage persistent knowledge and context.',
    'Projects' => 'Keep related conversations, files and tasks together.',
    _ => 'Your unified AI workspace.',
  };
}

class _PromptBox extends StatelessWidget {
  const _PromptBox();
  @override Widget build(BuildContext context) => TextField(maxLines: 4, decoration: const InputDecoration(hintText: 'Ask JagX anything…', suffixIcon: Icon(Icons.arrow_upward)));
}
class _Card extends StatelessWidget {
  final String title, description; final IconData icon;
  const _Card(this.title, this.description, this.icon);
  @override Widget build(BuildContext context) => SizedBox(width: 280, child: Card(child: Padding(padding: const EdgeInsets.all(18), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Icon(icon, size: 26), const SizedBox(height: 16), Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)), const SizedBox(height: 6), Text(description, style: const TextStyle(color: JagXTheme.muted)),
  ]))));
}
class _ComingSoon extends StatelessWidget { const _ComingSoon(); @override Widget build(BuildContext context) => const Card(child: Padding(padding: EdgeInsets.all(20), child: Text('Workspace ready. Feature modules are being connected to the JagX runtime.', style: TextStyle(color: JagXTheme.muted)))); }

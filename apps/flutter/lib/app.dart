import 'package:flutter/material.dart';
import 'theme.dart';

class JagXApp extends StatelessWidget {
  const JagXApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'JagX AI',
        debugShowCheckedModeBanner: false,
        theme: JagXTheme.dark(),
        home: const JagXHome(),
      );
}

class JagXHome extends StatefulWidget {
  const JagXHome({super.key});

  @override
  State<JagXHome> createState() => _JagXHomeState();
}

class _JagXHomeState extends State<JagXHome> {
  int index = 0;
  final prompt = TextEditingController();
  final messages = <String>[];

  static const items = [
    ('Home', Icons.home_outlined),
    ('Chat', Icons.chat_bubble_outline),
    ('Create', Icons.auto_awesome_outlined),
    ('Code', Icons.code),
    ('Research', Icons.travel_explore),
    ('Memory', Icons.psychology_outlined),
    ('Projects', Icons.folder_outlined),
  ];

  @override
  void dispose() {
    prompt.dispose();
    super.dispose();
  }

  void select(int value) => setState(() => index = value);

  void sendPrompt() {
    final value = prompt.text.trim();
    if (value.isEmpty) return;
    setState(() {
      messages.add(value);
      prompt.clear();
      index = 1;
    });
  }

  @override
  Widget build(BuildContext context) {
    final selected = items[index].$1;
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 20,
        title: Row(
          children: [
            Container(
              width: 34,
              height: 34,
              decoration: BoxDecoration(
                color: JagXTheme.accent.withValues(alpha: .16),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.auto_awesome, size: 18),
            ),
            const SizedBox(width: 10),
            const Text('JagX AI', style: TextStyle(fontWeight: FontWeight.w800)),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Settings',
            onPressed: () => _showSettings(context),
            icon: const Icon(Icons.settings_outlined),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final wide = constraints.maxWidth >= 900;
          final workspace = _Workspace(
            selected: selected,
            messages: messages,
            controller: prompt,
            onSend: sendPrompt,
            onNavigate: select,
          );
          if (!wide) return workspace;
          return Row(
            children: [
              NavigationRail(
                selectedIndex: index,
                onDestinationSelected: select,
                labelType: NavigationRailLabelType.all,
                destinations: [
                  for (final item in items)
                    NavigationRailDestination(
                      icon: Icon(item.$2),
                      selectedIcon: Icon(item.$2),
                      label: Text(item.$1),
                    ),
                ],
              ),
              const VerticalDivider(width: 1),
              Expanded(child: workspace),
            ],
          );
        },
      ),
      bottomNavigationBar: MediaQuery.sizeOf(context).width < 900
          ? NavigationBar(
              selectedIndex: index > 4 ? 0 : index,
              onDestinationSelected: select,
              destinations: [
                for (final item in items.take(5))
                  NavigationDestination(icon: Icon(item.$2), label: item.$1),
              ],
            )
          : null,
    );
  }

  void _showSettings(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('JagX settings', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
              const SizedBox(height: 6),
              const Text('Workspace preferences and runtime controls.', style: TextStyle(color: JagXTheme.muted)),
              const SizedBox(height: 18),
              ListTile(leading: const Icon(Icons.palette_outlined), title: const Text('Appearance'), trailing: const Text('Dark')),
              ListTile(leading: const Icon(Icons.security_outlined), title: const Text('Safety'), trailing: const Text('Protected')),
              ListTile(leading: const Icon(Icons.language_outlined), title: const Text('Web'), trailing: const Text('jagxai.name.ng')),
            ],
          ),
        ),
      ),
    );
  }
}

class _Workspace extends StatelessWidget {
  final String selected;
  final List<String> messages;
  final TextEditingController controller;
  final VoidCallback onSend;
  final ValueChanged<int> onNavigate;

  const _Workspace({
    required this.selected,
    required this.messages,
    required this.controller,
    required this.onSend,
    required this.onNavigate,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 1180),
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
          children: [
            if (selected == 'Home') _home(context) else _module(context),
          ],
        ),
      ),
    );
  }

  Widget _home(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 28),
          const Text('Your unified AI workspace.', style: TextStyle(fontSize: 38, fontWeight: FontWeight.w900, letterSpacing: -.8)),
          const SizedBox(height: 10),
          const Text('Think, create, code, research and organize your work with JagX.', style: TextStyle(color: JagXTheme.muted, fontSize: 17)),
          const SizedBox(height: 28),
          _PromptComposer(controller: controller, onSend: onSend),
          const SizedBox(height: 24),
          const _SectionLabel('Explore'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _FeatureCard('Chat', 'Ask questions, reason through problems and work with context.', Icons.chat_bubble_outline, () => onNavigate(1)),
              _FeatureCard('Create', 'Write, transform and prepare documents and media workflows.', Icons.auto_awesome_outlined, () => onNavigate(2)),
              _FeatureCard('Code', 'Build software with an AI coding workspace.', Icons.code, () => onNavigate(3)),
              _FeatureCard('Research', 'Collect sources and turn findings into structured work.', Icons.travel_explore, () => onNavigate(4)),
              _FeatureCard('Memory', 'Keep durable knowledge and useful context organized.', Icons.psychology_outlined, () => onNavigate(5)),
              _FeatureCard('Projects', 'Group conversations, files, code and research in one place.', Icons.folder_outlined, () => onNavigate(6)),
            ],
          ),
          const SizedBox(height: 28),
          const _SectionLabel('Capabilities'),
          const SizedBox(height: 12),
          const _CapabilityStrip(),
        ],
      );

  Widget _module(BuildContext context) {
    final data = _moduleData(selected);
    if (selected == 'Chat') {
      return _ChatView(messages: messages, controller: controller, onSend: onSend);
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 12),
        Row(children: [Icon(data.$3, size: 28), const SizedBox(width: 12), Text(selected, style: const TextStyle(fontSize: 34, fontWeight: FontWeight.w900))]),
        const SizedBox(height: 8),
        Text(data.$2, style: const TextStyle(color: JagXTheme.muted, fontSize: 16)),
        const SizedBox(height: 26),
        _ModuleHero(title: data.$1, icon: data.$3, onPrimary: () {}),
        const SizedBox(height: 18),
        Wrap(spacing: 12, runSpacing: 12, children: [
          _InfoTile(data.$4, data.$5),
          _InfoTile('Status', 'UI layer connected; runtime integration follows.'),
          _InfoTile('Workspace', 'Ready for real project data and model endpoints.'),
        ]),
      ],
    );
  }

  static (String, String, IconData, String, String) _moduleData(String value) => switch (value) {
        'Create' => ('Creative studio', 'Generate documents and media from one workspace.', Icons.auto_awesome, 'Documents', 'Draft reports, stories and structured files.'),
        'Code' => ('JagX Code', 'A focused IDE workspace for building and reviewing software.', Icons.code, 'Projects', 'Work with source files, tasks and safe execution.'),
        'Research' => ('Research desk', 'Turn questions into organized, evidence-backed research.', Icons.travel_explore, 'Sources', 'Collect, compare and summarize source material.'),
        'Memory' => ('Memory vault', 'Organize durable knowledge and reusable context.', Icons.psychology_outlined, 'Knowledge', 'Prepare memories, notes and provenance for retrieval.'),
        _ => ('Project hub', 'Keep related conversations, files and work together.', Icons.folder_outlined, 'Projects', 'Create project spaces for long-running work.'),
      };
}

class _PromptComposer extends StatelessWidget {
  final TextEditingController controller;
  final VoidCallback onSend;
  const _PromptComposer({required this.controller, required this.onSend});

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            children: [
              TextField(
                controller: controller,
                minLines: 2,
                maxLines: 6,
                onSubmitted: (_) => onSend(),
                decoration: const InputDecoration(hintText: 'Ask JagX anything…', border: InputBorder.none, filled: false),
              ),
              Row(
                children: [
                  IconButton(tooltip: 'Attach', onPressed: () {}, icon: const Icon(Icons.attach_file_outlined)),
                  IconButton(tooltip: 'Web research', onPressed: () {}, icon: const Icon(Icons.language_outlined)),
                  const Spacer(),
                  FilledButton.icon(onPressed: onSend, icon: const Icon(Icons.arrow_upward, size: 18), label: const Text('Send')),
                ],
              ),
            ],
          ),
        ),
      );
}

class _ChatView extends StatelessWidget {
  final List<String> messages;
  final TextEditingController controller;
  final VoidCallback onSend;
  const _ChatView({required this.messages, required this.controller, required this.onSend});

  @override
  Widget build(BuildContext context) => Column(
        children: [
          const SizedBox(height: 12),
          Row(children: [const Text('Chat', style: TextStyle(fontSize: 34, fontWeight: FontWeight.w900)), const Spacer(), OutlinedButton.icon(onPressed: () {}, icon: const Icon(Icons.add), label: const Text('New chat'))]),
          const SizedBox(height: 18),
          if (messages.isEmpty)
            const Padding(padding: EdgeInsets.symmetric(vertical: 90), child: Column(children: [Icon(Icons.chat_bubble_outline, size: 46), SizedBox(height: 14), Text('Start a conversation', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800)), SizedBox(height: 6), Text('Your messages will appear here.', style: TextStyle(color: JagXTheme.muted))]))
          else
            ...messages.map((message) => Align(alignment: Alignment.centerRight, child: Container(margin: const EdgeInsets.only(bottom: 12), padding: const EdgeInsets.all(14), constraints: const BoxConstraints(maxWidth: 720), decoration: BoxDecoration(color: JagXTheme.surfaceElevated, borderRadius: BorderRadius.circular(16)), child: Text(message)))),
          const SizedBox(height: 16),
          _PromptComposer(controller: controller, onSend: onSend),
        ],
      );
}

class _FeatureCard extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final VoidCallback onTap;
  const _FeatureCard(this.title, this.description, this.icon, this.onTap);

  @override
  Widget build(BuildContext context) => SizedBox(
        width: 360,
        child: Card(
          child: InkWell(
            onTap: onTap,
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(icon, size: 26),
                const SizedBox(height: 18),
                Text(title, style: const TextStyle(fontSize: 19, fontWeight: FontWeight.w800)),
                const SizedBox(height: 7),
                Text(description, style: const TextStyle(color: JagXTheme.muted, height: 1.4)),
                const SizedBox(height: 14),
                const Row(children: [Text('Open workspace', style: TextStyle(fontWeight: FontWeight.w700)), SizedBox(width: 5), Icon(Icons.arrow_forward, size: 16)]),
              ]),
            ),
          ),
        ),
      );
}

class _SectionLabel extends StatelessWidget {
  final String value;
  const _SectionLabel(this.value);
  @override
  Widget build(BuildContext context) => Text(value.toUpperCase(), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w800, letterSpacing: 1.3, color: JagXTheme.muted));
}

class _CapabilityStrip extends StatelessWidget {
  const _CapabilityStrip();
  @override
  Widget build(BuildContext context) => Wrap(spacing: 10, runSpacing: 10, children: const [
        Chip(avatar: Icon(Icons.image_outlined, size: 18), label: Text('Multimodal')),
        Chip(avatar: Icon(Icons.terminal_outlined, size: 18), label: Text('Tools')),
        Chip(avatar: Icon(Icons.memory_outlined, size: 18), label: Text('Memory')),
        Chip(avatar: Icon(Icons.security_outlined, size: 18), label: Text('Safety')),
        Chip(avatar: Icon(Icons.cloud_outlined, size: 18), label: Text('Web')),
        Chip(avatar: Icon(Icons.phone_android_outlined, size: 18), label: Text('Mobile + Web')),
      ]);
}

class _ModuleHero extends StatelessWidget {
  final String title;
  final IconData icon;
  final VoidCallback onPrimary;
  const _ModuleHero({required this.title, required this.icon, required this.onPrimary});
  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(22),
          child: Row(children: [
            Container(width: 54, height: 54, decoration: BoxDecoration(color: JagXTheme.accent.withValues(alpha: .15), borderRadius: BorderRadius.circular(16)), child: Icon(icon)),
            const SizedBox(width: 16),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800)), const SizedBox(height: 4), const Text('The interface is ready for its connected runtime.', style: TextStyle(color: JagXTheme.muted))])),
            FilledButton(onPressed: onPrimary, child: const Text('Get started')),
          ]),
        ),
      );
}

class _InfoTile extends StatelessWidget {
  final String title;
  final String value;
  const _InfoTile(this.title, this.value);
  @override
  Widget build(BuildContext context) => SizedBox(width: 360, child: Card(child: Padding(padding: const EdgeInsets.all(18), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontWeight: FontWeight.w800)), const SizedBox(height: 7), Text(value, style: const TextStyle(color: JagXTheme.muted, height: 1.35))])));
}

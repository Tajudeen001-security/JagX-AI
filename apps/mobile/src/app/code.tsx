import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

const files = ['src/App.tsx', 'src/api/client.ts', 'src/components/Chat.tsx', 'tests/App.test.tsx', 'package.json'];

export default function Code() {
  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}><Text style={styles.title}>JagX Code</Text><Pressable style={styles.run}><Text style={styles.runText}>Run</Text></Pressable></View>
      <View style={styles.workspace}>
        <View style={styles.explorer}><Text style={styles.label}>EXPLORER</Text>{files.map(file => <Text key={file} style={styles.file}>{file}</Text>)}</View>
        <ScrollView style={styles.editor}><Text style={styles.tab}>App.tsx</Text><Text style={styles.code}>{`export default function App() {\n  return (\n    <JagXWorkspace />\n  );\n}`}</Text></ScrollView>
      </View>
      <View style={styles.assistant}><Text style={styles.assistantTitle}>JagX Assistant</Text><Text style={styles.muted}>Explain, generate, refactor, debug and review this project.</Text></View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#09090B' },
  header: { padding: 18, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderBottomWidth: 1, borderColor: '#27272A' },
  title: { color: '#FAFAFA', fontSize: 24, fontWeight: '800' },
  run: { backgroundColor: '#F4F4F5', borderRadius: 10, paddingHorizontal: 16, paddingVertical: 9 },
  runText: { color: '#09090B', fontWeight: '700' },
  workspace: { flex: 1, flexDirection: 'row' },
  explorer: { width: 150, padding: 14, borderRightWidth: 1, borderColor: '#27272A' },
  label: { color: '#71717A', fontSize: 10, fontWeight: '800', marginBottom: 12 },
  file: { color: '#D4D4D8', fontSize: 12, paddingVertical: 7 },
  editor: { flex: 1, padding: 16 },
  tab: { color: '#FAFAFA', backgroundColor: '#18181B', padding: 10, borderRadius: 8, alignSelf: 'flex-start' },
  code: { color: '#D4D4D8', fontFamily: 'monospace', lineHeight: 22, marginTop: 18 },
  assistant: { borderTopWidth: 1, borderColor: '#27272A', padding: 16, backgroundColor: '#111113' },
  assistantTitle: { color: '#FAFAFA', fontWeight: '700', marginBottom: 5 },
  muted: { color: '#A1A1AA', lineHeight: 20 },
});

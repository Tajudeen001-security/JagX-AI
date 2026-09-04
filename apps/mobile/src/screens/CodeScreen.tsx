import React, { useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { theme } from '../theme';

const files = ['app.tsx', 'api.ts', 'model.py', 'README.md'];
const initialCode = `export async function runTask(input: string) {\n  const result = await jagx.execute({ input });\n  return result;\n}`;

export default function CodeScreen() {
  const [code, setCode] = useState(initialCode);
  const [activeFile, setActiveFile] = useState(files[0]);
  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}><View><Text style={styles.kicker}>JAGX CODE</Text><Text style={styles.title}>Workspace</Text></View><Pressable style={styles.run}><Text style={styles.runText}>Run</Text></Pressable></View>
      <View style={styles.workspace}>
        <View style={styles.explorer}><Text style={styles.section}>EXPLORER</Text>{files.map(file => <Pressable key={file} onPress={() => setActiveFile(file)} style={[styles.file, activeFile === file && styles.activeFile]}><Text style={styles.fileText}>{file}</Text></Pressable>)}</View>
        <View style={styles.editor}>
          <View style={styles.tab}><Text style={styles.tabText}>{activeFile}</Text></View>
          <TextInput value={code} onChangeText={setCode} multiline spellCheck={false} style={styles.code} textAlignVertical="top" />
        </View>
      </View>
      <View style={styles.bottom}><Text style={styles.bottomTitle}>JAGX ASSISTANT</Text><ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.actions}><Pressable style={styles.action}><Text style={styles.actionText}>Explain</Text></Pressable><Pressable style={styles.action}><Text style={styles.actionText}>Fix bug</Text></Pressable><Pressable style={styles.action}><Text style={styles.actionText}>Generate tests</Text></Pressable><Pressable style={styles.action}><Text style={styles.actionText}>Refactor</Text></Pressable></ScrollView></View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.background },
  header: { padding: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderBottomWidth: 1, borderBottomColor: theme.border },
  kicker: { color: theme.accent, fontSize: 10, fontWeight: '800', letterSpacing: 2 }, title: { color: theme.text, fontSize: 19, fontWeight: '700', marginTop: 3 },
  run: { backgroundColor: theme.accent, borderRadius: 12, paddingHorizontal: 17, paddingVertical: 9 }, runText: { color: theme.text, fontWeight: '800' },
  workspace: { flex: 1, flexDirection: 'row' }, explorer: { width: 118, borderRightWidth: 1, borderRightColor: theme.border, padding: 10 }, section: { color: theme.muted, fontSize: 9, fontWeight: '800', letterSpacing: 1.4, marginBottom: 10 }, file: { paddingVertical: 9, paddingHorizontal: 7, borderRadius: 7 }, activeFile: { backgroundColor: theme.surfaceRaised }, fileText: { color: theme.muted, fontSize: 12 },
  editor: { flex: 1, backgroundColor: '#0D0D0F' }, tab: { height: 38, justifyContent: 'center', paddingHorizontal: 14, borderBottomWidth: 1, borderBottomColor: theme.border }, tabText: { color: theme.text, fontSize: 12 }, code: { flex: 1, color: '#D4D4D8', fontFamily: 'monospace', fontSize: 13, lineHeight: 21, padding: 16 },
  bottom: { borderTopWidth: 1, borderTopColor: theme.border, padding: 13 }, bottomTitle: { color: theme.muted, fontSize: 9, fontWeight: '800', letterSpacing: 1.5, marginBottom: 9 }, actions: { gap: 8 }, action: { backgroundColor: theme.surfaceRaised, borderWidth: 1, borderColor: theme.border, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8 }, actionText: { color: theme.text, fontSize: 12 },
});

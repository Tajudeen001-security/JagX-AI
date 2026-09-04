import React, { useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { theme } from '../theme';

export default function ChatScreen() {
  const [message, setMessage] = useState('');
  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}>
        <View><Text style={styles.kicker}>JAGX AI</Text><Text style={styles.title}>New conversation</Text></View>
        <View style={styles.status}><View style={styles.dot} /><Text style={styles.statusText}>Online</Text></View>
      </View>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.welcome}>
          <Text style={styles.greeting}>How can I help?</Text>
          <Text style={styles.sub}>Ask, create, code, research, or work with your files.</Text>
        </View>
        <View style={styles.message}><Text style={styles.label}>JAGX</Text><Text style={styles.body}>I’m ready. I can work with the JagX runtime, tools, memory, code workspace, and multimodal inputs.</Text></View>
      </ScrollView>
      <View style={styles.composer}>
        <Pressable style={styles.attach}><Text style={styles.attachText}>+</Text></Pressable>
        <TextInput value={message} onChangeText={setMessage} placeholder="Message JagX AI..." placeholderTextColor={theme.muted} multiline style={styles.input} />
        <Pressable style={styles.send} onPress={() => setMessage('')}><Text style={styles.sendText}>↑</Text></Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.background },
  header: { paddingHorizontal: 20, paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: theme.border, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  kicker: { color: theme.accent, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
  title: { color: theme.text, fontSize: 18, fontWeight: '700', marginTop: 3 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 7, backgroundColor: theme.surfaceRaised, borderRadius: 999, paddingHorizontal: 11, paddingVertical: 7 },
  dot: { width: 7, height: 7, borderRadius: 4, backgroundColor: theme.success },
  statusText: { color: theme.muted, fontSize: 12 },
  content: { padding: 20, flexGrow: 1 },
  welcome: { marginTop: 70, marginBottom: 34 },
  greeting: { color: theme.text, fontSize: 34, fontWeight: '800', letterSpacing: -1 },
  sub: { color: theme.muted, fontSize: 15, lineHeight: 22, marginTop: 9, maxWidth: 330 },
  message: { backgroundColor: theme.surface, borderWidth: 1, borderColor: theme.border, borderRadius: theme.radius, padding: 18 },
  label: { color: theme.accent, fontSize: 10, fontWeight: '800', letterSpacing: 1.5, marginBottom: 9 },
  body: { color: theme.text, fontSize: 15, lineHeight: 23 },
  composer: { margin: 14, marginTop: 8, padding: 9, minHeight: 60, borderWidth: 1, borderColor: theme.border, borderRadius: 20, backgroundColor: theme.surface, flexDirection: 'row', alignItems: 'flex-end', gap: 8 },
  attach: { width: 38, height: 38, borderRadius: 19, backgroundColor: theme.surfaceRaised, alignItems: 'center', justifyContent: 'center' },
  attachText: { color: theme.text, fontSize: 23, fontWeight: '300' },
  input: { flex: 1, color: theme.text, fontSize: 15, maxHeight: 110, paddingHorizontal: 4, paddingVertical: 9 },
  send: { width: 38, height: 38, borderRadius: 19, backgroundColor: theme.accent, alignItems: 'center', justifyContent: 'center' },
  sendText: { color: theme.text, fontSize: 20, fontWeight: '800' },
});

import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

const files = ['src', 'app.ts', 'package.json', 'README.md'];

export function JagXCodeScreen() {
  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <View style={{ padding: 20, borderBottomWidth: 1, borderBottomColor: theme.border }}>
        <Text style={{ color: theme.text, fontSize: 28, fontWeight: '800' }}>JagX Code</Text>
        <Text style={{ color: theme.muted, marginTop: 5 }}>Build, inspect and improve projects.</Text>
      </View>
      <ScrollView contentContainerStyle={{ padding: 18, gap: 10 }}>
        {files.map((file, i) => (
          <View key={file} style={{ padding: 15, backgroundColor: theme.surface, borderRadius: 14, borderWidth: 1, borderColor: theme.border }}>
            <Text style={{ color: i === 1 ? theme.accent : theme.text, fontWeight: '700' }}>{i === 0 ? '▸ ' : '  '}{file}</Text>
          </View>
        ))}
        <View style={{ marginTop: 8, padding: 18, backgroundColor: '#0D0D0F', borderRadius: 16, borderWidth: 1, borderColor: theme.border }}>
          <Text style={{ color: theme.muted, fontFamily: 'monospace', lineHeight: 22 }}>{'const app = createJagXApp();\napp.start();'}</Text>
        </View>
      </ScrollView>
    </View>
  );
}

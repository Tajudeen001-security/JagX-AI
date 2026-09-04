import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

const items = ['Saved memories', 'Knowledge sources', 'Conversation history', 'Imported knowledge'];

export function MemoryScreen() {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: theme.background }} contentContainerStyle={{ padding: 24, gap: 14 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800' }}>Memory</Text>
      <Text style={{ color: theme.muted, lineHeight: 22 }}>Control what JagX remembers and where your knowledge comes from.</Text>
      {items.map((item) => (
        <View key={item} style={{ padding: 18, backgroundColor: theme.surface, borderRadius: theme.radius, borderWidth: 1, borderColor: theme.border }}>
          <Text style={{ color: theme.text, fontSize: 16, fontWeight: '700' }}>{item}</Text>
          <Text style={{ color: theme.muted, marginTop: 6 }}>Manage and review this collection.</Text>
        </View>
      ))}
    </ScrollView>
  );
}

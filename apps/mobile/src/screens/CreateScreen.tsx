import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

const tools = [
  ['Image', 'Create visual concepts and assets'],
  ['Audio', 'Draft narration, voice and sound workflows'],
  ['Video', 'Plan scenes, shots and production workflows'],
  ['Document', 'Generate structured documents and reports'],
  ['Story', 'Build long-form stories and storybooks'],
  ['App', 'Turn an idea into an application project'],
];

export function CreateScreen() {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: theme.background }} contentContainerStyle={{ padding: 24, gap: 14 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800' }}>Create</Text>
      <Text style={{ color: theme.muted, fontSize: 15, marginBottom: 10 }}>Choose a workspace and let JagX organize the task.</Text>
      {tools.map(([title, description]) => (
        <View key={title} style={{ backgroundColor: theme.surface, borderColor: theme.border, borderWidth: 1, borderRadius: theme.radius, padding: 20 }}>
          <Text style={{ color: theme.text, fontSize: 18, fontWeight: '700' }}>{title}</Text>
          <Text style={{ color: theme.muted, marginTop: 7, lineHeight: 21 }}>{description}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

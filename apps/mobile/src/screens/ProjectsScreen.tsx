import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

const projects = [
  ['JagX Code', 'Application development workspace', 'Active'],
  ['Research Lab', 'Research, sources and reports', 'Ready'],
  ['Creative Studio', 'Media and document projects', 'Ready'],
];

export function ProjectsScreen() {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: theme.background }} contentContainerStyle={{ padding: 24, gap: 14 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800' }}>Projects</Text>
      <Text style={{ color: theme.muted, marginBottom: 10 }}>Keep chats, files, code and generated work together.</Text>
      {projects.map(([name, description, status]) => (
        <View key={name} style={{ padding: 20, borderRadius: theme.radius, backgroundColor: theme.surface, borderWidth: 1, borderColor: theme.border }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', gap: 10 }}>
            <Text style={{ color: theme.text, fontSize: 18, fontWeight: '700' }}>{name}</Text>
            <Text style={{ color: theme.success, fontSize: 12, fontWeight: '700' }}>{status}</Text>
          </View>
          <Text style={{ color: theme.muted, marginTop: 8 }}>{description}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

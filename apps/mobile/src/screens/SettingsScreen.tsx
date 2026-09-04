import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

const settings = [
  ['Account', 'Profile and sign-in preferences'],
  ['Models', 'Choose available JagX models and behavior'],
  ['Appearance', 'Theme, density and accessibility'],
  ['Privacy', 'Memory, retention and data controls'],
  ['API', 'Developer access and API credentials'],
];

export function SettingsScreen() {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: theme.background }} contentContainerStyle={{ padding: 24, gap: 12 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800', marginBottom: 8 }}>Settings</Text>
      {settings.map(([title, description]) => (
        <View key={title} style={{ padding: 18, borderRadius: theme.radius, backgroundColor: theme.surface, borderWidth: 1, borderColor: theme.border }}>
          <Text style={{ color: theme.text, fontSize: 16, fontWeight: '700' }}>{title}</Text>
          <Text style={{ color: theme.muted, marginTop: 6 }}>{description}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

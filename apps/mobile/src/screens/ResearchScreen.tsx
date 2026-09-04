import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import { theme } from '../theme';

export function ResearchScreen() {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: theme.background }} contentContainerStyle={{ padding: 24, gap: 18 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800' }}>Research</Text>
      <View style={{ backgroundColor: theme.surface, borderWidth: 1, borderColor: theme.border, borderRadius: theme.radius, padding: 20 }}>
        <Text style={{ color: theme.text, fontSize: 18, fontWeight: '700' }}>Research workspace</Text>
        <Text style={{ color: theme.muted, marginTop: 8, lineHeight: 22 }}>Plan a question, gather sources, compare evidence, and produce a structured report.</Text>
      </View>
      {['Research plan', 'Sources', 'Evidence notes', 'Final report'].map((item, index) => (
        <View key={item} style={{ flexDirection: 'row', gap: 14, alignItems: 'center' }}>
          <View style={{ width: 34, height: 34, borderRadius: 17, backgroundColor: theme.surfaceRaised, borderWidth: 1, borderColor: theme.border, alignItems: 'center', justifyContent: 'center' }}>
            <Text style={{ color: theme.accent, fontWeight: '800' }}>{index + 1}</Text>
          </View>
          <Text style={{ color: theme.text, fontSize: 16 }}>{item}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

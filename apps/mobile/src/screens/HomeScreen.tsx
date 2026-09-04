import React from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { theme } from '../theme';

const actions = [
  ['Chat', 'Ask, reason, write and solve'],
  ['Create', 'Images, audio, video and documents'],
  ['Code', 'Build with JagX Code'],
  ['Research', 'Explore the web and synthesize sources'],
];

export function HomeScreen() {
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>JAGX AI</Text>
            <Text style={styles.title}>What are we building today?</Text>
          </View>
          <View style={styles.avatar}><Text style={styles.avatarText}>J</Text></View>
        </View>
        <View style={styles.composer}>
          <Text style={styles.composerTitle}>Start with anything</Text>
          <Text style={styles.composerHint}>Ask a question, attach files, or describe what you want to create.</Text>
          <Pressable style={styles.primary}><Text style={styles.primaryText}>New conversation</Text></Pressable>
        </View>
        <Text style={styles.section}>Explore JagX</Text>
        {actions.map(([title, description]) => (
          <Pressable key={title} style={styles.card}>
            <View style={styles.icon}><Text style={styles.iconText}>{title[0]}</Text></View>
            <View style={styles.cardCopy}><Text style={styles.cardTitle}>{title}</Text><Text style={styles.cardDescription}>{description}</Text></View>
            <Text style={styles.arrow}>›</Text>
          </Pressable>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.background },
  content: { padding: theme.spacing.lg, gap: theme.spacing.md },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md },
  eyebrow: { color: theme.accent, fontSize: 12, fontWeight: '800', letterSpacing: 2 },
  title: { color: theme.text, fontSize: 28, lineHeight: 34, fontWeight: '700', maxWidth: 290, marginTop: 6 },
  avatar: { width: 42, height: 42, borderRadius: 21, backgroundColor: theme.surfaceRaised, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: theme.border },
  avatarText: { color: theme.text, fontWeight: '700' },
  composer: { backgroundColor: theme.surface, borderColor: theme.border, borderWidth: 1, borderRadius: theme.radius, padding: theme.spacing.lg, gap: theme.spacing.sm },
  composerTitle: { color: theme.text, fontSize: 19, fontWeight: '700' },
  composerHint: { color: theme.muted, fontSize: 14, lineHeight: 21 },
  primary: { alignSelf: 'flex-start', marginTop: 8, backgroundColor: theme.accent, paddingHorizontal: 16, paddingVertical: 11, borderRadius: 12 },
  primaryText: { color: '#FFFFFF', fontWeight: '700' },
  section: { color: theme.text, fontSize: 18, fontWeight: '700', marginTop: 8 },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.surface, borderColor: theme.border, borderWidth: 1, borderRadius: theme.radius, padding: theme.spacing.md, gap: theme.spacing.md },
  icon: { width: 42, height: 42, borderRadius: 12, backgroundColor: theme.surfaceRaised, alignItems: 'center', justifyContent: 'center' },
  iconText: { color: theme.accent, fontWeight: '800', fontSize: 17 },
  cardCopy: { flex: 1 },
  cardTitle: { color: theme.text, fontSize: 16, fontWeight: '700' },
  cardDescription: { color: theme.muted, fontSize: 13, marginTop: 3 },
  arrow: { color: theme.muted, fontSize: 28 },
});

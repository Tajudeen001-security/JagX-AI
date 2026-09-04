import { Link } from 'expo-router';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

const actions = [
  ['Chat', 'Ask, reason, plan and work with JagX.'],
  ['Create', 'Images, audio, video, documents and stories.'],
  ['Code', 'Open JagX Code for projects and AI-assisted development.'],
  ['Research', 'Investigate topics with source-aware workflows.'],
  ['Memory', 'Save and organize useful knowledge.'],
  ['Projects', 'Keep conversations, files and tasks together.'],
];

export default function Home() {
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.eyebrow}>JAGX AI</Text>
        <Text style={styles.title}>Build, think, create.</Text>
        <Text style={styles.subtitle}>One workspace for intelligence, creation and software development.</Text>
        <View style={styles.primaryRow}>
          <Link href="/chat" asChild><Pressable style={styles.primary}><Text style={styles.primaryText}>Start a chat</Text></Pressable></Link>
          <Link href="/code" asChild><Pressable style={styles.secondary}><Text style={styles.secondaryText}>Open JagX Code</Text></Pressable></Link>
        </View>
        <View style={styles.grid}>
          {actions.map(([name, description]) => (
            <View key={name} style={styles.card}>
              <Text style={styles.cardTitle}>{name}</Text>
              <Text style={styles.cardText}>{description}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#09090B' },
  container: { padding: 24, gap: 16 },
  eyebrow: { color: '#A1A1AA', fontSize: 12, fontWeight: '700', letterSpacing: 2 },
  title: { color: '#FAFAFA', fontSize: 38, fontWeight: '800', marginTop: 8 },
  subtitle: { color: '#A1A1AA', fontSize: 16, lineHeight: 24, maxWidth: 560 },
  primaryRow: { flexDirection: 'row', gap: 12, marginTop: 12, flexWrap: 'wrap' },
  primary: { backgroundColor: '#F4F4F5', paddingHorizontal: 18, paddingVertical: 13, borderRadius: 14 },
  primaryText: { color: '#09090B', fontWeight: '700' },
  secondary: { borderWidth: 1, borderColor: '#27272A', paddingHorizontal: 18, paddingVertical: 13, borderRadius: 14 },
  secondaryText: { color: '#FAFAFA', fontWeight: '700' },
  grid: { gap: 12, marginTop: 20 },
  card: { backgroundColor: '#18181B', borderWidth: 1, borderColor: '#27272A', borderRadius: 18, padding: 18 },
  cardTitle: { color: '#FAFAFA', fontSize: 18, fontWeight: '700' },
  cardText: { color: '#A1A1AA', marginTop: 7, lineHeight: 21 },
});

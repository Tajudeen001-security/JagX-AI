import { useState } from 'react';
import { Pressable, SafeAreaView, StyleSheet, Text, TextInput, View } from 'react-native';

export default function Chat() {
  const [message, setMessage] = useState('');
  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Chat</Text>
        <View style={styles.empty}>
          <Text style={styles.heading}>What are we building?</Text>
          <Text style={styles.muted}>Ask JagX to reason, write, code, research or create.</Text>
        </View>
        <View style={styles.composer}>
          <TextInput value={message} onChangeText={setMessage} placeholder="Message JagX..." placeholderTextColor="#71717A" style={styles.input} multiline />
          <Pressable accessibilityLabel="Send message" style={styles.send} onPress={() => setMessage('')}><Text style={styles.sendText}>↑</Text></Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#09090B' },
  container: { flex: 1, padding: 20 },
  title: { color: '#FAFAFA', fontSize: 24, fontWeight: '800' },
  empty: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 8 },
  heading: { color: '#FAFAFA', fontSize: 25, fontWeight: '700' },
  muted: { color: '#A1A1AA', textAlign: 'center' },
  composer: { flexDirection: 'row', alignItems: 'flex-end', borderWidth: 1, borderColor: '#27272A', backgroundColor: '#18181B', borderRadius: 18, padding: 8 },
  input: { flex: 1, color: '#FAFAFA', minHeight: 44, maxHeight: 130, paddingHorizontal: 10, paddingTop: 10 },
  send: { width: 40, height: 40, borderRadius: 12, backgroundColor: '#F4F4F5', alignItems: 'center', justifyContent: 'center' },
  sendText: { color: '#09090B', fontSize: 22, fontWeight: '800' },
});

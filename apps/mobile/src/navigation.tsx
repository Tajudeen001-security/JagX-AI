import React from 'react';
import { Pressable, SafeAreaView, Text, View } from 'react-native';
import { CreateScreen } from './screens/CreateScreen';
import { MemoryScreen } from './screens/MemoryScreen';
import { ProjectsScreen } from './screens/ProjectsScreen';
import { ResearchScreen } from './screens/ResearchScreen';
import { SettingsScreen } from './screens/SettingsScreen';
import { theme } from './theme';

type Screen = 'Home' | 'Chat' | 'Create' | 'Code' | 'Research' | 'Memory' | 'Projects' | 'Settings';

const screens: Screen[] = ['Home', 'Chat', 'Create', 'Code', 'Research', 'Memory', 'Projects', 'Settings'];

export function JagXNavigation() {
  const [active, setActive] = React.useState<Screen>('Home');
  const content = active === 'Create' ? <CreateScreen /> : active === 'Research' ? <ResearchScreen /> : active === 'Memory' ? <MemoryScreen /> : active === 'Projects' ? <ProjectsScreen /> : active === 'Settings' ? <SettingsScreen /> : (
    <View style={{ flex: 1, backgroundColor: theme.background, padding: 24 }}>
      <Text style={{ color: theme.text, fontSize: 32, fontWeight: '800' }}>{active}</Text>
      <Text style={{ color: theme.muted, marginTop: 8 }}>JagX workspace</Text>
    </View>
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
      <View style={{ flex: 1 }}>{content}</View>
      <View style={{ flexDirection: 'row', borderTopWidth: 1, borderTopColor: theme.border, backgroundColor: theme.surface, paddingVertical: 8 }}>
        {screens.map((screen) => (
          <Pressable key={screen} onPress={() => setActive(screen)} style={{ flex: 1, alignItems: 'center', paddingVertical: 8 }}>
            <Text style={{ color: active === screen ? theme.accent : theme.muted, fontSize: 11, fontWeight: active === screen ? '800' : '500' }}>{screen}</Text>
          </Pressable>
        ))}
      </View>
    </SafeAreaView>
  );
}

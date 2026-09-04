import 'package:flutter/material.dart';

class JagXTheme {
  static const background = Color(0xFF09090B);
  static const surface = Color(0xFF111113);
  static const surfaceElevated = Color(0xFF18181B);
  static const border = Color(0xFF27272A);
  static const accent = Color(0xFF8B6CFF);
  static const text = Color(0xFFF4F4F5);
  static const muted = Color(0xFFA1A1AA);

  static ThemeData dark() {
    final scheme = ColorScheme.fromSeed(seedColor: accent, brightness: Brightness.dark).copyWith(surface: surface);
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: scheme,
      scaffoldBackgroundColor: background,
      appBarTheme: const AppBarTheme(backgroundColor: background, surfaceTintColor: Colors.transparent),
      cardTheme: const CardThemeData(color: surface, elevation: 0, margin: EdgeInsets.zero),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceElevated,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: const BorderSide(color: border)),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: const BorderSide(color: border)),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: const BorderSide(color: accent)),
      ),
    );
  }
}

package com.nightroadvision.app.ui.theme

import androidx.compose.ui.graphics.Color

// ---------------------------------------------------------------------------
// Night Road Vision – Color Palette
// A dark-first palette with high-contrast accents designed for visibility
// in low-light / night-vision contexts.
// ---------------------------------------------------------------------------

// --- Primary – Cyan / Teal -------------------------------------------------
val Cyan100 = Color(0xFFB2EBF2)
val Cyan200 = Color(0xFF80DEEA)
val Cyan300 = Color(0xFF4DD0E1)
val Cyan400 = Color(0xFF26C6DA)
val Cyan500 = Color(0xFF00BCD4)   // Primary
val Cyan600 = Color(0xFF00ACC1)
val Cyan700 = Color(0xFF0097A7)
val Cyan800 = Color(0xFF00838F)
val Cyan900 = Color(0xFF006064)

// --- Secondary – Amber / Warning -------------------------------------------
val Amber100 = Color(0xFFFFECB3)
val Amber200 = Color(0xFFFFE082)
val Amber300 = Color(0xFFFFD54F)
val Amber400 = Color(0xFFFFCA28)
val Amber500 = Color(0xFFFFC107)   // Secondary
val Amber600 = Color(0xFFFFB300)
val Amber700 = Color(0xFFFFA000)
val Amber800 = Color(0xFFFF8F00)
val Amber900 = Color(0xFFFF6F00)

// --- Tertiary – Red / Danger / Critical Alert ------------------------------
val Red400 = Color(0xFFEF5350)
val Red500 = Color(0xFFF44336)
val Red700 = Color(0xFFD32F2F)

// --- Neutral / Surface – Very dark greys (suitable for night use) ----------
val SurfaceDark05 = Color(0xFF0A0A0A)   // Deepest background
val SurfaceDark10 = Color(0xFF0D0D0D)
val SurfaceDark20 = Color(0xFF121212)   // Default background
val SurfaceDark30 = Color(0xFF1A1A1A)   // Elevated surface
val SurfaceDark40 = Color(0xFF212121)   // Card / container
val SurfaceDark50 = Color(0xFF2A2A2A)   // Higher elevation
val SurfaceDark60 = Color(0xFF333333)   // Outlines, dividers
val SurfaceDark70 = Color(0xFF3D3D3D)

// --- On-colors (text / icons on colored surfaces) -------------------------
val OnPrimary = Color(0xFF003133)        // Dark text on cyan
val OnSecondary = Color(0xFF3E2700)      // Dark text on amber
val OnSurface = Color(0xFFE0E0E0)       // Light text on dark surfaces
val OnSurfaceVariant = Color(0xFF9E9E9E) // Muted text on dark surfaces

// --- Detection overlay colors (bounding boxes, masks, etc.) ----------------
val DetectionBoxPerson = Color(0xFF00E5FF)    // Bright cyan – pedestrian
val DetectionBoxVehicle = Color(0xFF76FF03)   // Bright green – vehicle
val DetectionBoxAnimal = Color(0xFFFFAB00)    // Amber – animal
val DetectionBoxObstacle = Color(0xFFFF1744)  // Red – obstacle / danger
val DetectionBoxLane = Color(0xFF40C4FF)      // Light blue – lane markings
val DetectionBoxGeneral = Color(0xFFE0E0E0)   // White – unclassified

// --- Status / Feedback colors ---------------------------------------------
val StatusActive = Color(0xFF00E676)     // Green – active / ok
val StatusWarning = Color(0xFFFFAB00)    // Amber – warning
val StatusError = Color(0xFFFF1744)      // Red – error
val StatusInactive = Color(0xFF616161)   // Grey – inactive

// --- Performance stat colors (charts, graphs) -----------------------------
val StatPrimary = Color(0xFF00BCD4)
val StatSecondary = Color(0xFFFFC107)
val StatTertiary = Color(0xFF76FF03)
val StatGridLine = Color(0xFF2A2A2A)

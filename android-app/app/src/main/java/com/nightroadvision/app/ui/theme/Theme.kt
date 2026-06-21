package com.nightroadvision.app.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ---------------------------------------------------------------------------
// Night Road Vision – Material 3 Color Scheme
// Two schemes are defined but the dark scheme is the primary / default.
// The light scheme exists only as a fallback; the app will prefer dark.
// ---------------------------------------------------------------------------

private val DarkColorScheme = darkColorScheme(
    // Primary
    primary = Cyan500,
    onPrimary = OnPrimary,
    primaryContainer = Cyan800,
    onPrimaryContainer = Cyan100,

    // Secondary
    secondary = Amber500,
    onSecondary = OnSecondary,
    secondaryContainer = Amber700,
    onSecondaryContainer = Amber100,

    // Tertiary
    tertiary = Red500,
    onTertiary = Color.White,
    tertiaryContainer = Red700,
    onTertiaryContainer = Color.White,

    // Background / Surface
    background = SurfaceDark20,
    onBackground = OnSurface,
    surface = SurfaceDark30,
    onSurface = OnSurface,
    surfaceVariant = SurfaceDark40,
    onSurfaceVariant = OnSurfaceVariant,

    // Outline
    outline = SurfaceDark60,
    outlineVariant = SurfaceDark50,

    // Error
    error = Red500,
    onError = Color.White,
    errorContainer = Red700,
    onErrorContainer = Color.White,

    // Inverse
    inverseSurface = OnSurface,
    inverseOnSurface = SurfaceDark20,
    inversePrimary = Cyan700,

    // Scrim
    scrim = Color.Black,
)

private val LightColorScheme = lightColorScheme(
    primary = Cyan700,
    onPrimary = Color.White,
    primaryContainer = Cyan200,
    onPrimaryContainer = Cyan900,

    secondary = Amber600,
    onSecondary = Color.White,
    secondaryContainer = Amber200,
    onSecondaryContainer = Amber900,

    tertiary = Red500,
    onTertiary = Color.White,
    tertiaryContainer = Red400,
    onTertiaryContainer = Color.White,

    background = Color(0xFFFDFDFD),
    onBackground = Color(0xFF1A1C1E),
    surface = Color(0xFFFDFDFD),
    onSurface = Color(0xFF1A1C1E),
    surfaceVariant = Color(0xFFE2E2E9),
    onSurfaceVariant = Color(0xFF44474F),

    outline = Color(0xFF74777F),
    outlineVariant = Color(0xFFC4C6CF),

    error = Red500,
    onError = Color.White,
    errorContainer = Red400,
    onErrorContainer = Color(0xFF410002),
)

// ---------------------------------------------------------------------------
// Theme composable
// ---------------------------------------------------------------------------

/**
 * NightRoadVisionTheme
 *
 * This theme **always** uses the dark color scheme regardless of the system
 * setting. Night-vision applications must present a dark UI to avoid sudden
 * bright flashes that impair the driver's dark-adapted vision.
 *
 * @param content  The composable tree to theme.
 */
@Composable
fun NightRoadVisionTheme(
    content: @Composable () -> Unit,
) {
    val colorScheme = DarkColorScheme

    // Ensure the system status bar and navigation bar also use dark surfaces.
    val view = LocalView.current
    val activity = view.context as? Activity
    if (!view.isInEditMode && activity != null) {
        SideEffect {
            val window = activity.window
            window.statusBarColor = SurfaceDark20.toArgb()
            window.navigationBarColor = SurfaceDark20.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = false
                isAppearanceLightNavigationBars = false
            }
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = NightRoadVisionTypography,
        content = content,
    )
}

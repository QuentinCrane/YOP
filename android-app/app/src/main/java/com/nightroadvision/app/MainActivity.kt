package com.nightroadvision.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.nightroadvision.app.ui.screen.MainScreen

/**
 * Main entry point for Night Road Vision.
 *
 * Handles camera permission requests and hosts the Compose UI.
 */
class MainActivity : ComponentActivity() {

    private var permissionGranted = mutableStateOf(false)

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            permissionGranted.value = isGranted
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check initial permission state
        permissionGranted.value = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        // Request camera permission if not already granted
        if (!permissionGranted.value) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            val hasPermission by permissionGranted

            if (hasPermission) {
                MainScreen(modifier = Modifier.fillMaxSize())
            } else {
                NoPermissionScreen()
            }
        }
    }
}

@Composable
private fun NoPermissionScreen() {
    androidx.compose.foundation.layout.Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        androidx.compose.material3.Text(
            text = "夜视路况需要相机权限才能运行。\n请授予权限后重新启动应用。",
            color = androidx.compose.ui.graphics.Color.White,
            style = androidx.compose.material3.MaterialTheme.typography.bodyLarge
        )
    }
}

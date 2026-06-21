package com.nightroadvision.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.nightroadvision.app.ui.screen.MainScreen
import com.nightroadvision.app.ui.theme.NightRoadVisionTheme

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

        // Keep screen on while driving — prevents auto screen-off
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Check initial permission state
        permissionGranted.value = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        // Request camera permission if not already granted
        if (!permissionGranted.value) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            NightRoadVisionTheme {
                val hasPermission by permissionGranted

                if (hasPermission) {
                    MainScreen(modifier = Modifier.fillMaxSize())
                } else {
                    NoPermissionScreen(
                        onRequestPermission = {
                            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Refresh after returning from system settings; otherwise the permission
        // screen can remain stuck until the process is restarted.
        permissionGranted.value = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA,
        ) == PackageManager.PERMISSION_GRANTED
    }
}

@Composable
private fun NoPermissionScreen(onRequestPermission: () -> Unit) {
    androidx.compose.foundation.layout.Box(
        modifier = Modifier
            .fillMaxSize()
            .background(androidx.compose.ui.graphics.Color(0xFF0D1117)),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        androidx.compose.foundation.layout.Column(
            horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
            modifier = Modifier.padding(24.dp),
        ) {
            androidx.compose.material3.Text(
                text = "夜视路况需要相机权限才能运行",
                color = androidx.compose.ui.graphics.Color.White,
                style = androidx.compose.material3.MaterialTheme.typography.bodyLarge
            )
            androidx.compose.foundation.layout.Spacer(Modifier.height(16.dp))
            androidx.compose.material3.Button(onClick = onRequestPermission) {
                androidx.compose.material3.Text("授予相机权限")
            }
        }
    }
}

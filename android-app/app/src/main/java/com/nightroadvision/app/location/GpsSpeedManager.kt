package com.nightroadvision.app.location

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

/**
 * Provides real-time GPS speed via FusedLocationProviderClient.
 * Speed is derived from `Location.speed` (m/s) and converted to km/h.
 */
class GpsSpeedManager(
    private val context: Context,
    private val onSpeedUpdate: (Float) -> Unit,
) {
    companion object {
        private const val TAG = "GpsSpeedManager"
        private const val UPDATE_INTERVAL_MS = 1000L
    }

    private val client: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private var running = false

    private val callback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc: Location = result.lastLocation ?: return
            val speedKmh = loc.speed * 3.6f
            Log.d(TAG, "speed=${"%.1f".format(speedKmh)} km/h, accuracy=${loc.speedAccuracyMetersPerSecond}")
            onSpeedUpdate(speedKmh)
        }
    }

    fun start() {
        if (running) return
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "Location permission not granted")
            return
        }

        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, UPDATE_INTERVAL_MS)
            .setMinUpdateIntervalMillis(500)
            .build()

        try {
            client.requestLocationUpdates(request, callback, Looper.getMainLooper())
            running = true
            Log.i(TAG, "GPS speed updates started")
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException: ${e.message}")
        }
    }

    fun stop() {
        if (!running) return
        client.removeLocationUpdates(callback)
        running = false
        Log.i(TAG, "GPS speed updates stopped")
    }
}

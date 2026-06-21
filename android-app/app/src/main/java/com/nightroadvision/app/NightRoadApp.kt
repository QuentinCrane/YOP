package com.nightroadvision.app

import android.app.Application

/**
 * NightRoadApp
 *
 * Custom [Application] subclass for the Night Road Vision app. This is the
 * first component that Android instantiates when the process starts, making
 * it the correct place for app-wide initialisation such as:
 *
 * - Setting up logging / crash-reporting frameworks.
 * - Initialising dependency-injection containers.
 * - Configuring third-party SDKs that require an application context.
 * - Pre-loading model files or shared preferences.
 *
 * Register this class in `AndroidManifest.xml` with:
 * ```xml
 * <application android:name=".NightRoadApp" ... >
 * ```
 */
class NightRoadApp : Application() {

    override fun onCreate() {
        super.onCreate()
        instance = this

        // 初始化崩溃日志记录器
        CrashLogger.init(this)

        // 初始化文件日志（路径: /sdcard/Android/data/com.nightroadvision.app/files/NightRoadVision/app_log.txt）
        FileLogger.init(this)

        // ------------------------------------------------------------------
        // App-wide initialisation goes here. For example:
        //
        //   if (BuildConfig.DEBUG) {
        //       Timber.plant(Timber.DebugTree())
        //   }
        //
        //   InferenceEngine.init(this)
        // ------------------------------------------------------------------
    }

    companion object {
        /**
         * Convenient static reference to the [Application] instance.
         * Avoid holding strong references to this in long-lived objects to
         * prevent memory leaks; prefer injecting it where possible.
         */
        @Volatile
        private var instance: NightRoadApp? = null

        /** Returns the singleton [NightRoadApp] instance. */
        fun getInstance(): NightRoadApp =
            instance ?: throw IllegalStateException(
                "NightRoadApp.onCreate() has not been called yet."
            )
    }
}

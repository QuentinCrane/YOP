package com.nightroadvision.app

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 本地文件日志记录器
 * 日志保存路径: /sdcard/Android/data/com.nightroadvision.app/files/NightRoadVision/app_log.txt
 * 用于排查运行时问题，不依赖 logcat
 */
object FileLogger {

    private const val TAG = "FileLogger"
    private const val DIR_NAME = "NightRoadVision"
    private const val FILE_NAME = "app_log.txt"
    private const val MAX_FILE_SIZE = 5 * 1024 * 1024L // 5MB

    @Volatile
    private var logDir: File? = null
    private val writeLock = Any()

    fun init(context: Context) {
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        logDir = File(baseDir, DIR_NAME)
        logDir?.mkdirs()
        i(TAG, "FileLogger initialized, log dir: ${logDir?.absolutePath}")
    }

    fun i(tag: String, message: String) {
        Log.i(tag, message)
        append("I", tag, message)
    }

    fun w(tag: String, message: String) {
        Log.w(tag, message)
        append("W", tag, message)
    }

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        Log.e(tag, message, throwable)
        val fullMessage = if (throwable != null) {
            "$message\n${throwable.javaClass.simpleName}: ${throwable.message}"
        } else message
        append("E", tag, fullMessage)
    }

    fun d(tag: String, message: String) {
        Log.d(tag, message)
        // Per-frame camera/inference diagnostics are intentionally kept out of the
        // on-disk log. Synchronous file appends at detection FPS cause avoidable I/O,
        // heat and frame jitter during a ride. Info/warning/error events remain saved.
    }

    private fun append(level: String, tag: String, message: String) {
        try {
            synchronized(writeLock) {
                val dir = logDir ?: return
                val file = File(dir, FILE_NAME)

                // Rotate if too large. Remove the previous generation first because
                // File.renameTo does not replace an existing destination on Android.
                if (file.exists() && file.length() > MAX_FILE_SIZE) {
                    val backup = File(dir, "app_log_prev.txt")
                    if (backup.exists()) backup.delete()
                    file.renameTo(backup)
                }

                val timestamp = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()).format(Date())
                file.appendText("[$timestamp] $level/$tag: $message\n")
            }
        } catch (_: Exception) {
            // 写日志失败不能影响主逻辑
        }
    }

    fun getLogFilePath(): String = logDir?.let { File(it, FILE_NAME).absolutePath } ?: FILE_NAME
}

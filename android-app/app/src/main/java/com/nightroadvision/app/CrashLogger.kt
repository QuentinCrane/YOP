package com.nightroadvision.app

import android.content.Context
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 全局崩溃日志记录器
 * 将崩溃信息保存到 /sdcard/NightRoadVision/crash_log.txt
 */
class CrashLogger(private val context: Context) : Thread.UncaughtExceptionHandler {

    private val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()

    companion object {
        private const val DIR_NAME = "NightRoadVision"
        private const val FILE_NAME = "crash_log.txt"

        fun init(context: Context) {
            Thread.setDefaultUncaughtExceptionHandler(CrashLogger(context.applicationContext))
        }
    }

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        try {
            // 使用 app 外部目录，不需要存储权限
            // 路径: /sdcard/Android/data/com.nightroadvision.app/files/NightRoadVision/
            val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
            val logDir = File(baseDir, DIR_NAME)
            if (!logDir.exists()) logDir.mkdirs()

            val logFile = File(logDir, FILE_NAME)
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

            val sw = StringWriter()
            throwable.printStackTrace(PrintWriter(sw))

            val logEntry = buildString {
                appendLine("========================================")
                appendLine("时间: $timestamp")
                appendLine("线程: ${thread.name}")
                appendLine("异常: ${throwable.javaClass.name}")
                appendLine("信息: ${throwable.message}")
                appendLine("堆栈:")
                appendLine(sw.toString())
                appendLine()
            }

            logFile.appendText(logEntry)
        } catch (_: Exception) {
            // 如果写日志本身失败，忽略
        }

        // 交给默认处理器（系统会显示"应用已停止"对话框）
        defaultHandler?.uncaughtException(thread, throwable)
    }
}

package ir.blindvision.assistant

import android.content.Context

object Preferences {
    private const val FILE = "assistant"
    private const val CONFIDENCE = "confidence"
    private const val COOLDOWN = "cooldown_sec"
    private const val DISCLAIMER = "disclaimer_seen"

    fun confidence(context: Context): Float {
        return context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .getFloat(CONFIDENCE, 0.40f)
    }

    fun cooldownSeconds(context: Context): Int {
        return context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .getInt(COOLDOWN, 18)
    }

    fun save(context: Context, confidence: Float, cooldownSeconds: Int) {
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .edit()
            .putFloat(CONFIDENCE, confidence)
            .putInt(COOLDOWN, cooldownSeconds)
            .apply()
    }

    fun disclaimerSeen(context: Context): Boolean {
        return context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .getBoolean(DISCLAIMER, false)
    }

    fun markDisclaimerSeen(context: Context) {
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(DISCLAIMER, true)
            .apply()
    }
}

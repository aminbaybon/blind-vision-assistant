package ir.blindvision.assistant.vision

import android.graphics.RectF

enum class Side {
    LEFT,
    CENTER,
    RIGHT,
}

data class Seen(
    val className: String,
    val confidence: Float,
    val side: Side,
    val box: RectF,
    val confirmed: Boolean,
) {
    fun area(): Float = (box.width() * box.height()).coerceAtLeast(0f)
}

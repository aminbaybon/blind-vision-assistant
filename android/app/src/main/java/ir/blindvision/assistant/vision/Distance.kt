package ir.blindvision.assistant.vision

import android.graphics.RectF
import kotlin.math.tan

/**
 * Retired pinhole guess. Speech does not call this. The height, field of
 * view, horizon, and object heights were never checked against measured
 * distances, so they are not announced as meters.
 */
object Distance {
    private const val CAMERA_HEIGHT_M = 1.35
    private const val HORIZONTAL_FOV_DEG = 72.0
    private const val HORIZON = 0.50
    private val focalFactor: Double = (4.0 / 3.0) / (2.0 * tan(Math.toRadians(HORIZONTAL_FOV_DEG / 2.0)))

    fun meters(box: RectF): Float? = groundMeters(box)

    /** Ground contact when the box reaches the floor, otherwise a typical object height. */
    fun meters(box: RectF, className: String): Float? {
        val ground = groundMeters(box)
        val sized = sizeMeters(box, className)
        return when {
            ground != null && sized != null -> {
                val high = maxOf(ground, sized)
                val low = minOf(ground, sized)
                if (high / low < 2.2f) (ground + sized) / 2f else ground
            }
            ground != null -> ground
            else -> sized
        }
    }

    /** Coarse range. Keeps the previous bucket until the estimate clearly leaves it. */
    fun stableBand(previous: Int, meters: Float?): Int {
        if (meters == null) return if (previous >= 0) previous else -1
        val raw = rawBand(meters)
        if (previous < 0 || raw == previous) return raw
        val low = edge(previous)
        val high = edge(previous + 1)
        if (meters < low - 0.8f || meters > high + 0.8f) return raw
        return previous
    }

    fun rawBand(meters: Float): Int = when {
        meters < 2.5f -> 0
        meters < 5f -> 1
        meters < 9f -> 2
        meters < 16f -> 3
        else -> 4
    }

    private fun groundMeters(box: RectF): Float? {
        val belowHorizon = box.bottom - HORIZON
        if (belowHorizon <= 0.04) return null
        val meters = CAMERA_HEIGHT_M * focalFactor / belowHorizon
        if (meters < 0.4 || meters > 25.0) return null
        return meters.toFloat()
    }

    private fun sizeMeters(box: RectF, className: String): Float? {
        val height = typicalHeight[className] ?: return null
        if (box.height() < 0.05f) return null
        val meters = height * focalFactor / box.height().toDouble()
        if (meters < 0.4 || meters > 25.0) return null
        return meters.toFloat()
    }

    private fun edge(band: Int): Float = when (band) {
        0 -> 0f
        1 -> 2.5f
        2 -> 5f
        3 -> 9f
        else -> 16f
    }

    private val typicalHeight = mapOf(
        "person" to 1.65,
        "bicycle" to 1.10,
        "car" to 1.50,
        "motorcycle" to 1.20,
        "bus" to 3.00,
        "truck" to 2.80,
        "chair" to 0.90,
        "dining table" to 0.75,
        "bench" to 0.50,
        "dog" to 0.55,
        "cat" to 0.30,
        "couch" to 0.80,
    )
}

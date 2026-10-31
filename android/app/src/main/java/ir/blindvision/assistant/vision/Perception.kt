package ir.blindvision.assistant.vision

import android.graphics.RectF
import kotlin.math.abs

/**
 * Qualitative reading of a MiDaS map. Larger values are treated as closer,
 * which is the MiDaS inverse-depth convention. Nothing here is a meter.
 *
 * Keep the numeric rules aligned with `perception/geometry.py`.
 */
enum class Proximity { VERY_NEAR, NEAR, MID, FAR, UNKNOWN }

enum class Reliability { SUFFICIENT, WEAK, INSUFFICIENT }

enum class RegionState {
    POSSIBLE_FRONT_OBSTRUCTION,
    POSSIBLE_SIDE_SURFACE,
    NO_OBSTRUCTION_DETECTED,
    UNKNOWN,
}

data class DepthRange(val far: Float, val near: Float, val span: Float)

data class ProximityReading(
    val proximity: Proximity,
    val rank: Float?,
    val reliability: Reliability,
    val samples: Int,
)

data class RegionReading(val state: RegionState, val rank: Float?)

data class ForwardAssessment(
    val left: RegionReading,
    val center: RegionReading,
    val right: RegionReading,
)

data class SceneSnapshot(
    val assessment: ForwardAssessment,
    val front: Seen?,
    val proximity: Proximity,
    val reliability: Reliability,
    val depthFresh: Boolean,
)

class RelativeDepthAnalyzer {
    fun range(frame: DepthFrame): DepthRange? {
        val values = collect(frame, 0.20f, 0.80f, 0.15f, 0.95f)
        if (values.size < 12) return null
        values.sort()
        val far = percentile(values, 0.20f)
        val near = percentile(values, 0.90f)
        val mid = percentile(values, 0.50f)
        val span = near - far
        if (!span.isFinite() || span < 1e-3f || span < abs(mid) * 0.02f) return null
        return DepthRange(far, near, span)
    }

    fun median(frame: DepthFrame, x0: Float, x1: Float, y0: Float, y1: Float): Float? {
        val values = collect(frame, x0, x1, y0, y1)
        if (values.isEmpty()) return null
        return trimmedMedian(values)
    }

    fun collect(frame: DepthFrame, x0: Float, x1: Float, y0: Float, y1: Float): FloatArray {
        val side = frame.side
        val left = (x0.coerceIn(0f, 1f) * (side - 1)).toInt()
        val right = (x1.coerceIn(0f, 1f) * (side - 1)).toInt().coerceAtLeast(left)
        val top = (y0.coerceIn(0f, 1f) * (side - 1)).toInt()
        val bottom = (y1.coerceIn(0f, 1f) * (side - 1)).toInt().coerceAtLeast(top)
        val step = ((right - left + bottom - top) / 24).coerceAtLeast(1)
        val out = ArrayList<Float>(48)
        var y = top
        while (y <= bottom) {
            var x = left
            while (x <= right) {
                val value = frame.values[y * side + x]
                if (value.isFinite()) out += value
                x += step
            }
            y += step
        }
        return out.toFloatArray()
    }

    fun rank(value: Float, range: DepthRange): Float {
        return ((value - range.far) / range.span).coerceIn(0f, 1.5f)
    }

    companion object {
        fun percentile(sorted: FloatArray, fraction: Float): Float {
            val index = ((sorted.size - 1) * fraction).toInt().coerceIn(0, sorted.lastIndex)
            return sorted[index]
        }

        fun trimmedMedian(values: FloatArray): Float? {
            if (values.isEmpty()) return null
            val sorted = values.copyOf()
            sorted.sort()
            val drop = if (sorted.size >= 20) (sorted.size * 0.10f).toInt() else 0
            val end = sorted.size - drop
            if (end <= drop) return null
            return sorted[(drop + end - 1) / 2]
        }
    }
}

class ObstacleProximityAnalyzer(
    private val depth: RelativeDepthAnalyzer = RelativeDepthAnalyzer(),
) {
    fun read(frame: DepthFrame, box: RectF, previousRank: Float?): ProximityReading {
        val range = depth.range(frame) ?: return unknown()
        val inner = depth.collect(
            frame,
            box.left + box.width() * 0.25f,
            box.right - box.width() * 0.25f,
            box.top + box.height() * 0.25f,
            box.bottom - box.height() * 0.25f,
        )
        val value = RelativeDepthAnalyzer.trimmedMedian(inner) ?: return unknown()
        val rank = depth.rank(value, range)
        if (previousRank != null && abs(rank - previousRank) > 0.50f) {
            return ProximityReading(Proximity.UNKNOWN, rank, Reliability.INSUFFICIENT, inner.size)
        }
        val proximity = when {
            rank >= 0.75f -> Proximity.VERY_NEAR
            rank >= 0.55f -> Proximity.NEAR
            rank >= 0.35f -> Proximity.MID
            else -> Proximity.FAR
        }
        val reliability = when {
            inner.size < 8 -> Reliability.INSUFFICIENT
            inner.size < 12 || box.width() * box.height() < 0.02f -> Reliability.WEAK
            else -> Reliability.SUFFICIENT
        }
        return ProximityReading(proximity, rank, reliability, inner.size)
    }

    /** Inner pixels are closer than the surrounding ring, in relative units. */
    fun protrudes(frame: DepthFrame, box: RectF, range: DepthRange?): Boolean {
        val scale = range ?: return false
        val inner = depth.median(
            frame,
            box.left + box.width() * 0.25f,
            box.right - box.width() * 0.25f,
            box.top + box.height() * 0.25f,
            box.bottom - box.height() * 0.25f,
        ) ?: return false
        val around = depth.median(
            frame,
            (box.left - 0.08f).coerceAtLeast(0f),
            (box.right + 0.08f).coerceAtMost(1f),
            (box.top - 0.08f).coerceAtLeast(0f),
            (box.bottom + 0.08f).coerceAtMost(1f),
        ) ?: return false
        return inner - around > scale.span * 0.12f
    }

    private fun unknown() = ProximityReading(Proximity.UNKNOWN, null, Reliability.INSUFFICIENT, 0)
}

class DistanceReliabilityEvaluator {
    fun evaluate(reading: ProximityReading): Reliability {
        if (reading.proximity == Proximity.UNKNOWN || reading.samples < 8) return Reliability.INSUFFICIENT
        return reading.reliability
    }
}

class ForwardObstacleAnalyzer(
    private val depth: RelativeDepthAnalyzer = RelativeDepthAnalyzer(),
) {
    fun analyze(frame: DepthFrame): ForwardAssessment {
        val range = depth.range(frame) ?: return ForwardAssessment(unknown(), unknown(), unknown())
        val center = column(frame, range, 0.34f, 0.66f, front = true)
        val left = column(frame, range, 0.02f, 0.30f, front = false)
        val right = column(frame, range, 0.70f, 0.98f, front = false)
        return ForwardAssessment(left, center, right)
    }

    private fun column(
        frame: DepthFrame,
        range: DepthRange,
        x0: Float,
        x1: Float,
        front: Boolean,
    ): RegionReading {
        val upper = depth.median(frame, x0, x1, 0.35f, 0.55f)
        val lower = depth.median(frame, x0, x1, 0.70f, 0.90f)
        val whole = depth.median(frame, x0, x1, 0.38f, 0.82f)
        if (upper == null || lower == null || whole == null) return unknown()
        val rank = depth.rank(whole, range)
        val flat = abs(upper - lower) < range.span * 0.22f
        val receding = lower - upper > range.span * 0.18f
        val state = when {
            front && rank >= 0.62f && flat && !receding -> RegionState.POSSIBLE_FRONT_OBSTRUCTION
            !front && rank >= 0.60f && flat && !receding -> RegionState.POSSIBLE_SIDE_SURFACE
            receding -> RegionState.NO_OBSTRUCTION_DETECTED
            rank < 0.45f -> RegionState.NO_OBSTRUCTION_DETECTED
            else -> RegionState.UNKNOWN
        }
        return RegionReading(state, rank)
    }

    private fun unknown() = RegionReading(RegionState.UNKNOWN, null)
}

class SceneUnderstanding {
    private val depth = RelativeDepthAnalyzer()
    private val proximity = ObstacleProximityAnalyzer(depth)
    private val reliability = DistanceReliabilityEvaluator()
    private val forward = ForwardObstacleAnalyzer(depth)
    private val ranks = HashMap<String, Float>()

    fun interpret(frame: DepthFrame?, items: List<Seen>): SceneSnapshot {
        if (frame == null) {
            return SceneSnapshot(
                assessment = ForwardAssessment(
                    RegionReading(RegionState.UNKNOWN, null),
                    RegionReading(RegionState.UNKNOWN, null),
                    RegionReading(RegionState.UNKNOWN, null),
                ),
                front = pick(items, blocked = false) { false },
                proximity = Proximity.UNKNOWN,
                reliability = Reliability.INSUFFICIENT,
                depthFresh = false,
            )
        }
        val assessment = forward.analyze(frame)
        val range = depth.range(frame)
        val blocked = assessment.center.state == RegionState.POSSIBLE_FRONT_OBSTRUCTION
        val front = pick(items, blocked) { box -> proximity.protrudes(frame, box, range) }
        val reading = if (front == null) {
            null
        } else {
            val key = front.className + front.side.name
            val previous = ranks[key]
            val next = proximity.read(frame, front.box, previous)
            if (next.rank != null) ranks[key] = next.rank
            next
        }
        return SceneSnapshot(
            assessment = assessment,
            front = front,
            proximity = reading?.proximity ?: Proximity.UNKNOWN,
            reliability = if (reading == null) Reliability.INSUFFICIENT else reliability.evaluate(reading),
            depthFresh = true,
        )
    }

    private fun pick(items: List<Seen>, blocked: Boolean, protrudes: (RectF) -> Boolean): Seen? {
        val known = items.filter { believable(it, blocked, protrudes(it.box)) }
        val ahead = known.filter { it.side == Side.CENTER }
        val pool = if (ahead.isNotEmpty()) ahead else known
        return pool.maxByOrNull { it.confidence }
    }

    private fun believable(item: Seen, blocked: Boolean, protrudes: Boolean): Boolean {
        if (!item.confirmed || item.className !in SPOKEN) return false
        if (item.className != "person") return item.confidence >= 0.40f
        if (blocked && !protrudes) return false
        if (item.area() > 0.35f && !protrudes) return false
        if (item.confidence >= 0.72f) return true
        return protrudes && item.confidence >= 0.45f
    }

    companion object {
        val SPOKEN = setOf(
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "chair", "dining table", "bench", "dog", "cat", "stop sign",
            "traffic light", "backpack", "suitcase", "couch", "potted plant",
        )
    }
}

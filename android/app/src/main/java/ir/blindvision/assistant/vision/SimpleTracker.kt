package ir.blindvision.assistant.vision

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/** Keeps a short memory of boxes so a one-frame flicker is not treated as a stable object. */
class SimpleTracker {
    private data class Track(val className: String, var box: RectF, var hits: Int, var missed: Int)

    private val tracks = mutableListOf<Track>()

    fun update(seen: List<Seen>): List<Seen> {
        val unused = seen.toMutableList()
        val next = mutableListOf<Track>()
        val confirmed = mutableListOf<Seen>()
        for (track in tracks) {
            val match = unused
                .filter { it.className == track.className }
                .maxByOrNull { overlap(track.box, it.box) }
            if (match != null && overlap(track.box, match.box) >= 0.30f) {
                unused.remove(match)
                track.box = RectF(match.box)
                track.hits += 1
                track.missed = 0
                next += track
                confirmed += match.copy(confirmed = track.hits >= 2 || match.confidence >= 0.60f)
            } else if (track.missed < 2) {
                track.missed += 1
                next += track
            }
        }
        for (item in unused) {
            next += Track(item.className, RectF(item.box), hits = 1, missed = 0)
            if (item.confidence >= 0.60f) confirmed += item.copy(confirmed = true)
        }
        tracks.clear()
        tracks.addAll(next)
        return confirmed
    }

    private fun overlap(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val width = (right - left).coerceAtLeast(0f)
        val height = (bottom - top).coerceAtLeast(0f)
        val intersection = width * height
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        if (union <= 0f) return 0f
        return intersection / union
    }
}

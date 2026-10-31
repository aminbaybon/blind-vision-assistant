package ir.blindvision.assistant.voice

import ir.blindvision.assistant.vision.Proximity
import ir.blindvision.assistant.vision.RegionState
import ir.blindvision.assistant.vision.Reliability
import ir.blindvision.assistant.vision.SceneSnapshot
import ir.blindvision.assistant.vision.Seen
import ir.blindvision.assistant.vision.Side

data class Alert(val text: String, val priority: Int, val interrupt: Boolean)

/**
 * Spoken Persian stays qualitative. A missing obstacle is not described as a
 * safe path, and no sentence states a distance in meters.
 */
object PersianCopy {
    private val nouns = mapOf(
        "person" to "نفر",
        "bicycle" to "دوچرخه",
        "car" to "خودرو",
        "motorcycle" to "موتور",
        "bus" to "اتوبوس",
        "truck" to "کامیون",
        "chair" to "صندلی",
        "dining table" to "میز",
        "bench" to "نیمکت",
        "dog" to "سگ",
        "cat" to "گربه",
        "stop sign" to "تابلوی ایست",
        "traffic light" to "چراغ راهنما",
        "backpack" to "کوله‌پشتی",
        "suitcase" to "چمدان",
        "couch" to "مبل",
        "potted plant" to "گلدان",
    )

    fun phrase(snapshot: SceneSnapshot): Spoken? {
        val front = snapshot.front
        val close = snapshot.reliability == Reliability.SUFFICIENT &&
            (snapshot.proximity == Proximity.VERY_NEAR || snapshot.proximity == Proximity.NEAR)
        if (front != null && close) {
            return Spoken(objectKey(front, "close"), objectSentence(front, near = true), 90)
        }
        if (snapshot.depthFresh &&
            snapshot.assessment.center.state == RegionState.POSSIBLE_FRONT_OBSTRUCTION
        ) {
            return Spoken(
                "front_block",
                "مانعی در قسمت روبه‌روی تصویر دیده می‌شود. فاصله دقیق مانع مشخص نیست.",
                100,
            )
        }
        if (front != null) {
            return Spoken(objectKey(front, "seen"), objectSentence(front, near = false), 60)
        }
        if (snapshot.depthFresh && snapshot.assessment.left.state == RegionState.POSSIBLE_SIDE_SURFACE) {
            return Spoken(
                "side_left",
                "یک سطح نزدیک در سمت چپ تشخیص داده شده است. فاصله دقیق مشخص نیست.",
                70,
            )
        }
        if (snapshot.depthFresh && snapshot.assessment.right.state == RegionState.POSSIBLE_SIDE_SURFACE) {
            return Spoken(
                "side_right",
                "یک سطح نزدیک در سمت راست تشخیص داده شده است. فاصله دقیق مشخص نیست.",
                70,
            )
        }
        return null
    }

    fun describe(snapshot: SceneSnapshot?): String {
        if (snapshot == null) {
            return "شیء مشخصی در تصویر تشخیص داده نشد. این به معنی باز بودن مسیر نیست."
        }
        val lead = phrase(snapshot)?.text
        if (lead == null) {
            return "شیء مشخصی در تصویر تشخیص داده نشد. این به معنی باز بودن مسیر نیست."
        }
        return lead
    }

    private fun objectKey(item: Seen, kind: String): String = "obj|${item.className}|$kind"

    private fun objectSentence(item: Seen, near: Boolean): String {
        val noun = if (item.className == "person") "نفر" else nouns[item.className] ?: "شیء"
        val place = "یک $noun روبروی شماست"
        val closeness = if (near) " و نزدیک به نظر می‌رسد" else ""
        return "$place$closeness. فاصله دقیق مانع مشخص نیست."
    }
}

data class Spoken(val key: String, val text: String, val priority: Int)

class AlertEngine(var cooldownMs: Long = 18_000L) {
    private var stableKey = ""
    private var stableSince = 0L
    private var lastSpokenKey = ""
    private var lastSpokenAt = 0L
    private var lastPriority = 0

    fun next(snapshot: SceneSnapshot, nowMs: Long): Alert? {
        val spoken = PersianCopy.phrase(snapshot) ?: return null
        if (spoken.key != stableKey) {
            stableKey = spoken.key
            stableSince = nowMs
            return null
        }
        if (nowMs - stableSince < 1_200L) return null
        val moreUrgent = lastSpokenKey.isNotEmpty() && spoken.priority >= lastPriority + 15
        val same = spoken.key == lastSpokenKey
        val wait = when {
            moreUrgent -> 8_000L
            same -> maxOf(cooldownMs, 18_000L)
            else -> maxOf(cooldownMs, 12_000L)
        }
        if (nowMs - lastSpokenAt < wait) return null
        val interrupt = moreUrgent
        lastSpokenKey = spoken.key
        lastSpokenAt = nowMs
        lastPriority = spoken.priority
        return Alert(spoken.text, spoken.priority, interrupt)
    }
}

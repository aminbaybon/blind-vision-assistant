package ir.blindvision.assistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * On-device YOLO26n. The ONNX graph was exported from the project's yolo26n.pt
 * at 416 pixels. Output is [1, 84, 3549]: box center/size plus 80 class scores.
 */
class YoloDetector(context: Context) : AutoCloseable {
    var scoreThreshold: Float = 0.40f

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val pixels = IntArray(INPUT * INPUT)
    private val chw = FloatArray(3 * INPUT * INPUT)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val model = context.assets.open(MODEL).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            // Two threads leave room for speech and the camera on a 4-core budget phone.
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(model, options)
        inputName = session.inputNames.first()
    }

    fun detect(source: Bitmap): List<Seen> {
        val fitted = letterbox(source)
        fitted.bitmap.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        val plane = INPUT * INPUT
        for (index in pixels.indices) {
            val pixel = pixels[index]
            chw[index] = ((pixel shr 16) and 0xFF) / 255f
            chw[plane + index] = ((pixel shr 8) and 0xFF) / 255f
            chw[plane * 2 + index] = (pixel and 0xFF) / 255f
        }
        fitted.bitmap.recycle()
        val shape = longArrayOf(1, 3, INPUT.toLong(), INPUT.toLong())
        val input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(chw), shape)
        val results = session.run(Collections.singletonMap(inputName, input))
        try {
            val output = results[0] as OnnxTensor
            return decode(output.floatBuffer, source.width, source.height, fitted.scale, fitted.padX, fitted.padY)
        } finally {
            results.close()
            input.close()
        }
    }

    override fun close() {
        session.close()
    }

    private fun letterbox(source: Bitmap): Fitted {
        val scale = min(INPUT.toFloat() / source.width, INPUT.toFloat() / source.height)
        val width = (source.width * scale).roundToInt().coerceAtLeast(1)
        val height = (source.height * scale).roundToInt().coerceAtLeast(1)
        val padX = (INPUT - width) / 2f
        val padY = (INPUT - height) / 2f
        val out = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val dest = RectF(padX, padY, padX + width, padY + height)
        canvas.drawBitmap(source, null, dest, paint)
        return Fitted(out, scale, padX, padY)
    }

    private fun decode(
        buffer: FloatBuffer,
        imageWidth: Int,
        imageHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float,
    ): List<Seen> {
        buffer.rewind()
        val raw = ArrayList<Candidate>(32)
        val threshold = scoreThreshold
        for (index in 0 until PREDICTIONS) {
            var bestScore = threshold
            var bestClass = -1
            for (classId in 0 until CLASS_COUNT) {
                val score = buffer.get(valueIndex(4 + classId, index))
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classId
                }
            }
            if (bestClass < 0) continue
            val centerX = buffer.get(valueIndex(0, index))
            val centerY = buffer.get(valueIndex(1, index))
            val boxWidth = buffer.get(valueIndex(2, index))
            val boxHeight = buffer.get(valueIndex(3, index))
            val left = ((centerX - boxWidth / 2f) - padX) / scale
            val top = ((centerY - boxHeight / 2f) - padY) / scale
            val right = ((centerX + boxWidth / 2f) - padX) / scale
            val bottom = ((centerY + boxHeight / 2f) - padY) / scale
            val box = RectF(
                (left / imageWidth).coerceIn(0f, 1f),
                (top / imageHeight).coerceIn(0f, 1f),
                (right / imageWidth).coerceIn(0f, 1f),
                (bottom / imageHeight).coerceIn(0f, 1f),
            )
            if (box.width() < 0.01f || box.height() < 0.01f) continue
            raw += Candidate(bestClass, bestScore, box)
        }
        return nms(raw).map { candidate ->
            val center = (candidate.box.left + candidate.box.right) / 2f
            val side = when {
                center < 0.33f -> Side.LEFT
                center < 0.67f -> Side.CENTER
                else -> Side.RIGHT
            }
            Seen(COCO[candidate.classId], candidate.score, side, candidate.box, confirmed = false)
        }
    }

    private fun nms(items: List<Candidate>): List<Candidate> {
        val sorted = items.sortedByDescending { it.score }
        val kept = ArrayList<Candidate>(8)
        for (item in sorted) {
            val overlaps = kept.any { it.classId == item.classId && iou(it.box, item.box) > 0.50f }
            if (overlaps) continue
            kept += item
            if (kept.size == 8) break
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        if (union <= 0f) return 0f
        return intersection / union
    }

    private fun valueIndex(channel: Int, prediction: Int): Int = channel * PREDICTIONS + prediction

    private data class Fitted(val bitmap: Bitmap, val scale: Float, val padX: Float, val padY: Float)
    private data class Candidate(val classId: Int, val score: Float, val box: RectF)

    companion object {
        private const val MODEL = "yolo26n.onnx"
        private const val INPUT = 416
        private const val PREDICTIONS = 3549
        private const val CLASS_COUNT = 80

        private val COCO = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        )
    }
}

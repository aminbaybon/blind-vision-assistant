package ir.blindvision.assistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.sqrt

/**
 * MiDaS v2.1 small (`midas_small.onnx`).
 *
 * The ONNX graph already subtracts ImageNet mean (0.485, 0.456, 0.406) and
 * divides by std (0.229, 0.224, 0.225) before the first convolution. The
 * official `tf/run_onnx.py` path for this file is RGB in 0..1, stretched to
 * 256×256, then channel-first. It does not call NormalizeImage.
 *
 * Output is relative inverse depth: larger means closer, in network units.
 * It is not meters.
 */
class SceneDepth(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val pixels = IntArray(SIZE * SIZE)
    private val chw = FloatArray(3 * SIZE * SIZE)
    private val scaled = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(scaled)
    private val dest = RectF(0f, 0f, SIZE.toFloat(), SIZE.toFloat())
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val model = context.assets.open(MODEL).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(model, options)
        inputName = session.inputNames.first()
    }

    fun measure(source: Bitmap): DepthFrame? {
        canvas.drawBitmap(source, null, dest, paint)
        scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (index in pixels.indices) {
            val pixel = pixels[index]
            chw[index] = ((pixel shr 16) and 0xFF) / 255f
            chw[plane + index] = ((pixel shr 8) and 0xFF) / 255f
            chw[plane * 2 + index] = (pixel and 0xFF) / 255f
        }
        val shape = longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
        val input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(chw), shape)
        val results = session.run(Collections.singletonMap(inputName, input))
        try {
            val output = results[0] as OnnxTensor
            val values = FloatArray(output.floatBuffer.remaining())
            output.floatBuffer.get(values)
            val side = sqrt(values.size.toDouble()).toInt()
            if (side * side != values.size || side < 32) return null
            return DepthFrame(values, side)
        } finally {
            results.close()
            input.close()
        }
    }

    override fun close() {
        session.close()
        scaled.recycle()
    }

    companion object {
        private const val MODEL = "midas_small.onnx"
        private const val SIZE = 256
    }
}

/** Relative inverse-depth map. Index `y * side + x`. Larger values mean closer. */
class DepthFrame(
    val values: FloatArray,
    val side: Int,
)

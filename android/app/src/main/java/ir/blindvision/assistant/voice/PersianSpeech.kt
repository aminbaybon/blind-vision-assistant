package ir.blindvision.assistant.voice

import android.content.Context
import android.content.res.AssetManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.speech.tts.TextToSpeech
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * On-device Persian speech using the Piper gyro voice (fa_IR, medium)
 * through sherpa-onnx. The phone's own TTS is only a fallback if the
 * embedded model cannot be loaded.
 */
class PersianSpeech(
    context: Context,
    private val onReady: (persianVoice: Boolean) -> Unit,
) {
    private val appContext = context.applicationContext
    private val worker = Executors.newSingleThreadExecutor()
    private val lock = Any()
    private var token = 0
    private var ready = false
    private var closed = false
    private var pending: String? = null
    private var queued: String? = null
    private var draining = false
    private var embedded: OfflineTts? = null
    private var track: AudioTrack? = null
    private var system: TextToSpeech? = null
    var muted: Boolean = false
    var persianVoice: Boolean = false
        private set

    init {
        worker.execute { load() }
    }

    fun speak(text: String, interrupt: Boolean) {
        if (muted || text.isBlank() || closed) return
        if (interrupt) {
            bumpToken()
            stopTrack()
        } else if (currentToken() == 0) {
            bumpToken()
        }
        val startDrain = synchronized(lock) {
            if (!ready) {
                pending = text
                return
            }
            queued = text
            if (draining) return
            draining = true
            true
        }
        if (!startDrain) return
        worker.execute {
            while (!closed) {
                val next = synchronized(lock) {
                    val value = queued
                    queued = null
                    if (value == null) draining = false
                    value
                } ?: break
                say(next, currentToken())
            }
        }
    }

    fun stop() {
        bumpToken()
        stopTrack()
        system?.stop()
    }

    fun shutdown() {
        closed = true
        bumpToken()
        stopTrack()
        worker.shutdown()
        embedded?.release()
        embedded = null
        track?.release()
        track = null
        system?.shutdown()
        system = null
    }

    private fun load() {
        if (closed) return
        val loaded = try {
            val dataDir = copyEspeakData()
            val config = OfflineTtsConfig(
                model = OfflineTtsModelConfig(
                    vits = OfflineTtsVitsModelConfig(
                        model = "tts/fa_IR-gyro-medium.onnx",
                        tokens = "tts/tokens.txt",
                        dataDir = dataDir.absolutePath,
                    ),
                    numThreads = 2,
                    debug = false,
                    provider = "cpu",
                ),
            )
            OfflineTts(assetManager = appContext.assets, config = config)
        } catch (error: Throwable) {
            Log.e(TAG, "embedded Persian voice failed", error)
            null
        }
        if (closed) {
            loaded?.release()
            return
        }
        if (loaded != null) {
            embedded = loaded
            becomeReady(true)
            return
        }
        system = TextToSpeech(appContext) { status ->
            val engine = system
            val persian = status == TextToSpeech.SUCCESS && engine != null && selectPersian(engine)
            if (!persian) engine?.language = Locale.getDefault()
            becomeReady(persian)
        }
    }

    private fun becomeReady(hasPersianVoice: Boolean) {
        val queued = synchronized(lock) {
            if (closed) return
            persianVoice = hasPersianVoice
            ready = true
            val text = pending
            pending = null
            text
        }
        onReady(hasPersianVoice || embedded != null)
        if (!queued.isNullOrBlank()) say(queued, bumpToken())
    }

    private fun say(text: String, mine: Int) {
        if (closed || mine != currentToken()) return
        val engine = embedded
        if (engine != null) {
            val audio = try {
                engine.generate(text, 0, 0.95f)
            } catch (error: Throwable) {
                Log.e(TAG, "speech synthesis failed", error)
                return
            }
            if (closed || mine != currentToken()) return
            try {
                play(audio.samples, audio.sampleRate, mine)
            } catch (error: Throwable) {
                Log.e(TAG, "speech playback failed", error)
            }
            return
        }
        val fallback = system ?: return
        if (muted) return
        fallback.speak(text, TextToSpeech.QUEUE_FLUSH, null, "guide")
    }

    private fun play(samples: FloatArray, sampleRate: Int, mine: Int) {
        if (samples.isEmpty() || mine != currentToken()) return
        val pcm = ShortArray(samples.size)
        for (index in samples.indices) {
            val value = samples[index].coerceIn(-1f, 1f)
            pcm[index] = (value * 32767f).toInt().toShort()
        }
        val player = audioTrack(sampleRate)
        player.pause()
        player.flush()
        player.play()
        var offset = 0
        while (offset < pcm.size && mine == currentToken() && !closed) {
            val count = min(4096, pcm.size - offset)
            val written = player.write(pcm, offset, count)
            if (written <= 0) break
            offset += written
        }
        if (mine != currentToken() || closed) {
            player.pause()
            player.flush()
        }
    }

    private fun audioTrack(sampleRate: Int): AudioTrack {
        val existing = track
        if (existing != null && existing.sampleRate == sampleRate) return existing
        existing?.release()
        val minBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(sampleRate)
        val created = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(minBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track = created
        return created
    }

    private fun stopTrack() {
        val player = track ?: return
        try {
            player.pause()
            player.flush()
        } catch (_: IllegalStateException) {
        }
    }

    private fun copyEspeakData(): File {
        val dest = File(appContext.filesDir, "tts/espeak-ng-data")
        val marker = File(dest, ".ready")
        if (marker.isFile && File(dest, "lang/ira/fa").isFile) return dest
        if (dest.exists()) dest.deleteRecursively()
        copyAssetTree(appContext.assets, "tts/espeak-ng-data", dest)
        marker.writeText("fa_IR-gyro-medium")
        return dest
    }

    private fun selectPersian(tts: TextToSpeech): Boolean {
        val voice = try {
            tts.voices.orEmpty().firstOrNull {
                it.locale?.language.equals("fa", ignoreCase = true) &&
                    TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features
            }
        } catch (_: RuntimeException) {
            null
        }
        if (voice != null && tts.setVoice(voice) == TextToSpeech.SUCCESS) return true
        if (!missing(tts.setLanguage(Locale("fa", "IR")))) return true
        return !missing(tts.setLanguage(Locale("fa")))
    }

    private fun bumpToken(): Int = synchronized(lock) { ++token }

    private fun currentToken(): Int = synchronized(lock) { token }

    private fun missing(result: Int): Boolean {
        return result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED
    }

    companion object {
        private const val TAG = "BlindVision"

        private fun copyAssetTree(assets: AssetManager, path: String, dest: File) {
            val children = assets.list(path).orEmpty()
            if (children.isEmpty()) {
                dest.parentFile?.mkdirs()
                assets.open(path).use { input ->
                    dest.outputStream().use { output -> input.copyTo(output) }
                }
                return
            }
            dest.mkdirs()
            for (child in children) {
                copyAssetTree(assets, "$path/$child", File(dest, child))
            }
        }
    }
}

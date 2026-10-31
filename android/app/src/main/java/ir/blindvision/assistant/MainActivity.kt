package ir.blindvision.assistant

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.util.Size
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import ir.blindvision.assistant.databinding.ActivityMainBinding
import ir.blindvision.assistant.vision.DepthFrame
import ir.blindvision.assistant.vision.ImageFrames
import ir.blindvision.assistant.vision.SceneDepth
import ir.blindvision.assistant.vision.SceneSnapshot
import ir.blindvision.assistant.vision.SceneUnderstanding
import ir.blindvision.assistant.vision.SimpleTracker
import ir.blindvision.assistant.vision.YoloDetector
import ir.blindvision.assistant.voice.AlertEngine
import ir.blindvision.assistant.voice.PersianCopy
import ir.blindvision.assistant.voice.PersianSpeech
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var speech: PersianSpeech
    private val alertEngine = AlertEngine()
    private val tracker = SimpleTracker()
    private val understanding = SceneUnderstanding()
    private val latestScene = AtomicReference<SceneSnapshot?>(null)
    private var detector: YoloDetector? = null
    private var depthModel: SceneDepth? = null
    private var latestDepth: DepthFrame? = null
    private var lastDepthNs = 0L
    private var depthAtNs = 0L
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraRunning = false
    private var inferenceBusy = false
    private var lastInferenceNs = 0L

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showStatus(getString(R.string.status_camera_denied))
    }

    override fun attachBaseContext(newBase: android.content.Context) {
        super.attachBaseContext(LocaleHelper.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        analysisExecutor = Executors.newSingleThreadExecutor()
        speech = PersianSpeech(this) { persianVoice ->
            runOnUiThread {
                if (!persianVoice) {
                    binding.statusText.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
                    showStatus(getString(R.string.tts_missing))
                } else if (Preferences.disclaimerSeen(this)) {
                    speech.speak(getString(R.string.voice_ready), interrupt = true)
                }
            }
        }
        binding.cameraButton.setOnClickListener {
            vibrate(40)
            if (cameraRunning) stopCamera() else requestCamera()
        }
        binding.voiceButton.setOnClickListener {
            vibrate(40)
            speech.muted = !speech.muted
            if (speech.muted) speech.stop()
            showVoiceState()
        }
        binding.describeButton.setOnClickListener {
            vibrate(40)
            val text = PersianCopy.describe(latestScene.get())
            showStatus(text)
            speech.speak(text, interrupt = true)
        }
        binding.settingsButton.setOnClickListener {
            vibrate(40)
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        loadModel()
        maybeShowDisclaimer()
        requestCamera()
    }

    override fun onResume() {
        super.onResume()
        detector?.scoreThreshold = Preferences.confidence(this)
        alertEngine.cooldownMs = Preferences.cooldownSeconds(this) * 1000L
    }

    override fun onDestroy() {
        stopCamera()
        analysisExecutor.shutdown()
        analysisExecutor.awaitTermination(3, TimeUnit.SECONDS)
        detector?.close()
        depthModel?.close()
        speech.shutdown()
        super.onDestroy()
    }

    private fun loadModel() {
        analysisExecutor.execute {
            try {
                val loaded = YoloDetector(applicationContext)
                loaded.scoreThreshold = Preferences.confidence(this)
                detector = loaded
                try {
                    depthModel = SceneDepth(applicationContext)
                } catch (error: Throwable) {
                    Log.e(TAG, "depth model failed", error)
                    depthModel = null
                }
                runOnUiThread { showStatus(getString(R.string.status_ready)) }
            } catch (error: Throwable) {
                Log.e(TAG, "model load failed", error)
                runOnUiThread { showStatus(getString(R.string.model_failed)) }
            }
        }
    }

    private fun maybeShowDisclaimer() {
        if (Preferences.disclaimerSeen(this)) return
        AlertDialog.Builder(this)
            .setMessage(R.string.disclaimer)
            .setCancelable(false)
            .setPositiveButton(R.string.disclaimer_ok) { _, _ ->
                Preferences.markDisclaimerSeen(this)
                vibrate(40)
                speech.speak(getString(R.string.disclaimer), interrupt = true)
            }
            .show()
    }

    private fun requestCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = binding.preview.surfaceProvider
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetResolution(Size(640, 480))
                    .build()
                analysis.setAnalyzer(analysisExecutor, ::analyze)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                cameraRunning = true
                showCameraState()
            } catch (error: Exception) {
                Log.e(TAG, "camera start failed", error)
                showStatus(getString(R.string.status_camera_error))
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraRunning = false
        if (::binding.isInitialized) showCameraState()
    }

    private fun analyze(image: ImageProxy) {
        val now = System.nanoTime()
        if (inferenceBusy || now - lastInferenceNs < 250_000_000L) {
            image.close()
            return
        }
        val active = detector
        if (active == null) {
            image.close()
            return
        }
        inferenceBusy = true
        lastInferenceNs = now
        val started = System.nanoTime()
        try {
            val bitmap = ImageFrames.toUprightBitmap(image)
            if (bitmap != null) {
                val seen = tracker.update(active.detect(bitmap))
                val nowNs = System.nanoTime()
                if (nowNs - lastDepthNs > 800_000_000L) {
                    lastDepthNs = nowNs
                    try {
                        val measured = depthModel?.measure(bitmap)
                        if (measured != null) {
                            latestDepth = measured
                            depthAtNs = nowNs
                        }
                    } catch (error: Throwable) {
                        Log.e(TAG, "depth failed", error)
                    }
                }
                val fresh = latestDepth != null && nowNs - depthAtNs < 1_500_000_000L
                val snapshot = understanding.interpret(if (fresh) latestDepth else null, seen)
                latestScene.set(snapshot)
                bitmap.recycle()
                val alert = alertEngine.next(snapshot, System.currentTimeMillis())
                val elapsedMs = (System.nanoTime() - started) / 1_000_000L
                if (elapsedMs > 0 && alert != null) Log.i(TAG, "inference ${elapsedMs}ms alert=${alert.text}")
                if (alert != null) {
                    runOnUiThread {
                        showStatus(alert.text)
                        speech.speak(alert.text, alert.interrupt)
                        if (alert.priority >= 90) vibrate(80)
                    }
                }
            }
        } catch (error: Throwable) {
            Log.e(TAG, "frame failed", error)
        } finally {
            image.close()
            inferenceBusy = false
        }
    }

    private fun showStatus(text: String) {
        binding.statusText.text = text
    }

    private fun showCameraState() {
        val running = cameraRunning
        binding.cameraButton.text = getString(if (running) R.string.camera_live_label else R.string.camera_idle_label)
        binding.cameraButton.contentDescription = getString(if (running) R.string.stop_camera else R.string.start_camera)
    }

    private fun showVoiceState() {
        val muted = speech.muted
        binding.voiceButton.text = getString(if (muted) R.string.voice_muted_label else R.string.voice_live_label)
        binding.voiceButton.contentDescription = getString(if (muted) R.string.voice_off else R.string.voice_on)
    }

    private fun vibrate(durationMs: Long) {
        val vibrator = getSystemService(Vibrator::class.java) ?: return
        vibrator.vibrate(VibrationEffect.createOneShot(durationMs, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    companion object {
        private const val TAG = "BlindVision"
    }
}

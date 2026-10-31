package ir.blindvision.assistant

import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.widget.SeekBar
import androidx.appcompat.app.AppCompatActivity
import ir.blindvision.assistant.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySettingsBinding

    override fun attachBaseContext(newBase: android.content.Context) {
        super.attachBaseContext(LocaleHelper.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val confidenceProgress = ((Preferences.confidence(this) - 0.30f) * 100f).toInt().coerceIn(0, 40)
        val cooldownProgress = (Preferences.cooldownSeconds(this) - 4).coerceIn(0, 8)
        binding.confidenceBar.progress = confidenceProgress
        binding.cooldownBar.progress = cooldownProgress
        showValues()

        val listener = object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) = showValues()
            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        }
        binding.confidenceBar.setOnSeekBarChangeListener(listener)
        binding.cooldownBar.setOnSeekBarChangeListener(listener)
        binding.saveButton.setOnClickListener {
            Preferences.save(this, confidence(), cooldown())
            vibrate()
            binding.saveButton.announceForAccessibility(getString(R.string.settings_saved))
            finish()
        }
    }

    private fun showValues() {
        binding.confidenceValue.text = String.format("%.2f", confidence())
        binding.cooldownValue.text = cooldown().toString()
    }

    private fun confidence(): Float = 0.30f + binding.confidenceBar.progress / 100f

    private fun cooldown(): Int = 4 + binding.cooldownBar.progress

    private fun vibrate() {
        val vibrator = getSystemService(Vibrator::class.java) ?: return
        vibrator.vibrate(VibrationEffect.createOneShot(40, VibrationEffect.DEFAULT_AMPLITUDE))
    }
}

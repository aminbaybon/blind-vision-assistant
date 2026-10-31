# Debug builds do not minify. Keep ONNX Runtime classes if release minify is enabled later.
-keep class ai.onnxruntime.** { *; }
-keep class com.k2fsa.sherpa.onnx.** { *; }

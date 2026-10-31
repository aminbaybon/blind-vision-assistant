# Blind vision assistant

An experimental Android app for people with low vision. This repository is about **five to six hours** of work: a Python video prototype, then an app that runs on the phone, offline, and speaks Persian.

Current app version: **0.4.1**

This app does not replace a white cane or any other mobility aid. It does not report distance in meters, and it does not say that a path is safe.

## What those hours produced

The app uses the rear camera and does three things on the device:

1. It detects familiar objects (person, car, chair, and the rest of the COCO classes).
2. It estimates depth only relative to the same frame: something looks nearer or farther. That value is not meters.
3. If something in front, or to the left or right, looks close, it speaks one conservative Persian sentence. For example: «مانعی در قسمت روبه‌روی تصویر دیده می‌شود. فاصله دقیق مانع مشخص نیست.» (“An obstacle is visible in the front of the image. The exact distance is unknown.”) If nothing clear is detected, it stays silent. Silence does not mean the way is clear.

The interface is Persian, the typeface is Vazir, and the default voice is the Persian voice embedded in the app.

The experimental debug build is on [Releases](https://github.com/aminbaybon/blind-vision-assistant/releases). It is not a store-signed release. Uninstall any older build before installing this one.

## Technical details

| Part | Choice | Details |
| --- | --- | --- |
| Platform | Android | Kotlin, minSdk 26 (Android 8), target/compile 35, app version 0.4.1 |
| Camera | CameraX 1.4.1 | Rear RGB camera only. Preview target 640×480 |
| Runtime | CPU | ONNX Runtime Android 1.31.1. No GPU, NPU, ARCore, LiDAR, or depth sensor required |
| Network | None | Inference is offline. The internet permission is removed from the merged manifest |
| Objects | YOLO26 Nano | ONNX, FP32, about 9.3 MB, input `images` shaped `[1,3,416,416]`, output `[1,84,3549]`, 80 COCO classes |
| Depth | MiDaS v2.1 Small | ONNX, FP32, about 64 MB, 256×256 input, 256×256 map. A larger value means nearer in that frame, not meters |
| Speech | Piper `fa_IR-gyro-medium` | About 60 MB, through sherpa-onnx 1.13.8. Phonemizer data from espeak-ng |
| Scheduling | One thread | A frame is dropped if the previous pass is still running, or if less than 250 ms has passed. Depth runs at most once every 800 ms. The same sentence is not repeated before 18 seconds |
| Native ABIs | ARM | `arm64-v8a` and `armeabi-v7a`. x86 emulator libraries are not in the APK |
| Debug APK size | 269,605,717 bytes | Most of that is the three models and the native libraries |

The Python sample (`main.py`) is separate from the app. It runs YOLO26n on a video and writes English descriptions. Its meter figure is a ground-plane geometric estimate. The Android app does not speak that sentence.

## What still needs work

This version shows a technical path. It is not a validated product.

- MiDaS depth on this phone has not been checked against a tape measure. Until there is real data, saying “two meters” would be wrong, so the app does not say it.
- Speed and memory have not been measured on a 3 GB phone. The design target is a mid-range phone, but device FPS and RAM are not recorded.
- The models are FP32. INT8 is worth doing only after the output is checked on real images.
- Wall, curb, sidewalk, and walkable path are not separate classes. The “front obstacle” sentence means a near surface in the center of the image, not the name of an object.
- Stair detection is not in this version. It was not reliable from an ordinary RGB camera without labeled data, so it was removed.
- The app does not give a turn (“turn left”) or call a path safe, and it should not, until free space is actually measured.
- A field test with low-vision users, larger touch targets, and a signed release build are still undone.
- The current APK is a debug build and is not installed from a store.

## Build the app

JDK 17, the Android SDK, and Gradle 8.11.1 are required. From the `android` directory:

```bat
gradle assembleDebug
```

Output:

`android/app/build/outputs/apk/debug/app-debug.apk`

## Python sample

Python 3.10 or newer:

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

The input video is not included in this repository.

## Component licenses

This sample code is experimental. Each model weight has its own license:

| Component | License |
| --- | --- |
| YOLO26n (Ultralytics) | AGPL-3.0 |
| MiDaS v2.1 small | MIT |
| sherpa-onnx | Apache-2.0 |
| Piper voice `fa_IR-gyro-medium` | License of the k2-fsa release |
| espeak-ng-data | GPL-3.0 |
| Vazir font | SIL Open Font License |

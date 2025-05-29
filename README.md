# Navitzer: Visual Navigation Assistant

**Navitzer** is an AI-powered real-time visual navigation assistant built with YOLOE segmentation. It uses a webcam to detect and identify objects, helping users (especially those with visual impairments) understand their surroundings through audio-based scene descriptions.

## 🚀 Features

- Real-time object detection and segmentation using YOLOE
- Webcam input with live visualization
- Audio output describing detected objects
- Easy integration with other sensors for assistive navigation (e.g., GPS, ultrasonic sensors)

## 🧠 Model Used

- **YOLOE Segmentation Model**: `yoloe-11l-seg-pf.pt`
  - Pretrained on a segmentation-capable dataset
  - Capable of detecting and segmenting multiple object classes in real-time

## 🖥️ Requirements

- Python 3.8+
- PyTorch (compatible with your GPU/CPU)
- OpenCV (`cv2`)
- `ultralytics` or your custom YOLOE inference package
- `torchvision`
- (Optional) `pyttsx3` or `gTTS` for speech output

Install requirements via:

```bash
pip install -r requirements.txt

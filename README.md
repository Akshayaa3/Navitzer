# Navitzer: Visual Navigation Assistance System

**Navitzer** is an AI-based real-time visual navigation assistance system designed to enhance environmental awareness for individuals with visual impairments and other spatial challenges. It employs object detection, semantic segmentation, and depth estimation to analyze the user's surroundings and deliver auditory feedback. 

The system has evolved from an initial YOLOv8 segmentation model to a more optimized and hardware-efficient solution utilizing YOLOE segmentation in conjunction with MiDaS for monocular depth estimation. MiDaS currently serves as a temporary substitute for the Intel RealSense D421 stereo camera, which will be incorporated in future iterations for improved depth perception.

---

## Key Features

- Real-time object detection and segmentation using YOLOE
- Monocular depth estimation via MiDaS (temporary RealSense replacement)
- Live video streaming through a webcam
- Spoken scene interpretation using text-to-speech
- Modular architecture suitable for integration with additional sensors (GPS, ultrasonic, IMU)

---

## System Architecture and Model Evaluation

### Phase 1: Initial Implementation (YOLOv8 Segmentation)

- **Model**: YOLOv8 with segmentation head  
- **Datasets**: COCO + Open Images V7  
- **Challenges**:
  - Suboptimal segmentation under real-world noise
  - High inference latency on resource-constrained platforms
  - No built-in support for depth estimation

### Phase 2: Depth Estimation Trials

Multiple monocular depth estimation models were evaluated for clarity, inference speed, and compatibility with segmentation output:

| Model         | Evaluation Summary                                     |
|---------------|--------------------------------------------------------|
| **MonoLITE**    | Lightweight; insufficient detail in complex scenes    |
| **DPT-Hybrid**  | High precision; computationally intensive             |
| **MiDaS v3.1 Small** | ✅ Chosen for balance between clarity and efficiency |

> **Note**: MiDaS is a provisional solution. **Intel RealSense D421** will be integrated for accurate stereo depth sensing in the final deployment.

### Phase 3: Current Architecture (YOLOE + MiDaS)

- **Detection Model**: `yoloe-11l-seg-pf.pt`
- **Depth Estimation**: `midas_v3_small_256.pt`
- **Input Source**: Standard USB webcam
- **Output**: Detected objects with approximate depth, converted to audio feedback

---

## Requirements

- Python 3.8 or higher
- PyTorch (GPU/CPU compatible)
- OpenCV
- `timm` (for MiDaS)
- Optional: `pyttsx3` or `gTTS` for audio synthesis

### Installation

```bash
pip install -r requirements.txt

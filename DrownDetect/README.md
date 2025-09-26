# DrownDetect
This code is a prototype demonstrating drowning detection by identifying rapid or random movements of a person. You can enhance it by adding to detecting only the person’s head if only the head is consistently detected, they may be considered drowning. Another approach is monitoring head angle, since drowning individuals often tilt their heads up to breathe.

## Description
DrownDetect is a real-time drowning detection system that leverages YOLOv8 for object detection and OpenCV for video processing. The system analyzes video feeds to identify individuals exhibiting signs of drowning, such as unusual movement patterns or only head visibility, and provides visual alerts to signal potential drowning incidents.

## Features
- **Real-Time Detection:** Processes video frames in real-time to detect potential drowning cases.
- **Head-Only Detection:** Identifies drowning even when only the head of a person is visible.
- **Visual Alerts:** Displays bounding boxes around detected individuals and alerts when drowning is detected.

## Installation

### Prerequisites
- **Python 3.8.10 or higher**
- **pip** (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/dihhzy/DrownDetect.git
cd DrownDetect

Create a Virtual Environment (Optional but Recommended)
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python DrownDetect.py
```
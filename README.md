# OceanONE-ML-platform
Here is the deployed link: https://oceanone-ml-platform.streamlit.app/

OceanONE-ML-platform is an end-to-end machine learning ecosystem designed for marine robotics and underwater data analysis. Inspired by the OceanOne humanoid diver, this platform provides the tools necessary to process underwater imagery, train haptic-feedback models, and deploy autonomous navigation algorithms for deep-sea exploration.
 What the Project Does

The platform streamlines the workflow for researchers and developers working with underwater datasets. It bridges the gap between raw oceanic sensor data and actionable machine learning models.

    Data Augmentation: Specialized filters for underwater image restoration (dehazing, color correction).

    Model Zoo: Pre-trained models for marine species identification and wreckage detection.

    Haptic Integration: Modules for processing tactile feedback data from robotic manipulators.

    Deployment: Lightweight inference engines optimized for edge devices used in ROVs and AUVs.

 Why the Project is Useful

Underwater environments present unique challenges: light absorption, backscatter, and high pressure. This platform is useful because it:

    Reduces Development Time: Provides a standardized pipeline for marine-specific ML tasks.

    Improves Accuracy: Includes domain-specific preprocessing that standard ML libraries lack.

    Enhances Autonomy: Enables robots to "see" and "feel" better in murky, deep-sea conditions.

 How Users Can Get Started
Prerequisites

    Python 3.8 or higher

    CUDA-enabled GPU (recommended for training)
Installation

    Clone the repository:
    Bash

    git clone https://github.com/akshayfouzder2005/OceanONE-ML-platform.git
    cd OceanONE-ML-platform

    Install dependencies:
    Bash

    pip install -r requirements.txt

Usage Example

To run a basic inference on an underwater image:

from oceanone.models import MarineDetector
from oceanone.utils import color_correct

# Load image
img = color_correct("samples/underwater_wreck.jpg")

# Initialize model
model = MarineDetector(weights="pretrained/yolov8_marine.pt")

# Perform detection
results = model.predict(img)
results.show()

🤝 Where Users Can Get Help

    Documentation: Detailed guides can be found in the /docs folder.

    Issue Tracker: Report bugs or request features via GitHub Issues.

    Discussions: Join our community on GitHub Discussions for Q&A.

👥 Who Maintains and Contributes

    Maintainer: Akshay Fouzder

  Some screenshot of the project: 
  <img width="1848" height="1022" alt="image" src="https://github.com/user-attachments/assets/0b0c1f22-3432-4105-9982-8712db3f76ba" />
  <img width="1284" height="758" alt="image" src="https://github.com/user-attachments/assets/1e4c8119-a98b-4a50-991d-6ffdbe95f690" />
  <img width="1364" height="841" alt="image" src="https://github.com/user-attachments/assets/d00dadb1-e6fc-4b16-9b99-9c7555c83f55" />
  <img width="1819" height="983" alt="image" src="https://github.com/user-attachments/assets/a503b04b-1107-44d7-b53a-8a5f52b2abde" />
  <img width="1464" height="926" alt="image" src="https://github.com/user-attachments/assets/1e1261e4-d8b5-4a75-93a0-320ab8278382" />
  <img width="1403" height="865" alt="image" src="https://github.com/user-attachments/assets/39d018c1-4583-4a26-af30-dd281410db96" />







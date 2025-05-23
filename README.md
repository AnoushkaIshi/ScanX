# ScanX

A cutting-edge AI-powered medical imaging assistant designed to help radiologists and healthcare professionals by answering visual questions on radiology images and generating detailed, accurate medical reports with ease.


## Tech Stack

**Frontend:**  Streamlit (for interactive UI)

**Backend:**  Python (for AI/ML model integration)

**AI/ML Models:** BLIP, Hugging Face NLP models, Large Language Models (LLMs)

**Others:**  Git for version control


## Features

- Visual Question Answering (VQA) on radiology images using the BLIP model  
- Automated medical report generation powered by Hugging Face NLP models and Large Language Models (LLMs)  
- Interactive and user-friendly Streamlit interface for easy image upload and question input  
- High accuracy in medical image understanding (83.7%) for reliable insights  
- Real-time response for quick radiology assessments  
- Modular architecture enabling easy integration of additional AI models and features  


## Lessons Learned

Building ScanX was a highly educational experience that involved working at the intersection of cutting-edge AI research and practical software development. Throughout the project, I learned several important lessons and faced numerous challenges that helped me grow both technically and professionally.

### Key Learnings:

1. **Integrating Vision-Language Models in Real-World Applications:**  
   Implementing the BLIP model for Visual Question Answering in the radiology domain was eye-opening. I gained deep insights into how vision and language modalities can be combined effectively, especially in sensitive fields like healthcare where accuracy is critical. Understanding model inputs, fine-tuning on domain-specific data, and evaluating performance were crucial parts of this process.

2. **Fine-tuning NLP Models for Medical Report Generation:**  
   Working with Hugging Face NLP transformers and Large Language Models tailored for medical text highlighted the importance of domain adaptation. Medical language is nuanced and complex, so general NLP models had to be carefully fine-tuned on clinical datasets to generate relevant and coherent reports.

3. **Developing Interactive and User-Friendly Interfaces:**  
   Creating the Streamlit front end taught me the significance of usability in AI applications. It was important to balance technical complexity with simplicity, ensuring that radiologists and medical professionals could easily upload images, ask questions, and receive insights without a steep learning curve.

4. **Real-Time Data Handling and Synchronization:**  
   Although Streamlit is primarily for prototyping, handling real-time interactions like image uploads and question processing challenged me to optimize data flow and asynchronous operations efficiently.

5. **Collaboration and Version Control Best Practices:**  
   Managing the project code, collaborating with team members, and maintaining clean commit histories improved my proficiency with Git and GitHub workflows. I also learned how to document code and projects professionally for easy onboarding and knowledge sharing.

### Challenges Faced and Solutions:

- **Challenge:** Understanding and adapting complex pre-trained AI models (BLIP and Hugging Face transformers) to work cohesively in the radiology domain.  
  **Solution:** I invested time in researching model architectures, reviewing scientific papers, and experimenting with different fine-tuning strategies on publicly available medical datasets until the models achieved satisfactory performance.

- **Challenge:** Ensuring medical report generation was clinically relevant and free of jargon or irrelevant content.  
  **Solution:** Collaborated with domain experts for feedback and iteratively refined training data and model parameters to enhance output quality.

- **Challenge:** Integrating AI backend services with the Streamlit frontend smoothly and handling latency during model inference.  
  **Solution:** Optimized model serving by using efficient batch processing, caching results where appropriate, and providing clear loading indicators in the UI to improve user experience.

---

Overall, this project strengthened my skills in AI model development, software engineering, and cross-disciplinary collaboration. It also gave me a strong appreciation for the challenges and responsibilities involved in deploying AI tools in healthcare, where accuracy and reliability can directly impact patient outcomes.
---
## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AnoushkaIshi/ScanX.git
   cd ScanX
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

5. **Access the app in your browser at:**
  ```bash
http://localhost:8501

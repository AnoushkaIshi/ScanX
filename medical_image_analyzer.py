import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Medical Image Analyzer",
    page_icon="🏥",
    layout="wide"
)
st.title("Medical Image Analysis Tool")
st.markdown("""
This application uses AI to analyze medical images and provide explanations.
1. Enter patient information
2. Upload a medical image
3. Ask specific questions about the image
4. Get AI-powered analysis and diagnosis
""")

with st.sidebar:
    st.header("About")
    st.markdown("This app uses ScanX model for medical visual question answering and Hugging Face LLM for detailed medical explanations.")
    
    hf_token = HuggingFace Token  
    
    st.warning("Note: This is a prototype tool.")

if 'image' not in st.session_state:
    st.session_state.image = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'blip_model' not in st.session_state:
    st.session_state.blip_model = None
if 'vqa_result' not in st.session_state:
    st.session_state.vqa_result = None
if 'llm_explanation' not in st.session_state:
    st.session_state.llm_explanation = None
if 'custom_question' not in st.session_state:
    st.session_state.custom_question = ""
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'standard_questions' not in st.session_state:
    st.session_state.standard_questions = [
        "What abnormalities can be seen in this image?",
        "Is there any pathology visible?",
        "What might be the diagnosis based on this image?",
        "Are there any concerning features in this image?",
        "What is the main finding in this image?"
    ]

@st.cache_resource
def load_blip_model(token):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_auth_token=token)
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", use_auth_token=token)
    return processor, model

def perform_vqa(image, question, processor, model):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def get_hf_explanation(vqa_result, image_type, patient_info, token):
    import requests
    
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
    headers = {"Authorization": f"Bearer {token}"}
    
    patient_context = ""
    if patient_info:
        patient_context = f"""
        Patient Information:
        - Age: {patient_info.get('age', 'Not provided')}
        - Gender: {patient_info.get('gender', 'Not provided')}
        - Clinical History: {patient_info.get('clinical_history', 'Not provided')}
        - Chief Complaint: {patient_info.get('chief_complaint', 'Not provided')}
        - Current Medications: {patient_info.get('medications', 'Not provided')}
        """
    
    prompt = f"""
    Based on AI analysis of a {image_type} medical image, the following observations were made:
    
    {vqa_result}
    
    {patient_context}
    
    As a medical expert, provide a detailed explanation of these findings, potential diagnoses, 
    recommended follow-up tests, and educational information about the identified conditions.
    """
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]["generated_text"]
            else:
                return str(result)
        else:
            return f"Error from Hugging Face API: {response.status_code}\n\nUsing rule-based analysis instead."
    except Exception as e:
        return f"Error generating explanation: {str(e)}\n\nUsing rule-based analysis instead."

def get_rule_based_analysis(vqa_results, patient_info):
  
    findings = []
    diagnosis = "Unknown"
    
    for question, answer in vqa_results.items():
        if "diagnosis" in question.lower():
            diagnosis = answer
        if "abnormalities" in question.lower() or "finding" in question.lower():
            findings.append(answer)
    
   
    patient_age = patient_info.get('age', 'Not provided')
    patient_gender = patient_info.get('gender', 'Not provided')
    clinical_history = patient_info.get('clinical_history', 'Not provided')
    
    age_related_factors = ""
    if patient_age != 'Not provided':
        try:
            age = int(patient_age)
            if age > 65:
                age_related_factors = "Given the patient's advanced age, conditions such as degenerative changes and age-related cardiovascular diseases should be considered."
            elif age < 18:
                age_related_factors = "Given the patient's young age, congenital or developmental conditions should be considered."
        except:
            pass
    
    analysis = f"""
    ## Medical Image Analysis
    
    ### Patient Information
    - **Patient ID:** {patient_info.get('id', 'Not provided')}
    - **Age:** {patient_age}
    - **Gender:** {patient_gender}
    - **Clinical History:** {clinical_history}
    - **Chief Complaint:** {patient_info.get('chief_complaint', 'Not provided')}
    - **Current Medications:** {patient_info.get('medications', 'Not provided')}
    
    ### AI-Detected Findings:
    - Primary observation: {", ".join(findings) if findings else "No specific findings detected"}
    - Suggested diagnosis: {diagnosis}
    
    ### Potential Clinical Significance:
    
    {age_related_factors}
    
    Based on the AI analysis, the following conditions might be considered:
    
    1. **{diagnosis.title() if diagnosis != "Unknown" else "Cardiac Abnormality"}**
       - Common symptoms include chest pain, shortness of breath, and fatigue
       - May be associated with structural or functional changes in the heart
       
    2. **Differential Diagnoses to Consider:**
       - Coronary artery disease
       - Cardiomyopathy
       - Valvular heart disease
       - Congestive heart failure
       - Arrhythmias
    
    ### Recommended Follow-up:
    
    1. **Additional Imaging:**
       - Echocardiogram for detailed cardiac structure and function
       - ECG/EKG for electrical activity
       - Cardiac stress test to evaluate exercise capacity
       
    2. **Laboratory Tests:**
       - Complete blood count
       - Cardiac enzymes (troponin, CK-MB)
       - BNP (B-type natriuretic peptide)
       - Lipid profile
       
    3. **Specialist Consultation:**
       - Cardiology referral for comprehensive evaluation
    
    ### Important Limitations:
    
    This analysis is generated by an AI system with limited information. A proper diagnosis requires:
    - Complete patient history
    - Physical examination
    - Multiple diagnostic tests
    - Clinical expertise

    """
    
    return analysis


st.header("Patient Information")
with st.expander("Enter Patient Details", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_id = st.text_input("Patient ID/MRN", key="patient_id")
        patient_age = st.number_input("Age", min_value=0, max_value=120, step=1, key="patient_age")
        patient_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, step=0.1, key="patient_weight")
    
    with col2:
        patient_name = st.text_input("Patient Name", key="patient_name")
        patient_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other", "Prefer not to say"], key="patient_gender")
        patient_height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, step=0.1, key="patient_height")
    
    with col3:
        patient_dob = st.date_input("Date of Birth", key="patient_dob")
        study_date = st.date_input("Study Date", datetime.now(), key="study_date")
        referring_physician = st.text_input("Referring Physician", key="referring_physician")

    col1, col2 = st.columns(2)
    
    with col1:
        chief_complaint = st.text_area("Chief Complaint", key="chief_complaint", height=100)
        medications = st.text_area("Current Medications", key="medications", height=100)
    
    with col2:
        clinical_history = st.text_area("Clinical History", key="clinical_history", height=100)
        allergies = st.text_area("Allergies", key="allergies", height=100)
    
    if st.button("Save Patient Information"):
        st.session_state.patient_info = {
            'id': patient_id,
            'name': patient_name,
            'age': patient_age,
            'dob': patient_dob.strftime("%Y-%m-%d") if patient_dob else None,
            'gender': patient_gender if patient_gender != "Select" else None,
            'weight': patient_weight,
            'height': patient_height,
            'study_date': study_date.strftime("%Y-%m-%d") if study_date else None,
            'referring_physician': referring_physician,
            'chief_complaint': chief_complaint,
            'clinical_history': clinical_history,
            'medications': medications,
            'allergies': allergies
        }
        st.success("Patient information saved!")

st.header("Upload Medical Image")
upload_col, preview_col = st.columns([1, 1])

with upload_col:
    uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png", "dcm"])
    image_type = st.selectbox(
        "Image type",
        ["X-ray", "MRI", "CT scan", "Ultrasound", "Microscopy", "Other medical image"]
    )
    
    modality_details = st.text_input("Modality Details (e.g., PA view, T2-weighted)", 
                                     placeholder="Enter specific details about the imaging modality")
    
    anatomical_region = st.selectbox(
        "Anatomical Region",
        ["Brain", "Chest", "Abdomen", "Pelvis", "Spine", "Extremity", "Cardiac", "Other"]
    )
    
    if uploaded_file is not None:
        try:
         
            image = Image.open(uploaded_file)
            st.session_state.image = image
            
            if st.session_state.processor is None or st.session_state.blip_model is None:
                with st.spinner("Loading BLIP model..."):
                    try:
                        st.session_state.processor, st.session_state.blip_model = load_blip_model(hf_token)
                    except Exception as e:
                        st.error(f"Error loading BLIP model: {str(e)}")
                        st.info("Attempting to load model without token...")
                        st.session_state.processor, st.session_state.blip_model = load_blip_model(None)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with preview_col:
    if st.session_state.image is not None:
        st.image(st.session_state.image, caption="Uploaded Medical Image", use_container_width=True)
        
        st.markdown("#### Image Metadata")
        metadata_cols = st.columns(2)
        with metadata_cols[0]:
            st.markdown(f"**Type:** {image_type}")
            st.markdown(f"**Region:** {anatomical_region}")
        with metadata_cols[1]:
            st.markdown(f"**Details:** {modality_details}")
            st.markdown(f"**Study Date:** {st.session_state.patient_info.get('study_date', 'Not provided')}")


if st.session_state.image is not None:
    st.header("Medical Image Analysis")
    
    question_tab, result_tab = st.tabs(["Ask Questions", "Analysis Results"])
    
    with question_tab:
        st.subheader("Select Questions to Ask")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Standard Medical Questions")
            selected_questions = []
            for q in st.session_state.standard_questions:
                if st.checkbox(q, key=f"checkbox_{q}"):
                    selected_questions.append(q)
        
        with col2:
            st.markdown("#### Custom Question")
            custom_question = st.text_input("Enter your custom medical question:", key="custom_question_input")
            if custom_question:
                st.session_state.custom_question = custom_question
        

        st.markdown("#### Quick Analysis")
        if st.button("Quick Analyze (Basic Diagnosis)"):
            selected_questions = ["What might be the diagnosis based on this image?"]
        
        analyze_button = st.button("Analyze Image")
        if analyze_button:
            if not selected_questions and not st.session_state.custom_question:
                st.warning("Please select at least one question to ask.")
            else:
                all_questions = selected_questions + ([st.session_state.custom_question] if st.session_state.custom_question else [])
                
                with st.spinner("Analyzing image..."):
                   
                    combined_results = {}
                    for question in all_questions:
                        answer = perform_vqa(st.session_state.image, question, st.session_state.processor, st.session_state.blip_model)
                        combined_results[question] = answer
                    
                    st.session_state.vqa_result = combined_results
                    
                   
                    combined_vqa_str = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in combined_results.items()])
                    
                    hf_explanation = get_hf_explanation(
                        combined_vqa_str, 
                        image_type, 
                        st.session_state.patient_info,
                        hf_token
                    )
                    
                    if "Error" in hf_explanation or len(hf_explanation.strip()) < 50:
                        st.session_state.llm_explanation = get_rule_based_analysis(
                            combined_results,
                            st.session_state.patient_info
                        )
                    else:
                        st.session_state.llm_explanation = hf_explanation
                
                st.success("Analysis complete! Check the Results tab.")
    
    with result_tab:
        if st.session_state.vqa_result:
         
            if st.session_state.patient_info:
                patient_name = st.session_state.patient_info.get('name', 'Not provided')
                patient_id = st.session_state.patient_info.get('id', 'Not provided')
                patient_age = st.session_state.patient_info.get('age', 'Not provided')
                patient_gender = st.session_state.patient_info.get('gender', 'Not provided')
                
                st.markdown(f"### Patient: {patient_name} (ID: {patient_id})")
                st.markdown(f"**Age:** {patient_age} | **Gender:** {patient_gender} | **Study Date:** {st.session_state.patient_info.get('study_date', 'Not provided')}")
                st.markdown("---")
            
            st.subheader("Visual Question Answering Results")
            for question, answer in st.session_state.vqa_result.items():
                st.markdown(f"**Q: {question}**")
                st.markdown(f"A: {answer}")
            
            st.markdown("---")
            
            st.subheader("AI Medical Explanation")
            if st.session_state.llm_explanation:
                st.markdown(st.session_state.llm_explanation)
                
              
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
           
                patient_info_section = ""
                if st.session_state.patient_info:
                    patient_info_section = f"""
                    # Medical Image Analysis Report
                    
                    ## Patient Information
                    
                    - **Patient Name:** {st.session_state.patient_info.get('name', 'Not provided')}
                    - **Patient ID:** {st.session_state.patient_info.get('id', 'Not provided')}
                    - **Date of Birth:** {st.session_state.patient_info.get('dob', 'Not provided')}
                    - **Age:** {st.session_state.patient_info.get('age', 'Not provided')}
                    - **Gender:** {st.session_state.patient_info.get('gender', 'Not provided')}
                    - **Weight:** {st.session_state.patient_info.get('weight', 'Not provided')} kg
                    - **Height:** {st.session_state.patient_info.get('height', 'Not provided')} cm
                    - **Referring Physician:** {st.session_state.patient_info.get('referring_physician', 'Not provided')}
                    
                    ### Clinical Information
                    
                    - **Chief Complaint:** {st.session_state.patient_info.get('chief_complaint', 'Not provided')}
                    - **Clinical History:** {st.session_state.patient_info.get('clinical_history', 'Not provided')}
                    - **Current Medications:** {st.session_state.patient_info.get('medications', 'Not provided')}
                    - **Allergies:** {st.session_state.patient_info.get('allergies', 'Not provided')}
                    
                    ## Study Details
                    
                    - **Study Date:** {st.session_state.patient_info.get('study_date', 'Not provided')}
                    - **Modality:** {image_type}
                    - **Anatomical Region:** {anatomical_region}
                    - **Modality Details:** {modality_details}
                    """
                else:
                    patient_info_section = "# Medical Image Analysis Report\n\n*No patient information provided*"
                
                report_content = f"""
                {patient_info_section}
                
                ## AI Analysis Results
                
                ### Visual Question Answering Results
                
                {chr(10).join([f"**Q: {q}**\nA: {a}\n" for q, a in st.session_state.vqa_result.items()])}
                
                ### AI Medical Explanation
                
                {st.session_state.llm_explanation}
                
                ## Report Information
                
                - **Generated on:** {current_date}
                - **Analysis Method:** AI-assisted image analysis (BLIP + LLM)
                
                """
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Report (Markdown)",
                        report_content,
                        file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    
                    pdf_content = report_content.replace("###", "**").replace("##", "**")
                    st.download_button(
                        "Download Report (TXT)",
                        pdf_content,
                        file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
else:
    st.info("Please upload a medical image to begin analysis.")

st.markdown("---")
st.markdown("""
            
<div style='text-align: center; color: gray; font-size: small;'>
<p>         DISCLAIMER: This application is for  research purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding any medical conditions or concerns.</p>
</div>
""", unsafe_allow_html=True)
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
from pdf2image import convert_from_path
import base64
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from langchain.chains import RetrievalQA 
import google.generativeai as genai
# from paddleocr import PaddleOCR
import torch
from PIL import Image
import io
import sys
sys.path.append("D:\CLIP-main")  # Replace this with the actual path to the CLIP directory

import clip
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=r"C:\Users\vidhyashree.v\Downloads\en_PP-OCRv3_det_infer",
#                 rec_model_dir=r"C:\Users\vidhyashree.v\Downloads\en_PP-OCRv4_rec_infer",
#                 cls_model_dir=r"C:\Users\vidhyashree.v\Downloads\ch_ppocr_mobile_v2.0_cls_infer")
poppler_path = r"C:\Users\vidhyashree.v\Downloads\Release-23.08.0-0\poppler-23.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_path


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set Streamlit page configuration
st.set_page_config(page_title="Chat PDF", page_icon="🤖", layout="wide")

# Custom CSS to reduce whitespace at the top of the page
st.markdown("""
  <style>
     .block-container {     
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
    }

    .stHeadingContainer {      
       margin-top: -18px;
    }
    .row-widget.stButton{    
       margin-top: -2px;
       margin-bottom: -5px; 
    } 
  </style>
""", unsafe_allow_html=True)

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Convert PDF to images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to a temporary location
            uploaded_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Convert the saved file to images
            images = convert_from_path(uploaded_file_path, output_folder=temp_dir)
            
            # Convert images to bytes and extract text
            text = ""
            for image in images:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_file:
                    image.save(img_file, 'PNG')
                    img_file.seek(0)
                    img_bytes = img_file.read()
                    # print(img_bytes)

                    # Convert bytes to text using decode()
                    text = base64.b64encode(img_bytes).decode()
                    # print(img_bytes)
                    # Extract text using OCR
                    # image_text = image_bytes_to_text(img_bytes)
                    # text += image_text + "\n"
                    # print(text)
            return text
    else:
        raise FileNotFoundError("No file uploaded")
    


# Chunking OCR text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Vectorizing the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the vector store locally

    return vector_store




# def image_bytes_to_text(img_bytes):
#     # Convert bytes to PIL Image
#     image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     # Preprocess the image
#     image_input = preprocess(image).unsqueeze(0).to(device)
#     # Encode the image using CLIP
#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#     # Convert image features to text using CLIP's tokenizer
#     text = clip.tokenize(["a photo"])  # Placeholder text
#     logits_per_image, _ = model(image_input, text)
#     predicted_labels = logits_per_image.argmax(1)
#     return clip.decode_batch(predicted_labels)[0]




def display_pdf(pdf_doc):
    with open(f"temp_files/{pdf_doc.name}", "wb") as f:
        f.write(pdf_doc.getbuffer())
    st.markdown(f"**{pdf_doc.name}**")
    pdf_path = f"temp_files/{pdf_doc.name}"
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def user_input(user_question, image_data):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
    retriever = image_data.as_retriever(search_type='similarity', search_kwargs={'k': 50}) 
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)

    answer = chain.run(user_question) 
    print(answer)
    return answer 

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

vector_store = None  # Initialize vector_store as None globally

def main():
    global vector_store  # Access the global vector_store variable

    if st.button("New Chat"):
        st.session_state.messages = []

    st.subheader("Chat with PDF💁")
    initialize_chat()

    with st.sidebar:
        st.subheader("PDF-View📖")
        pdf_doc = st.file_uploader("Upload PDF", type="pdf")

        ## Custom CSS to reduce whitespace in the upload pdf 
        css = '''
        <style>
            [data-testid='stFileUploader'] {
                width: max-content;
            }
            [data-testid='stFileUploader'] section {
                padding: 0;
                float: left;
            }
            [data-testid='stFileUploader'] section > input + div {
                display: none;
            }
            [data-testid='stFileUploader'] section + div {
                float: right;
                padding-top: 0;
            }

        </style>
        '''

        st.markdown(css, unsafe_allow_html=True)

        initialize_chat()

        if pdf_doc is not None:
            display_pdf(pdf_doc)

            try:
                image_data = input_image_details(pdf_doc)
                print(image_data)
                text_chunks = get_text_chunks(image_data)
                vector_store = get_vector_store(text_chunks)  # Update vector_store only if PDF is uploaded successfully
                if vector_store is not None:
                    st.success("Done")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


        # input_prompt = """
        #     You are an expert in understanding pdf. We will upload pdf
        #     and you will have to answer any questions based on the uploaded pdf only.
        #     """

    user_question = st.chat_input("What is up?")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        # image_data = input_image_details(pdf_doc) if pdf_doc else None
        # print(image_data)
        response = user_input(user_question, vector_store)
        st.session_state.messages.append({"role": "assistant", "content": response})
    

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                edited_question = st.text_input("", value=message["content"], label_visibility="collapsed")
                if edited_question != message["content"] and edited_question.strip() != "":
                    # User has edited the question
                    st.session_state.messages[i]["content"] = edited_question
                    answer = user_input(edited_question, vector_store)
                    # Find the index of the existing answer for the edited question
                    existing_answer_index = next((index for index, msg in enumerate(st.session_state.messages[i+1:]) if msg["role"] == "assistant"), None)
                    if existing_answer_index is not None:
                        # Update the existing answer for the edited question
                        st.session_state.messages[i+1+existing_answer_index]["content"] = answer
                    else:
                        # Add the new answer for the edited question
                        st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.markdown(message["content"])

if __name__ == "__main__":
    main()




















# #code for hilighting the pdf -trial

# # import streamlit as st
# # import os
# # import base64
# # import tempfile
# # import numpy as np
# # import cv2
# # import pytesseract
# # from PIL import Image, ImageEnhance, ImageFilter
# # from pdf2image import convert_from_path
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.vectorstores import FAISS
# # from langchain.chains import RetrievalQA
# # import fitz
# # from dotenv import load_dotenv
# # import google.generativeai as genai

# # poppler_path = r"C:\Users\vidhyashree.v\Downloads\Release-23.08.0-0\poppler-23.08.0\Library\bin"
# # os.environ["PATH"] += os.pathsep + poppler_path
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vidhyashree.v\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Set Streamlit page configuration
# # st.set_page_config(page_title="Chat PDF", page_icon="🤖", layout="wide")

# # # Custom CSS to reduce whitespace at the top of the page
# # st.markdown("""
# #   <style>
# #      .block-container {     
# #     padding-top: 1rem;
# #     padding-bottom: 0rem;
# #     margin-top: 1rem;
# #     }

# #     .stHeadingContainer {      
# #        margin-top: -18px;
# #     }
# #     .row-widget.stButton{    
# #        margin-top: -2px;
# #        margin-bottom: -5px; 
# #     } 
# #   </style>
# # """, unsafe_allow_html=True)

# # # Function for fixing orientation of image
# # def fix_orientation(image):
# #     try:
# #         image_array = np.array(image)
# #         img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
# #         info = pytesseract.image_to_osd(img_gray, output_type=pytesseract.Output.DICT, config='--psm 0 -c min_characters_to_try=150')
# #         h, w = image_array.shape[:2]
# #         center = (h // 2, w // 2)
# #         rotation_angle = -info['rotate']
# #         M = cv2.getRotationMatrix2D(center, angle=rotation_angle, scale=1.0)
# #         rotated_image = cv2.warpAffine(image_array, M, (w, h))
# #         rotated_image_pil = Image.fromarray(rotated_image)
# #         return rotated_image_pil
# #     except:
# #         return None

# # # Function for pre-processing image
# # def img_preprocessing(img):
# #     try:
# #         img = Image.fromarray(img)
# #     except:
# #         img = img
# #     img1 = img.filter(ImageFilter.GaussianBlur(radius=0.8))
# #     numpy_img = np.array(img1)
# #     opencv_Img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)
# #     invert_img = cv2.bitwise_not(opencv_Img)
# #     level_1_upscale = cv2.pyrUp(invert_img)
# #     enhancer = ImageEnhance.Sharpness(Image.fromarray(level_1_upscale))
# #     image = enhancer.enhance(2.8)
# #     return image

# # # Orientation fixing and Preprocessing, Ocr
# # def get_pdf_text(pdf_doc):
# #     text = ""
# #     bboxes = []
    
# #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
# #         tmp_file.write(pdf_doc.read())
# #         tmp_file_path = tmp_file.name
    
# #     images = convert_from_path(tmp_file_path)
    
# #     for i, image in enumerate(images):
# #         fixed_image = fix_orientation(image)
        
# #         if fixed_image:
# #             preprocessed_image = img_preprocessing(fixed_image)
# #             page_text = pytesseract.image_to_string(preprocessed_image, lang='fra+eng+chi_sim')
# #             page_data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)
            
# #             for j, word in enumerate(page_data['text']):
# #                 if word.strip():
# #                     x, y, width, height = page_data['left'][j], page_data['top'][j], page_data['width'][j], page_data['height'][j]
# #                     bbox = (x, y, x + width, y + height)
# #                     bboxes.append(bbox)
            
# #             text += page_text + "\n"
    
# #     os.unlink(tmp_file_path)
    
# #     return text, bboxes


# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # # Vectorizing the text chunks
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")  # Save the vector store locally

# #     return vector_store

# # #Takes user question as input and do similarity search using model and langchain to give answer
# # def user_input(user_question, vector_store):
# #     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
# #     retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
# #     # print(retriever) 
# #     chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)

# #     answer = chain.run(user_question) 
# #     print(answer)
# #     return answer 
    
# # # Function to display pdf
# # def display_pdf(pdf_doc):
# #     with open(f"temp_files/{pdf_doc.name}", "wb") as f:
# #         f.write(pdf_doc.getbuffer())
# #     st.markdown(f"**{pdf_doc.name}**")
# #     pdf_path = f"temp_files/{pdf_doc.name}"
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="900" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # Function to initialize new chat
# # def initialize_chat():
# #     if "messages" not in st.session_state:
# #         st.session_state.messages = []

# # def highlight_text_in_pdf(pdf_file, output_file, bboxes):
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
# #         tmp_file.write(pdf_file.read())
# #         tmp_file_path = tmp_file.name

# #     # Add debug statements to check if the temporary file is empty
# #     if os.path.getsize(tmp_file_path) == 0:
# #         st.error("Temporary PDF file is empty!")
# #         return

# #     # Continue with the highlighting process
# #     pdf_document = fitz.open(tmp_file_path)
    
# #     for page_num, page in enumerate(pdf_document):
# #         for bbox in bboxes[page_num]:
# #             highlight = page.add_rect_annot(bbox)
# #             highlight.set_colors({"stroke": (1, 1, 0)})  # Yellow color
    
# #     pdf_document.save(output_file)
# #     pdf_document.close()

# #     os.unlink(tmp_file_path)




# # def display_highlighted_pdf(pdf_doc, highlighted_pdf_path):
# #     with open(highlighted_pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="900" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # vector_store = None  # Initialize vector_store as None globally

# # def main():
# #     global vector_store
    
# #     if st.button("New Chat"):
# #         st.session_state.messages = []

# #     st.subheader("Chat with PDF💁")
# #     initialize_chat()

# #     with st.sidebar:
# #         st.subheader("PDF-View📖")
# #         pdf_doc = st.file_uploader("Upload PDF", type="pdf")

# #         # Custom CSS to reduce whitespace in the upload pdf 
# #         css = '''
# #         <style>
# #             [data-testid='stFileUploader'] {
# #                 width: max-content;
# #             }
# #             [data-testid='stFileUploader'] section {
# #                 padding: 0;
# #                 float: left;
# #             }
# #             [data-testid='stFileUploader'] section > input + div {
# #                 display: none;
# #             }
# #             [data-testid='stFileUploader'] section + div {
# #                 float: right;
# #                 padding-top: 0;
# #             }
# #         </style>
# #         '''

# #         st.markdown(css, unsafe_allow_html=True)

# #         initialize_chat()

# #         if pdf_doc is not None:
# #             display_pdf(pdf_doc)

# #             try:
# #                 raw_text, bboxes = get_pdf_text(pdf_doc)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 vector_store = get_vector_store(text_chunks)  # Update vector_store only if PDF is uploaded successfully
# #                 if vector_store is not None:
# #                     st.success("Done")
# #             except Exception as e:
# #                 st.error(f"An error occurred: {str(e)}")

# #     user_question = st.chat_input("What is up?")

# #     if user_question:
# #         st.session_state.messages.append({"role": "user", "content": user_question})
# #         answer = user_input(user_question, vector_store)
# #         st.session_state.messages.append({"role": "assistant", "content": answer})

# #         # Highlight the text in the PDF
# #         highlight_pdf_path = "highlighted_pdf.pdf"
# #         highlight_text_in_pdf(pdf_doc, highlight_pdf_path, bboxes)
        
# #         # Display the highlighted PDF
# #         display_highlighted_pdf(pdf_doc, highlight_pdf_path)

# #     for i, message in enumerate(st.session_state.messages):
# #         with st.chat_message(message["role"]):
# #             if message["role"] == "user":
# #                 edited_question = st.text_input("", value=message["content"], label_visibility="collapsed")
# #                 if edited_question != message["content"] and edited_question.strip() != "":
# #                     # User has edited the question
# #                     st.session_state.messages[i]["content"] = edited_question
# #                     answer = user_input(edited_question, vector_store)
# #                     # Find the index of the existing answer for the edited question
# #                     existing_answer_index = next((index for index, msg in enumerate(st.session_state.messages[i+1:]) if msg["role"] == "assistant"), None)
# #                     if existing_answer_index is not None:
# #                         # Update the existing answer for the edited question
# #                         st.session_state.messages[i+1+existing_answer_index]["content"] = answer
# #                     else:
# #                         # Add the new answer for the edited question
# #                         st.session_state.messages.append({"role": "assistant", "content": answer})


# # if __name__ == "__main__":
# #     main()












































# # # import streamlit as st
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # import os
# # # import base64
# # # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # # import google.generativeai as genai
# # # from langchain.vectorstores import FAISS
# # # from langchain_google_genai import ChatGoogleGenerativeAI
# # # from langchain.chains.question_answering import load_qa_chain
# # # from langchain.prompts import PromptTemplate
# # # from dotenv import load_dotenv
# # # import pytesseract
# # # from PIL import Image
# # # from pdf2image import convert_from_path
# # # import tempfile
# # # import pandas as pd

# # # # Add Poppler directory to the PATH environment variable
# # # poppler_path = r"C:\Users\vidhyashree.v\Downloads\Release-23.08.0-0\poppler-23.08.0\Library\bin"
# # # os.environ["PATH"] += os.pathsep + poppler_path
# # # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vidhyashree.v\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # # # Load environment variables
# # # load_dotenv()
# # # os.getenv("GOOGLE_API_KEY")
# # # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # # Set Streamlit page configuration
# # # st.set_page_config(page_title="Chat PDF", page_icon="🤖", layout="wide")




# # # # # Custom CSS to reduce whitespace at the top of the page
# # # st.markdown("""
# # #   <style>
# # #      .block-container {     
# # #     padding-top: 1rem;
# # #     padding-bottom: 0rem;
# # #     margin-top: 1rem;
# # #     }

# # #     .stHeadingContainer {      
# # #        margin-top: -18px;
# # #     }
# # #     .row-widget.stButton{    
# # #        margin-top: -2px;
# # #        margin-bottom: -5px; 
# # #     } 
# # #   </style>
# # # """, unsafe_allow_html=True)

# # # # block-container- for th chat window
# # # # stHeadingContainer - pdf viewer text
# # # # .row-widget.stButton - Process pdf button


# # # def get_pdf_text(pdf_doc):
# # #     text = ""
# # #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
# # #         tmp_file.write(pdf_doc.read())
# # #         tmp_file_path = tmp_file.name
    
# # #     images = convert_from_path(tmp_file_path)
# # #     for image in images:
# # #         page_text = pytesseract.image_to_string(image)
# # #         text += page_text + "\n"

# # #     os.unlink(tmp_file_path)
# # #     print(text)
# # #     return text

# # # def get_text_chunks(text):
# # #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# # #     chunks = text_splitter.split_text(text)
# # #     return chunks

# # # def get_vector_store(text_chunks):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# # #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# # #     vector_store.save_local("faiss_index")

# # # def get_conversational_chain():
# # #     prompt_template = """
# # #     Answer the question or summarize the topic in a point-wise manner, Make sure to provide all relevant details. 
# # #     If the answer is not available in the provided context, simply state, "Answer is not available in the context." Please avoid providing incorrect information.\n\n
# # #     Context:\n {context}?\n
# # #     Question: \n{question}\n

# # #     Answer:
# # #     """

# # #     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# # #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# # #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # #     return chain

# # # def user_input(user_question):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# # #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# # #     docs = new_db.similarity_search(user_question)
# # #     chain = get_conversational_chain()
# # #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# # #     return response["output_text"]

# # # def display_pdf(pdf_doc):
# # #     with open(f"temp_files/{pdf_doc.name}", "wb") as f:
# # #         f.write(pdf_doc.getbuffer())
# # #     st.markdown(f"**{pdf_doc.name}**")
# # #     pdf_path = f"temp_files/{pdf_doc.name}"
# # #     with open(pdf_path, "rb") as f:
# # #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# # #     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="900" type="application/pdf"></iframe>'
# # #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # def initialize_chat():
# # #     if "messages" not in st.session_state:
# # #         st.session_state.messages = []

# # # def main():
# # #     if st.button("New Chat"):
# # #         # Server.get_current()._reloader.reload()
# # #         st.session_state.messages = []
# # #         # st.markdown("<script>window.location.reload(true);</script>", unsafe_allow_html=True)

# # #     st.subheader("Chat with PDF💁") 
# # #     initialize_chat()

# # #     with st.sidebar:
# # #         st.subheader("PDF-View📖")
# # #         pdf_doc = st.file_uploader("Upload PDF", type="pdf")
# # #         css = '''
# # #         <style>
# # #             [data-testid='stFileUploader'] {
# # #                 width: max-content;
# # #             }
# # #             [data-testid='stFileUploader'] section {
# # #                 padding: 0;
# # #                 float: left;
# # #             }
# # #             [data-testid='stFileUploader'] section > input + div {
# # #                 display: none;
# # #             }
# # #             [data-testid='stFileUploader'] section + div {
# # #                 float: right;
# # #                 padding-top: 0;
# # #             }

# # #         </style>
# # #         '''

# # #         st.markdown(css, unsafe_allow_html=True)
        
# # #         initialize_chat()

# # #         if pdf_doc:
# # #             display_pdf(pdf_doc)

# # #         if st.button("Process PDF"):
# # #             with st.spinner("Processing PDF..."):
# # #                 raw_text = get_pdf_text(pdf_doc)
# # #                 text_chunks = get_text_chunks(raw_text)
# # #                 get_vector_store(text_chunks)
# # #                 st.success("Done")


# # #     user_question = st.chat_input("What is up?")

# # #     if user_question:
# # #         st.session_state.messages.append({"role": "user", "content": user_question})
# # #         answer = user_input(user_question)
# # #         st.session_state.messages.append({"role": "assistant", "content": answer})

# # #     for message in st.session_state.messages:
# # #         with st.chat_message(message["role"]):
# # #             st.markdown(message["content"])

# # # if __name__ == "__main__":
# # #     main()



































# # # #code for tessaract
# # # import streamlit as st
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # import os
# # # import base64
# # # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # # import google.generativeai as genai
# # # from langchain.vectorstores import FAISS
# # # from langchain_google_genai import ChatGoogleGenerativeAI
# # # from langchain.chains.question_answering import load_qa_chain
# # # from langchain.prompts import PromptTemplate
# # # from dotenv import load_dotenv
# # # import pytesseract
# # # from PIL import Image
# # # from pdf2image import convert_from_path
# # # import tempfile

# # # # Add Poppler directory to the PATH environment variable
# # # poppler_path = r"C:\Users\vidhyashree.v\Downloads\Release-23.08.0-0\poppler-23.08.0\Library\bin"
# # # os.environ["PATH"] += os.pathsep + poppler_path
# # # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vidhyashree.v\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # # # Load environment variables
# # # load_dotenv()
# # # os.getenv("GOOGLE_API_KEY")
# # # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# # # # Function to get text from PDFs using OCR
# # # def get_pdf_text(pdf_doc):
# # #     text = ""
# # #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
# # #         tmp_file.write(pdf_doc.read())
# # #         tmp_file_path = tmp_file.name
    
# # #     images = convert_from_path(tmp_file_path)
# # #     for image in images:
# # #         # Perform OCR on each page image and concatenate the extracted text
# # #         page_text = pytesseract.image_to_string(image)
# # #         text += page_text + "\n"

# # #     # Delete temporary file
# # #     os.unlink(tmp_file_path)
# # #     print(text)
# # #     return text

# # # # Function to split text into chunks
# # # def get_text_chunks(text):
# # #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# # #     chunks = text_splitter.split_text(text)
# # #     return chunks


# # # # Function to create vector store
# # # def get_vector_store(text_chunks):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# # #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# # #     vector_store.save_local("faiss_index")


# # # # Function to create conversational chain
# # # def get_conversational_chain():
# # #     prompt_template = """
# # #     Answer the question or summarize the topic in a point-wise manner, Make sure to provide all relevant details. 
# # #     If the answer is not available in the provided context, simply state, "Answer is not available in the context." Please avoid providing incorrect information.\n\n
# # #     Context:\n {context}?\n
# # #     Question: \n{question}\n

# # #     Answer:
# # #     """

# # #     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# # #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# # #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # #     return chain


# # # # Function for user input and getting response
# # # def user_input(user_question):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# # #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# # #     docs = new_db.similarity_search(user_question)
# # #     chain = get_conversational_chain()
# # #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# # #     return response["output_text"]


# # # # Function to display uploaded PDF
# # # def display_pdf(pdf_doc):
# # #     with open(f"temp_files/{pdf_doc.name}", "wb") as f:
# # #         f.write(pdf_doc.getbuffer())
# # #     st.markdown(f"**{pdf_doc.name}**")
# # #     pdf_path = f"temp_files/{pdf_doc.name}"
# # #     with open(pdf_path, "rb") as f:
# # #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# # #     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="900" type="application/pdf"></iframe>'
# # #     st.markdown(pdf_display, unsafe_allow_html=True)


# # # # Function to initialize chat session
# # # def initialize_chat():
# # #     if "messages" not in st.session_state:
# # #         st.session_state.messages = []


# # # # Main function
# # # def main():
# # #     st.set_page_config(page_title="Chat PDF", page_icon="🤖", layout="wide")
# # #     st.header("Chat with PDF💁")
# # #       # Sidebar for Q&A
# # #     with st.sidebar:
# # #         st.header("PDF-View📖")
# # #         pdf_doc = st.file_uploader("Upload PDF", type="pdf")
# # #         initialize_chat()

# # #         if st.button("Process PDF"):
# # #             with st.spinner("Processing PDF..."):
# # #                 raw_text = get_pdf_text(pdf_doc)
# # #                 text_chunks = get_text_chunks(raw_text)
# # #                 get_vector_store(text_chunks)
# # #                 st.success("Done")

# # #         if pdf_doc:
# # #             display_pdf(pdf_doc)

# # #     # # Sidebar for Q&A
# # #     # with st.sidebar:
# # #     #     st.header("Chat with PDF💁")

# # #     # Section for asking questions and getting answers
# # #     user_question = st.chat_input("What is up?")
# # #     if user_question:
# # #         st.session_state.messages.append({"role": "user", "content": user_question})
# # #         answer = user_input(user_question)
# # #         st.session_state.messages.append({"role": "assistant", "content": answer})

# # #     #Display the ChatHistory
# # #         for message in st.session_state.messages:
# # #             with st.chat_message(message["role"]):
# # #                 st.markdown(message["content"])


# # # if __name__ == "__main__":
# # #     main()

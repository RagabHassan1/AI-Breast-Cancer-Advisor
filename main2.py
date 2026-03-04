from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import gradio as gr
import time
import traceback
from image import load_models, preprocess_image, get_prediction, print_results
import cv2
import numpy as np
import tempfile

# Initialize the language model
model = OllamaLLM(model="qwen3:0.6b")

# Load the breast cancer classification model
model_path = 'breast_xception (1).h5'
try:
    classification_model = load_models(model_path)
    print("Image classification model loaded successfully")
except Exception as e:
    print(f"Error loading classification model: {str(e)}")
    classification_model = None

template = """
أنت خبير في سرطان الثدي تقدم نصائح حول الفحص الذاتي والوقاية من المرض.

لديك مصدرين رئيسيين للمعلومات:
1. كيفية الفحص الذاتي (self_examination.txt)
2. كيفية الوقاية من سرطان الثدي (breast_prevention.txt)

عند الإجابة على الأسئلة:
- ركز فقط على معلومات الفحص الذاتي من ملف self_examination.txt
- ثم اذكر طرق الوقاية من ملف breast_prevention.txt
- لا تذكر أي معلومات أخرى غير متعلقة بهذين الموضوعين

هذه المعلومات المتاحة: {context}
السؤال من المستخدم باللغة العربية: {question}

ابدأ إجابتك بتحية ودية ثم اتبع الهيكل التالي:
1. الفحص الذاتي 
2. الوقاية من المرض

أجب بإيجاز وبلغة عربية واضحة فقط.
لا تظهر أي عملية تفكير أو كلمات إنجليزية.
لا تترجم أي شيء إلى الإنجليزية.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def respond(question):
    try:
        print(f"\n[Received Question]: {question}")
        
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        print(f"[Found {len(docs)} relevant documents]")
        
        if not docs:
            error_msg = "❗ لم يتم العثور على معلومات ذات صلة في الوثائق."
            print(f"[Response]: {error_msg}")
            return error_msg
        
        context = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"context": context, "question": question})
        
        print(f"[Response]:\n{result}")
        return result
    except Exception as e:
        error_msg = f"حدث خطأ: {str(e)}"
        print(f"\n[Error]: {str(e)}")
        traceback.print_exc()
        return error_msg

def classify_image(image):
    if classification_model is None:
        return "❗ نموذج التصنيف غير متوفر حالياً"
    
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image_path = tmp_file.name
            cv2.imwrite(image_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        
        # Preprocess and classify the image
        processed_image = preprocess_image(image_path)
        prediction = get_prediction(processed_image)
        result = print_results(prediction)
        
        # Translate the result to Arabic
        result_ar = result.replace("Normal breast - no tumor detected", "طبيعي - لا يوجد ورم")
        result_ar = result_ar.replace("Benign tumor detected", "تم اكتشاف ورم حميد")
        result_ar = result_ar.replace("Malignant tumor detected", "تم اكتشاف ورم خبيث")
        result_ar = result_ar.replace("confidence", "ثقة")
        
        return result_ar
    except Exception as e:
        error_msg = f"حدث خطأ أثناء معالجة الصورة: {str(e)}"
        print(f"\n[Image Error]: {str(e)}")
        traceback.print_exc()
        return error_msg

css = """
body {
    direction: rtl;
}
.gradio-container {
    direction: rtl;
}
.output-text {
    text-align: right;
    direction: rtl;
}
"""

# Create Gradio interface with both text and image inputs
with gr.Blocks() as demo:
    gr.Markdown("## خبير الفحص الذاتي والوقاية من سرطان الثدي")
    gr.Markdown("اسأل عن الفحص الذاتي أو الوقاية من سرطان الثدي أو قم بتحميل صورة للفحص")
    
    with gr.Tab("الأسئلة النصية"):
        text_input = gr.Textbox(label="اسأل سؤالك", placeholder="كيف أفحص ثديي؟")
        text_output = gr.Textbox(label="إجابة الاستشاري")
        text_button = gr.Button("إرسال")
        
        examples = gr.Examples(
            examples=[
                ["كيف أفحص ثديي ذاتياً؟"],
                ["كيف يمكنني اكتشاف أي تغيرات في ثديي؟"]
            ],
            inputs=[text_input]
        )
    
    with gr.Tab("فحص الصورة"):
        image_input = gr.Image(label="قم بتحميل صورة للفحص")
        image_output = gr.Textbox(label="نتيجة الفحص")
        image_button = gr.Button("فحص الصورة")
    
    text_button.click(respond, inputs=text_input, outputs=text_output)
    image_button.click(classify_image, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    print("Starting breast cancer advisor...")
    print("The application is now running. Questions and responses will be printed below:\n")
    demo.launch(css=css)

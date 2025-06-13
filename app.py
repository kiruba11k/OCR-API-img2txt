import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import torch

from transformers import AutoProcessor, AutoModelForTokenClassification

# Load LayoutLMv3 fine-tuned model
processor = AutoProcessor.from_pretrained("Kiruba11/layoutlmv3-resume-ner2")
model = AutoModelForTokenClassification.from_pretrained("Kiruba11/layoutlmv3-resume-ner2")
model.eval().to("cpu")

OCR_API_KEY = st.secrets["OCR_API_KEY"]

def ocr_space_parse(image: Image.Image, uploaded_file_name="image.jpg"):
    # Ensure filename has valid image extension
    if not uploaded_file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        uploaded_file_name += ".jpg"  # Default fallback

    # Convert image to buffer (JPEG-encoded)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # Ensure it is actually JPEG format
    buffer.seek(0)

    # Manually set file tuple (filename, fileobj, content_type)
    files = {
        'filename': (uploaded_file_name, buffer, 'image/jpeg')
    }

    data = {
        'apikey': OCR_API_KEY,
        'isOverlayRequired': True,
        'language': 'eng',
        'scale': True,
    }

    response = requests.post(
        'https://api.ocr.space/parse/image',
        files=files,
        data=data
    )

    result = response.json()

    if "ParsedResults" not in result:
        st.error("❌ OCR failed. Full response:")
        st.json(result)
        return [], []

    parsed_text = []
    parsed_boxes = []

    try:
        lines = result['ParsedResults'][0]['TextOverlay']['Lines']
        for line in lines:
            for word_data in line['Words']:
                word = word_data['WordText']
                left = word_data['Left']
                top = word_data['Top']
                width = word_data['Width']
                height = word_data['Height']
                parsed_text.append(word)
                parsed_boxes.append([left, top, left + width, top + height])
    except Exception as e:
        st.error(f"OCR structure parsing error: {e}")
        return [], []

    return parsed_text, parsed_boxes

def normalize_boxes(boxes, width, height):
    return [
        [
            int(1000 * (x0 / width)),
            int(1000 * (y0 / height)),
            int(1000 * (x1 / width)),
            int(1000 * (y1 / height)),
        ]
        for x0, y0, x1, y1 in boxes
    ]

def predict_entities(image):
    words, boxes = ocr_space_parse(image)
    if not words:
        return []

    norm_boxes = normalize_boxes(boxes, *image.size)
    encodings = processor(image, words=words, boxes=norm_boxes, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = processor.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    results = []
    for token, label in zip(tokens, labels):
        if token not in processor.tokenizer.all_special_tokens and label != "O":
            results.append((token.replace("▁", ""), label))
    return results

# Streamlit UI
st.title(" OCR Extractor ")
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=file.name)

    words, boxes = ocr_space_parse(image, uploaded_file_name=file.name)
    st.write(words)
    
        entities = predict_entities(image, file.name)
        grouped = {}
        for word, label in entities:
            key = label[2:] if '-' in label else label
            grouped[key] = grouped.get(key, "") + word + " "
        grouped["File"] = file.name
        all_data.append(grouped)

    df = pd.DataFrame(all_data)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "layoutlmv3_entities.csv", "text/csv")
else:
    st.info("Please upload one or more images to begin.")

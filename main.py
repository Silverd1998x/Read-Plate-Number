import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import easyocr
import re



def detect_plate(image):
    model = YOLO("best_model/best_detect_plate.pt")
    results = model.predict(image)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
        # Iterate through the results and draw bounding boxes
    for result in results:                                         # iterate results
        boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
        for box in boxes:                                          # iterate boxes
            r = box.xyxy[0].astype(int)                            # get corner points as int
            cropped_image = image_np[r[1]:r[3], r[0]:r[2]]
            image_detect = cv2.rectangle(image_np, r[:2], r[2:], (0, 255, 0), 2)
            

    new_size = (200, 200)
    resized_image = cv2.resize(cropped_image, new_size)
    return image_detect, resized_image


def crop_image(image):
    image_1 = [10,200,100,200]
    image_2 = [10,200,10,100]
    cropped_1 = image[image_1[2]:image_1[3], image_1[0]:image_1[1]]
    cropped_2 = image[image_2[2]:image_2[3], image_2[0]:image_2[1]]
    return cropped_1, cropped_2


def remove_special_characters(input_string):
    # Define a regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9]'  # This pattern allows alphanumeric characters and spaces
    # Use re.sub to replace the matched special characters with an empty string
    result_string = re.sub(pattern, "", input_string)
    return result_string


def read_text_in_plate(image):
    cropped_1,cropped_2 = crop_image(image)
    # Tạo một đối tượng EasyOCR với ngôn ngữ bạn muốn sử dụng
    reader = easyocr.Reader(['en'])  # 'en' cho tiếng Anh, bạn có thể chọn ngôn ngữ khác tùy thuộc vào nhu cầu
    # Đọc biển số từ hình ảnh
    result_1 = reader.readtext(cropped_2)
    result_2 = reader.readtext(cropped_1)
    for detection in result_1:
        text_1 = detection[1]
    for detection in result_2:
        text_2 = detection[1]
    text_plate=f"{remove_special_characters(text_1)}-{remove_special_characters(text_2)}"
    return text_plate


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title(" DEMO - ĐỌC BIỂN SỐ XE")

    uploaded_file = st.file_uploader("Chọn hình ảnh ........", type="jpg")
    if uploaded_file is not None:

        # Load the image
        image = Image.open(uploaded_file)
        detect_image, results = detect_plate(image)
        text_plate = read_text_in_plate(results)

        col1, col2, col3 = st.columns(3)
        with col1:       
            st.header("Hình ảnh đầu vào ...")
            st.image(image)

        with col2:
            st.header("Nhận diện biển số xe ...")
            st.image(detect_image)
            st.header("Tách biển số xe ...")
            st.image(results)

        with col3:
            st.header("Kết quả đầu ra ...")
            st.text(f"Biển số xe: {text_plate}")
            st.header(text_plate)

if __name__ == "__main__":
    main()

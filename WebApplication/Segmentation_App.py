import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():

    st.set_page_config(page_title="HistologyNet Segmentation App",page_icon="👋", layout="wide")

    # Title and sidebar
    st.title("HistologyNet Segmentation App")

    st.write(
        """


        Welcome to the HistologyNet Segmentation App! This app allows you to segment histology images using our Self Supervised model.

        Upload an image using the sidebar, then click the button to segment it.

        """
    )

    
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("___________________________________________", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
        # Button to perform segmentation
        if st.sidebar.button("Segment Histology", key="segment_button"):
            with st.spinner("Segmenting..."):
                segmented_image = segment_iris(image)
                pil_img = numpy_to_pil(segmented_image)
                st.image(segmented_image, caption="Segmented Histology", use_column_width=True, channels="GRAY")

                buf = io.BytesIO()
                # Convert image to 'RGB' mode
                pil_img_rgb = pil_img.convert('RGB')

                pil_img_rgb.save(buf, format="JPEG")

                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Segmented Image",
                    data=byte_im,
                    file_name="segmented_image.jpeg",
                    mime="image/jpeg"
                )
                st.success("Segmentation completed successfully!")
      # Display an example image
    example_image = Image.open("Histology.png")  # Replace "example_image.jpg" with the path to your example image
    st.image(example_image, caption="Example Image", use_column_width=True)

    # File upload


def segment_iris(image):
    import cv2
    from transformers import SamModel, SamConfig, SamProcessor
    import torch

    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the model architecture with the loaded configuration
    my_model = SamModel(config=model_config)
    #Update the model by loading the weights from saved file.
    my_model.load_state_dict(torch.load("model_checkpoint.pth", map_location=torch.device('cpu')))
    print("Model loaded")
    # set the device to cuda if available, otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device ="cpu"
    my_model.to(device)

    import numpy as np
    import random
    import torch
    import matplotlib.pyplot as plt

    import os
    from PIL import Image


    test_image = image
    prompt = [0,0,255,255]

    inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    my_model.eval()

    # forward pass
    with torch.no_grad():
        outputs = my_model(**inputs, multimask_output=False)

    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

    # save_image(medsam_seg, result_path, file_path_image)
    return medsam_seg*255
def numpy_to_pil(image_np):
    return Image.fromarray(image_np)

    # st.success("Segmentation completed successfully!")

if __name__ == '__main__':
    main()

import streamlit as st
import torch
import torchxrayvision as xrv
import numpy as np
import skimage
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
     page_title="AI Medical Imaging Assistant",
    page_icon="🩻",
    layout="wide"
)

st.title(" AI-Powered Medical Imaging Assistant")
st.write(
    "Educational prototype for chest X-ray analysis."
)

st.warning(
    "Research demonstration only. This tool is not validated "
    "for diagnosis or treatment decisions. Results require "
    "review by a qualified radiologist."
)

@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )
    model.eval()
    return model

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    if st.button("Analyze X-ray"):
        with st.spinner("Loading model and analyzing..."):
            try:
                model = load_model()

                # Preprocess the image
                image_array = np.array(image)

                image_array = skimage.transform.resize(
                    image_array,
                    (224, 224),
                    preserve_range=True
                ).astype(np.float32)

                image_array = (
                    image_array / 127.5
                ) - 1.0

                image_tensor = torch.from_numpy(
                    image_array
                ).unsqueeze(0).unsqueeze(0)

                # Model prediction
                with torch.no_grad():
                    predictions = model(image_tensor)

                scores = predictions[0].cpu().numpy()
                labels = model.pathologies

                # Display the model scores
                results = sorted(
                    zip(labels, scores),
                    key=lambda item: item[1],
                    reverse=True
                )

                with col2:
                    st.subheader("Model output")

                    st.write(
                        "Highest model scores (not a diagnosis):"
                    )

                    for label, score in results[:5]:
                        st.write(
                            f"**{label}**: {score:.3f}"
                        )

                # Create a draft report
                st.subheader("Draft report")

                st.write(
                    "Image uploaded and processed successfully."
                )
                st.write(
                    "The model generated scores for its supported "
                    "chest X-ray findings. These scores must be "
                    "reviewed and interpreted by a radiologist."
                )

                st.download_button(
                    "Download draft report",
                    data=(
                        "AI Medical Imaging Assistant\n\n"
                        "Educational prototype output.\n"
                        "This is not a medical diagnosis.\n\n"
                        + "\n".join(
                            f"{label}: {score:.3f}"
                            for label, score in results[:5]
                        )
                    ),
                    file_name="draft_report.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(
                    f"Analysis failed: {error}"
                )
                st.info(
                    "Check your package versions and internet "
                    "connection. The pretrained weights may need "
                    "to download on first use."
                )
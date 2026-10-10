import cv2
import streamlit as st
from ultralytics import YOLO, YOLOWorld
from PIL import Image
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multiple Object Detection",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# GET APP FOLDER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = [
        YOLO(os.path.join(BASE_DIR, "yolo26s.pt")),
        YOLOWorld(os.path.join(BASE_DIR, "yolov8s-world.pt")),
        YOLO(os.path.join(BASE_DIR, "last.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "Bridge", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "Bridge", "2best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "Bridge Demage", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "Desert", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "forest", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "sea", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "store", "best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "University", "1best.pt")),
        YOLO(os.path.join(BASE_DIR, "Models", "University", "best.pt"))
    ]

    return models


models = load_models()


# ============================================================
# TITLE
# ============================================================

st.title("🎯 Multiple Object Detection")

st.write(
    "Upload an image and detect multiple objects "
    "using different YOLO models."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

upload_file = st.file_uploader(
    "📤 Choose an Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IF IMAGE IS UPLOADED
# ============================================================

if upload_file is not None:

    # Open uploaded image
    image = Image.open(upload_file)

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )


    # ========================================================
    # DETECT BUTTON
    # ========================================================

    if st.button("🚀 Detect Objects"):

        st.write("### 🔍 Detecting Objects...")

        # Total object counter
        total_objects = 0


        # ====================================================
        # RUN ALL MODELS
        # ====================================================

        for model in models:

            results = model(
                image,
                conf=0.5,
                imgsz=640,
                verbose=False
            )


            # =================================================
            # GET DETECTIONS
            # =================================================

            for result in results:

                for box in result.boxes:

                    # Bounding Box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Confidence
                    conf = float(
                        box.conf[0]
                    )

                    # Class ID
                    cls = int(
                        box.cls[0]
                    )

                    # Class Name
                    name = model.names[cls]

                    total_objects += 1


        # ====================================================
        # DRAW FINAL RESULT
        # ====================================================

        # Use the result from all detections
        result_image = image.copy()

        # Convert PIL image to OpenCV image
        img = cv2.cvtColor(
            __import__("PIL").Image.Image.tobytes(result_image)
            if False else
            cv2.imread(upload_file.name),
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # RUN MODELS AGAIN FOR DISPLAY
        # ====================================================

        img = cv2.resize(
            img,
            (640, 640)
        )

        for model in models:

            results = model(
                img,
                conf=0.5,
                imgsz=640,
                verbose=False
            )

            for result in results:

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    conf = float(
                        box.conf[0]
                    )

                    cls = int(
                        box.cls[0]
                    )

                    name = model.names[cls]


                    # Draw rectangle
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )


                    # Draw label
                    cv2.putText(
                        img,
                        f"{name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        1
                    )


        # ====================================================
        # SHOW RESULT
        # ====================================================

        st.write("### 🎯 Detection Result")

        st.image(
            img,
            caption="Detected Objects",
            width="stretch"
        )


        # ====================================================
        # INFORMATION
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="🎯 Objects Detected",
                value=total_objects
            )

        with col2:

            st.metric(
                label="🤖 YOLO Models",
                value=len(models)
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Detection completed successfully!"
        )
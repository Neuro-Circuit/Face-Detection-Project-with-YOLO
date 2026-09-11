import os
import cv2
import pickle
import numpy as np
import tkinter as tk

from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

YOLO_MODEL_PATH = "models/yolov8m-face.pt"
DATABASE_PATH = "face_database.pkl"

SIMILARITY_THRESHOLD = 0.35
YOLO_CONFIDENCE = 0.40

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
)


# ============================================================
# Load Models
# ============================================================

print("Loading YOLO...")

try:
    yolo = YOLO(YOLO_MODEL_PATH)
except Exception as error:
    print("Failed to load YOLO:")
    print(error)
    raise


print("Loading InsightFace...")

try:
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

except Exception as error:
    print("Failed to load InsightFace:")
    print(error)
    raise


print("Models loaded successfully.")


# ============================================================
# Global Variables
# ============================================================

database = []

video_capture = None
video_running = False
current_video_path = None
current_photo = None


# ============================================================
# Face Embedding
# ============================================================

def normalize_embedding(embedding):

    if embedding is None:
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.size == 0:
        return None

    norm = np.linalg.norm(embedding)

    if norm <= 0:
        return None

    return embedding / norm


def cosine_similarity(a, b):

    a = normalize_embedding(a)
    b = normalize_embedding(b)

    if a is None or b is None:
        return -1.0

    return float(
        np.dot(a, b)
    )


# ============================================================
# Database
# ============================================================

def load_database():

    global database

    if not os.path.exists(DATABASE_PATH):

        database = []

        return

    try:

        with open(
            DATABASE_PATH,
            "rb"
        ) as file:

            loaded = pickle.load(file)

        if isinstance(loaded, list):
            database = loaded
        else:
            database = []

        # Remove broken database entries.
        valid_database = []

        for item in database:

            if not isinstance(item, dict):
                continue

            if "name" not in item:
                continue

            if "embedding" not in item:
                continue

            embedding = normalize_embedding(
                item["embedding"]
            )

            if embedding is None:
                continue

            item["embedding"] = embedding

            valid_database.append(item)

        database = valid_database

        print(
            f"Loaded {len(database)} reference images."
        )

    except Exception as error:

        print(
            "Could not load database:",
            error
        )

        database = []


def save_database():

    try:

        with open(
            DATABASE_PATH,
            "wb"
        ) as file:

            pickle.dump(
                database,
                file
            )

        print(
            f"Database saved: {len(database)} reference images."
        )

        return True

    except Exception as error:

        messagebox.showerror(
            "Database Error",
            "Could not save the face database.\n\n"
            + str(error)
        )

        return False


# ============================================================
# Safe Face Detection
# ============================================================

def get_faces(image):

    if image is None:
        return []

    try:

        faces = face_app.get(
            image
        )

        if faces is None:
            return []

        return list(faces)

    except Exception as error:

        print(
            "InsightFace error:",
            error
        )

        messagebox.showerror(
            "Face Analysis Error",
            "InsightFace could not analyze the image.\n\n"
            "Error:\n"
            + str(error)
        )

        return []


# ============================================================
# Add Reference Image
# ============================================================

def add_reference_image():

    try:

        stop_video()

        path = filedialog.askopenfilename(
            parent=root,
            title="Select Reference Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ]
        )

        if not path:
            return

        image = cv2.imread(
            path
        )

        if image is None:

            messagebox.showerror(
                "Image Error",
                "The selected image could not be opened."
            )

            return

        status_label.config(
            text="Analyzing reference image..."
        )

        root.update_idletasks()

        faces = get_faces(
            image
        )

        if len(faces) == 0:

            status_label.config(
                text="Ready"
            )

            messagebox.showwarning(
                "No Face Found",
                "No face was detected in the selected image.\n\n"
                "Please choose an image containing a clear face."
            )

            return

        # Select the largest face.
        largest_face = max(
            faces,
            key=lambda face:
            max(
                0,
                float(face.bbox[2] - face.bbox[0])
            )
            *
            max(
                0,
                float(face.bbox[3] - face.bbox[1])
            )
        )

        embedding = normalize_embedding(
            getattr(
                largest_face,
                "embedding",
                None
            )
        )

        if embedding is None:

            status_label.config(
                text="Ready"
            )

            messagebox.showerror(
                "Embedding Error",
                "A face was detected, but its embedding "
                "could not be generated."
            )

            return

        name = simpledialog.askstring(
            "Person Name",
            "Enter the person's name:",
            parent=root
        )

        if name is None:
            status_label.config(
                text="Ready"
            )
            return

        name = name.strip()

        if not name:

            status_label.config(
                text="Ready"
            )

            messagebox.showwarning(
                "Invalid Name",
                "The person's name cannot be empty."
            )

            return

        # Save a new reference image.
        # Multiple images for the same person are supported.
        database.append(
            {
                "name": name,
                "embedding": embedding,
                "image_path": path
            }
        )

        if not save_database():

            # Roll back if saving failed.
            database.pop()

            status_label.config(
                text="Ready"
            )

            return

        update_people_list()

        status_label.config(
            text="Reference image saved"
        )

        add_chat_message(
            "AI",
            f'Reference image saved for "{name}".'
        )

        messagebox.showinfo(
            "Reference Saved",
            f'Reference image for "{name}" was saved successfully.'
        )

        status_label.config(
            text="Ready"
        )

    except Exception as error:

        status_label.config(
            text="Ready"
        )

        messagebox.showerror(
            "Reference Image Error",
            "An error occurred while adding the reference image.\n\n"
            + str(error)
        )

        print(
            "Reference image error:",
            error
        )


# ============================================================
# Face Recognition
# ============================================================

def recognize_face(embedding):

    if not database:
        return "Unknown", 0.0

    embedding = normalize_embedding(
        embedding
    )

    if embedding is None:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = -1.0

    for person in database:

        reference_embedding = normalize_embedding(
            person.get("embedding")
        )

        if reference_embedding is None:
            continue

        score = cosine_similarity(
            embedding,
            reference_embedding
        )

        if score > best_score:

            best_score = score
            best_name = person.get(
                "name",
                "Unknown"
            )

    if best_score >= SIMILARITY_THRESHOLD:

        return (
            best_name,
            best_score
        )

    return (
        "Unknown",
        best_score
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        /
        union_area
    )


def find_best_matching_face(
    yolo_box,
    insight_faces
):

    best_face = None
    best_iou = 0.0

    for face in insight_faces:

        try:

            face_box = (
                float(face.bbox[0]),
                float(face.bbox[1]),
                float(face.bbox[2]),
                float(face.bbox[3]),
            )

            iou = calculate_iou(
                yolo_box,
                face_box
            )

            if iou > best_iou:

                best_iou = iou
                best_face = face

        except Exception:
            continue

    if best_iou >= 0.10:
        return best_face

    return None


# ============================================================
# Drawing
# ============================================================

def draw_name(
    frame,
    name,
    x1,
    y1,
    x2,
    y2
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    text_size = cv2.getTextSize(
        name,
        font,
        font_scale,
        thickness
    )[0]

    text_width = text_size[0]
    text_height = text_size[1]

    label_x1 = max(
        0,
        x1
    )

    label_y1 = min(
        frame.shape[0] - 1,
        y2
    )

    label_x2 = min(
        frame.shape[1] - 1,
        x1 + text_width + 18
    )

    label_y2 = min(
        frame.shape[0],
        y2 + text_height + 16
    )

    # Blue background
    cv2.rectangle(
        frame,
        (
            label_x1,
            label_y1
        ),
        (
            label_x2,
            label_y2
        ),
        (255, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        name,
        (
            label_x1 + 8,
            label_y2 - 6
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_face_result(
    frame,
    x1,
    y1,
    x2,
    y2,
    name
):

    # Blue bounding box
    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (255, 0, 0),
        3
    )

    draw_name(
        frame,
        name,
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# Process One Frame
# ============================================================

def recognize_frame(frame):

    if frame is None:
        return frame

    try:

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        yolo_result = yolo(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )[0]

        # ----------------------------------------------------
        # InsightFace
        # ----------------------------------------------------

        insight_faces = get_faces(
            frame
        )

        # ----------------------------------------------------
        # If YOLO found no faces,
        # use InsightFace detections directly.
        # ----------------------------------------------------

        if (
            yolo_result.boxes is None
            or
            len(yolo_result.boxes) == 0
        ):

            for face in insight_faces:

                try:

                    coordinates = face.bbox

                    x1 = max(
                        0,
                        int(coordinates[0])
                    )

                    y1 = max(
                        0,
                        int(coordinates[1])
                    )

                    x2 = min(
                        frame.shape[1] - 1,
                        int(coordinates[2])
                    )

                    y2 = min(
                        frame.shape[0] - 1,
                        int(coordinates[3])
                    )

                    name, score = recognize_face(
                        face.embedding
                    )

                    draw_face_result(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        name
                    )

                except Exception:
                    continue

            return frame

        used_faces = set()

        # ----------------------------------------------------
        # Process YOLO faces
        # ----------------------------------------------------

        for box in yolo_result.boxes:

            try:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1 = max(
                    0,
                    int(coordinates[0])
                )

                y1 = max(
                    0,
                    int(coordinates[1])
                )

                x2 = min(
                    frame.shape[1] - 1,
                    int(coordinates[2])
                )

                y2 = min(
                    frame.shape[0] - 1,
                    int(coordinates[3])
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                yolo_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                matched_face = find_best_matching_face(
                    yolo_box,
                    insight_faces
                )

                name = "Unknown"

                if matched_face is not None:

                    face_id = id(
                        matched_face
                    )

                    if face_id not in used_faces:

                        used_faces.add(
                            face_id
                        )

                        name, score = recognize_face(
                            matched_face.embedding
                        )

                        print(
                            f"{name} - similarity: {score:.3f}"
                        )

                draw_face_result(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    name
                )

            except Exception as error:

                print(
                    "Frame face processing error:",
                    error
                )

        # ----------------------------------------------------
        # Draw InsightFace faces missed by YOLO
        # ----------------------------------------------------

        for face in insight_faces:

            try:

                face_id = id(
                    face
                )

                if face_id in used_faces:
                    continue

                coordinates = face.bbox

                x1 = max(
                    0,
                    int(coordinates[0])
                )

                y1 = max(
                    0,
                    int(coordinates[1])
                )

                x2 = min(
                    frame.shape[1] - 1,
                    int(coordinates[2])
                )

                y2 = min(
                    frame.shape[0] - 1,
                    int(coordinates[3])
                )

                name, score = recognize_face(
                    face.embedding
                )

                draw_face_result(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    name
                )

            except Exception:
                continue

        return frame

    except Exception as error:

        print(
            "Frame processing error:",
            error
        )

        return frame


# ============================================================
# Display Frame
# ============================================================

def display_frame(frame):

    global current_photo

    if frame is None:
        return

    try:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb_frame
        )

        display_width = max(
            400,
            media_label.winfo_width()
        )

        display_height = max(
            300,
            media_label.winfo_height()
        )

        image_width = image.width
        image_height = image.height

        if image_width <= 0 or image_height <= 0:
            return

        scale = min(
            display_width / image_width,
            display_height / image_height
        )

        scale = min(
            scale,
            1.0
        )

        new_width = max(
            1,
            int(image_width * scale)
        )

        new_height = max(
            1,
            int(image_height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        current_photo = ImageTk.PhotoImage(
            image
        )

        media_label.configure(
            image=current_photo,
            text=""
        )

        media_label.image = current_photo

    except Exception as error:

        print(
            "Display error:",
            error
        )


# ============================================================
# Recognize Image
# ============================================================

def recognize_image():

    stop_video()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            )
        ]
    )

    if not path:
        return

    image = cv2.imread(
        path
    )

    if image is None:

        messagebox.showerror(
            "Image Error",
            "Could not open the selected image."
        )

        return

    add_chat_message(
        "You",
        f"Image selected: {os.path.basename(path)}"
    )

    status_label.config(
        text="Analyzing image..."
    )

    root.update_idletasks()

    result = recognize_frame(
        image.copy()
    )

    display_frame(
        result
    )

    status_label.config(
        text="Ready"
    )

    add_chat_message(
        "AI",
        "Image recognition finished."
    )


# ============================================================
# Start Video
# ============================================================

def start_video_from_path(path):

    global video_capture
    global video_running
    global current_video_path

    stop_video()

    video_capture = cv2.VideoCapture(
        path
    )

    if not video_capture.isOpened():

        video_capture = None

        messagebox.showerror(
            "Video Error",
            "Could not open the selected video."
        )

        return

    current_video_path = path
    video_running = True

    add_chat_message(
        "You",
        f"Video selected: {os.path.basename(path)}"
    )

    add_chat_message(
        "AI",
        "Video recognition started."
    )

    status_label.config(
        text="Playing video..."
    )

    update_video_frame()


# ============================================================
# Recognize Video
# ============================================================

def recognize_video():

    path = filedialog.askopenfilename(
        parent=root,
        title="Select Video",
        filetypes=[
            (
                "Video Files",
                "*.mp4 *.avi *.mov *.mkv *.wmv"
            )
        ]
    )

    if not path:
        return

    start_video_from_path(
        path
    )


# ============================================================
# Video Update
# ============================================================

def update_video_frame():

    global video_capture
    global video_running

    if not video_running:
        return

    if video_capture is None:
        return

    try:

        success, frame = video_capture.read()

        if not success:

            stop_video()

            add_chat_message(
                "AI",
                "Video finished."
            )

            return

        result = recognize_frame(
            frame
        )

        display_frame(
            result
        )

        root.after(
            1,
            update_video_frame
        )

    except Exception as error:

        print(
            "Video processing error:",
            error
        )

        stop_video()

        messagebox.showerror(
            "Video Error",
            str(error)
        )


# ============================================================
# Stop Video
# ============================================================

def stop_video():

    global video_capture
    global video_running

    video_running = False

    if video_capture is not None:

        try:
            video_capture.release()
        except Exception:
            pass

        video_capture = None

    if "status_label" in globals():

        status_label.config(
            text="Ready"
        )


# ============================================================
# Upload Image / Video
# ============================================================

def upload_media():

    path = filedialog.askopenfilename(
        parent=root,
        title="Select Image or Video",
        filetypes=[
            (
                "Images and Videos",
                "*.jpg *.jpeg *.png *.bmp *.webp "
                "*.mp4 *.avi *.mov *.mkv *.wmv"
            ),
            (
                "Images",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            (
                "Videos",
                "*.mp4 *.avi *.mov *.mkv *.wmv"
            ),
        ]
    )

    if not path:
        return

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension in IMAGE_EXTENSIONS:

        stop_video()

        image = cv2.imread(
            path
        )

        if image is None:

            messagebox.showerror(
                "Image Error",
                "Could not open the selected image."
            )

            return

        add_chat_message(
            "You",
            f"Image selected: {os.path.basename(path)}"
        )

        status_label.config(
            text="Analyzing image..."
        )

        root.update_idletasks()

        result = recognize_frame(
            image.copy()
        )

        display_frame(
            result
        )

        status_label.config(
            text="Ready"
        )

        add_chat_message(
            "AI",
            "Image recognition finished."
        )

    elif extension in VIDEO_EXTENSIONS:

        start_video_from_path(
            path
        )

    else:

        messagebox.showerror(
            "Unsupported File",
            "This file type is not supported."
        )


# ============================================================
# Chat
# ============================================================

def add_chat_message(
    sender,
    message
):

    chat_box.config(
        state="normal"
    )

    chat_box.insert(
        "end",
        f"{sender}: {message}\n\n"
    )

    chat_box.config(
        state="disabled"
    )

    chat_box.see(
        "end"
    )


# ============================================================
# Saved People
# ============================================================

def update_people_list():

    for widget in people_frame.winfo_children():

        widget.destroy()

    if not database:

        tk.Label(
            people_frame,
            text="No reference faces saved.",
            font=(
                "Segoe UI",
                10
            )
        ).pack(
            anchor="w",
            padx=10,
            pady=8
        )

        return

    people = {}

    for item in database:

        name = item.get(
            "name",
            "Unknown"
        )

        if name not in people:
            people[name] = 0

        people[name] += 1

    tk.Label(
        people_frame,
        text=f"Total reference images: {len(database)}",
        font=(
            "Segoe UI",
            10,
            "bold"
        )
    ).pack(
        anchor="w",
        padx=10,
        pady=(8, 5)
    )

    for name, count in people.items():

        row = tk.Frame(
            people_frame
        )

        row.pack(
            fill="x",
            padx=5,
            pady=3
        )

        tk.Label(
            row,
            text=name,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side="left",
            padx=5
        )

        tk.Label(
            row,
            text=f"({count} reference images)",
            font=(
                "Segoe UI",
                9
            )
        ).pack(
            side="left"
        )


# ============================================================
# Clear Database
# ============================================================

def clear_database():

    global database

    if not database:

        messagebox.showinfo(
            "Database",
            "The database is already empty."
        )

        return

    answer = messagebox.askyesno(
        "Delete All Reference Faces",
        "Are you sure you want to delete ALL saved reference faces?"
    )

    if not answer:
        return

    database = []

    save_database()

    update_people_list()

    add_chat_message(
        "AI",
        "All reference faces were deleted."
    )


# ============================================================
# Clear Display
# ============================================================

def clear_media():

    stop_video()

    media_label.configure(
        image="",
        text="No media selected"
    )

    media_label.image = None

    status_label.config(
        text="Ready"
    )

    add_chat_message(
        "AI",
        "Media display cleared."
    )


# ============================================================
# Main Window
# ============================================================

root = tk.Tk()

root.title(
    "AI Face Recognition"
)

root.geometry(
    "1400x850"
)

root.minsize(
    1100,
    700
)


# ============================================================
# Main Layout
# ============================================================

left_frame = tk.Frame(
    root,
    width=320
)

left_frame.pack(
    side="left",
    fill="y",
    padx=10,
    pady=10
)

left_frame.pack_propagate(
    False
)


right_frame = tk.Frame(
    root
)

right_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ============================================================
# Left Panel
# ============================================================

tk.Label(
    left_frame,
    text="AI Face Recognition",
    font=(
        "Segoe UI",
        20,
        "bold"
    )
).pack(
    pady=(10, 20)
)


tk.Label(
    left_frame,
    text="Reference Faces",
    font=(
        "Segoe UI",
        14,
        "bold"
    )
).pack(
    anchor="w",
    pady=(5, 8)
)


tk.Label(
    left_frame,
    text=(
        "Add reference photos and assign a name.\n"
        "You can add unlimited reference photos."
    ),
    justify="left",
    font=(
        "Segoe UI",
        9
    )
).pack(
    anchor="w",
    pady=(0, 8)
)


tk.Button(
    left_frame,
    text="Add Reference Image",
    command=add_reference_image,
    font=(
        "Segoe UI",
        11,
        "bold"
    ),
    height=2
).pack(
    fill="x",
    pady=4
)


# ============================================================
# Recognition
# ============================================================

tk.Label(
    left_frame,
    text="Recognition",
    font=(
        "Segoe UI",
        14,
        "bold"
    )
).pack(
    anchor="w",
    pady=(20, 8)
)


tk.Button(
    left_frame,
    text="Recognize Image",
    command=recognize_image,
    font=(
        "Segoe UI",
        11,
        "bold"
    ),
    height=2
).pack(
    fill="x",
    pady=4
)


tk.Button(
    left_frame,
    text="Recognize Video",
    command=recognize_video,
    font=(
        "Segoe UI",
        11,
        "bold"
    ),
    height=2
).pack(
    fill="x",
    pady=4
)


tk.Button(
    left_frame,
    text="Upload Image / Video",
    command=upload_media,
    font=(
        "Segoe UI",
        11
    ),
    height=2
).pack(
    fill="x",
    pady=4
)


tk.Button(
    left_frame,
    text="Stop Video",
    command=stop_video,
    font=(
        "Segoe UI",
        10
    )
).pack(
    fill="x",
    pady=4
)


tk.Button(
    left_frame,
    text="Clear Display",
    command=clear_media,
    font=(
        "Segoe UI",
        10
    )
).pack(
    fill="x",
    pady=4
)


# ============================================================
# Saved People
# ============================================================

tk.Label(
    left_frame,
    text="Saved People",
    font=(
        "Segoe UI",
        13,
        "bold"
    )
).pack(
    anchor="w",
    pady=(20, 5)
)


people_frame = tk.Frame(
    left_frame,
    relief="groove",
    borderwidth=1
)

people_frame.pack(
    fill="both",
    expand=True
)


tk.Button(
    left_frame,
    text="Delete All Reference Faces",
    command=clear_database,
    font=(
        "Segoe UI",
        9
    )
).pack(
    fill="x",
    pady=(6, 0)
)


# ============================================================
# Right Panel
# ============================================================

tk.Label(
    right_frame,
    text="Face Recognition Result",
    font=(
        "Segoe UI",
        18,
        "bold"
    )
).pack(
    anchor="w",
    pady=(5, 8)
)


# ============================================================
# Media Display
# ============================================================

media_frame = tk.Frame(
    right_frame,
    relief="groove",
    borderwidth=2
)

media_frame.pack(
    fill="both",
    expand=True
)


media_label = tk.Label(
    media_frame,
    text="No media selected",
    font=(
        "Segoe UI",
        16
    ),
    anchor="center"
)

media_label.pack(
    fill="both",
    expand=True
)


# ============================================================
# Chat
# ============================================================

chat_frame = tk.Frame(
    right_frame
)

chat_frame.pack(
    fill="x",
    pady=8
)


chat_box = tk.Text(
    chat_frame,
    height=6,
    font=(
        "Segoe UI",
        10
    ),
    wrap="word",
    state="disabled"
)

chat_box.pack(
    fill="both",
    expand=True
)


# ============================================================
# Status
# ============================================================

bottom_frame = tk.Frame(
    right_frame
)

bottom_frame.pack(
    fill="x",
    pady=5
)


status_label = tk.Label(
    bottom_frame,
    text="Ready",
    font=(
        "Segoe UI",
        10,
        "bold"
    )
)

status_label.pack(
    side="left"
)


tk.Label(
    bottom_frame,
    text=(
        "Blue box = face detected | "
        "Name = recognized | "
        "Unknown = not found in database"
    ),
    font=(
        "Segoe UI",
        9
    )
).pack(
    side="right"
)


# ============================================================
# Initialize
# ============================================================

load_database()

update_people_list()

add_chat_message(
    "AI",
    "Welcome!"
)

add_chat_message(
    "AI",
    "Add reference images first, then recognize images or videos."
)


# ============================================================
# Close
# ============================================================

def on_close():

    stop_video()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)


# ============================================================
# Start Application
# ============================================================

root.mainloop()
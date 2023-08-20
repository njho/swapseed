import os
import gdown
import insightface
from insightface.app import FaceAnalysis
from faceswap import swap_face_single

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown
model_url = "https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
model_output_path = "inswapper/inswapper_128.onnx"
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model(
    "inswapper/inswapper_128.onnx", download=False, download_zip=False
)

# Load images
target_img = "images/man_target.jpeg"
source_img = "images/woman1.jpeg"

# Swap faces between two images
# swap_n_show(target_img, source_img, app, swapper)

# Swap faces within the same image
# swap_n_show_same_img(target_img, app, swapper)

# Add face to an image
swap_face_single(
    target_img,
    source_img,
    app,
    swapper,
    enhance=True,
    enhancer="REAL-ESRGAN 2x",
    device="cpu",
)

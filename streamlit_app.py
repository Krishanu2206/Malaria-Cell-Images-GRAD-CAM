import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ModuleNotFoundError:
    cv2 = None
    HAS_CV2 = False

# --------- Page Configuration ---------
st.set_page_config(page_title="Malaria Cell Grad-CAM", layout="wide")

# --------- Model definition (matches notebook) ---------
def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.MaxPool2d(kernel_size=2)
    )

class MalariaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --------- Utilities (denormalize, upsample) ---------
def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)
    img = (tensor.cpu() * std) + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

RESIZE_SIZE = 128


def resize_image(arr, size=(RESIZE_SIZE, RESIZE_SIZE)):
    """Resize an HWC uint8 image using cv2 if available, else Pillow."""
    if HAS_CV2:
        return cv2.resize(arr, size)
    return np.array(Image.fromarray(arr).resize(size, Image.BILINEAR))

def upsampleHeatmap(raw_map, img):
    """Overlay heatmap on original image."""
    raw_map = np.float32(raw_map)
    m, M = raw_map.min(), raw_map.max()
    raw_map = 255 * ((raw_map - m) / max((M - m), 1e-8))
    raw_map = np.uint8(np.clip(raw_map, 0, 255))

    if HAS_CV2:
        raw_map = cv2.resize(raw_map, (RESIZE_SIZE, RESIZE_SIZE))
        colored = cv2.applyColorMap(raw_map, cv2.COLORMAP_JET)
    else:
        # Pillow fallback: pseudo heatmap without OpenCV dependency
        raw_map = np.array(Image.fromarray(raw_map).resize((RESIZE_SIZE, RESIZE_SIZE), Image.BILINEAR))
        colored = np.zeros((RESIZE_SIZE, RESIZE_SIZE, 3), dtype=np.uint8)
        colored[..., 0] = raw_map  # R
        colored[..., 1] = np.clip(255 - raw_map, 0, 255)  # G
        colored[..., 2] = 64  # B baseline

    overlay = np.uint8(colored * 0.7 + img * 0.3)
    return overlay

# --------- Grad-CAM implementation (hook-based) ---------
def im2gradCAM(model, x, device):
    """Compute Grad-CAM heatmap and prediction."""
    model.eval()
    x = x.to(device)

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        # Store a fully independent copy — never a view of the graph
        activations = out.detach().clone()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        # register_backward_hook (non-full) passes grad_output directly.
        # We immediately detach + clone + contiguous so we hold a plain
        # numpy-ready tensor with no autograd ancestry whatsoever.
        g = grad_out[0]
        if g is not None:
            gradients = g.detach().clone().contiguous()

    # Hook the final conv layer: model.model[5] = ConvBlock(512,64), [1] = Conv2d
    target_layer = model.model[5][1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    # Use register_backward_hook instead of register_full_backward_hook to
    # avoid the "output is a view / inplace modification forbidden" autograd
    # error that full-backward hooks trigger on PyTorch >= 2.x.
    handle_b = target_layer.register_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            logits = model(x)
            pred_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
            model.zero_grad()
            logits[0, pred_idx].backward()

        if activations is None or gradients is None:
            raise RuntimeError('Failed to capture activations or gradients')

        # All arithmetic below is outside autograd — use no_grad for safety
        with torch.no_grad():
            # Pool gradients over spatial dims → (C,)
            pooled_grads = torch.mean(gradients.cpu(), dim=(0, 2, 3))
            # Weight each activation channel and average → (H, W)
            acts = activations.cpu()                                   # (1, C, H, W)
            weighted_acts = acts * pooled_grads.view(1, -1, 1, 1)     # (1, C, H, W)
            heatmap = torch.mean(weighted_acts, dim=1)[0].numpy()      # (H, W)

        return heatmap, pred_idx
    finally:
        handle_f.remove()
        handle_b.remove()

# --------- Streamlit UI ---------
st.title('🔬 Malaria Cell Classification with Grad-CAM')
st.markdown('Upload cell images to classify as **Parasitized** or **Uninfected** with visual explanations via Grad-CAM')

# Sidebar controls
with st.sidebar:
    st.header('⚙️ Settings')

    weights_file = st.text_input(
        'Path to model weights',
        value='models/malaria_grad_cam.pth',
        help='Path to the saved PyTorch checkpoint'
    )

    use_cuda = st.checkbox(
        'Use GPU (CUDA)',
        value=torch.cuda.is_available(),
        disabled=not torch.cuda.is_available()
    )

    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    st.info(f'Using device: **{device}**')

    if st.button('🔄 Load Model Weights', use_container_width=True):
        st.session_state.load_model = True

# Load model function with caching
@st.cache_resource
def load_model_checkpoint(weights_path, device):
    """Load trained model from checkpoint."""
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f'Model file not found: {weights_path}')

    checkpoint = torch.load(weights_path, map_location=device)

    # Extract metadata
    num_classes = checkpoint.get('num_classes', 2)
    class_names = checkpoint.get('class_names', ['Parasitized', 'Uninfected'])

    # Initialize and load model
    model = MalariaClassifier(num_classes=num_classes)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, class_names

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.class_names = None
    st.session_state.load_model = False

# Load model if requested
if st.session_state.load_model:
    try:
        with st.spinner('Loading model...'):
            st.session_state.model, st.session_state.class_names = \
                load_model_checkpoint(weights_file, device)
        st.sidebar.success('✓ Model loaded successfully!')
        st.session_state.load_model = False
    except Exception as e:
        st.sidebar.error(f'✗ Error loading model: {str(e)}')
        st.session_state.load_model = False

# Main content
if st.session_state.model is None:
    st.info('📋 Please load the model weights using the sidebar to begin.')
else:
    st.success(f'✓ Model ready | Classes: {st.session_state.class_names}')

    # File uploader
    uploaded_files = st.file_uploader(
        'Upload cell images (PNG, JPG, JPEG)',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        st.markdown('---')
        st.subheader(f'📊 Results ({len(uploaded_files)} image(s))')

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Load and preprocess image
                image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
                input_tensor = transform(image)

                # Generate predictions and Grad-CAM
                heatmap, pred_idx = im2gradCAM(
                    st.session_state.model,
                    input_tensor.unsqueeze(0),
                    device
                )

                # Denormalize and overlay
                orig_img = denormalize(input_tensor)
                vis_img = resize_image(orig_img, (RESIZE_SIZE, RESIZE_SIZE))
                cam_overlay = upsampleHeatmap(heatmap, vis_img)

                # Display results
                pred_class = st.session_state.class_names[pred_idx]

                col1, col2 = st.columns(2)
                with col1:
                    st.image(vis_img, caption='Original Image', width="stretch")
                with col2:
                    st.image(cam_overlay, caption='Grad-CAM Heatmap', width="stretch")

                st.markdown(
                    f'**{uploaded_file.name}** → Predicted: **{pred_class}** '
                    f'({"🟢 Uninfected" if pred_idx == 1 else "🔴 Parasitized"})',
                    unsafe_allow_html=True
                )
                st.markdown('---')

            except Exception as e:
                st.error(f'Error processing {uploaded_file.name}: {str(e)}')

st.markdown('---')
st.markdown(
    '''**How to use:**
    1. Load model weights from sidebar
    2. Upload one or more cell images
    3. View predictions and Grad-CAM visualizations

    **Grad-CAM** shows which regions of the cell image influenced the model's prediction.
    '''
)
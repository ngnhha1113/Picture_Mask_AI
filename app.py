import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# 1. Cấu hình trang
st.set_page_config(page_title="AI Shield Final Fix", page_icon="🛡️")
st.title("🛡️ AI Image Shield (Bản Fix Lỗi Xám Màu)")

# 2. Tải Model
@st.cache_resource
def load_model():
    # Dùng model VGG19 thay vì MobileNet vì cấu trúc texture của nó phức tạp hơn
    # giúp lớp nhiễu tự nhiên hơn, ít bị lộ hơn.
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# 3. Các hàm xử lý ảnh tách biệt
# Hàm này chỉ đưa ảnh về dạng Tensor 0-1, KHÔNG chuẩn hóa (để giữ màu đúng)
def image_to_tensor(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Giá trị từ 0.0 đến 1.0
    ])
    return transform(image).unsqueeze(0)

# Hàm này CHỈ dùng để chuẩn hóa trước khi đưa vào model
def normalize_for_model(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std

# 4. Thuật toán I-FGSM Logic Mới
def attack_logic(model, original_tensor_01, epsilon, alpha, num_steps):
    """
    original_tensor_01: Ảnh gốc dạng tensor giá trị 0-1 (chưa normalize)
    """
    # Tạo bản sao để tấn công
    adv_image = original_tensor_01.clone().detach()
    adv_image.requires_grad = True
    
    # Lấy nhãn mục tiêu ban đầu
    initial_norm = normalize_for_model(original_tensor_01)
    output = model(initial_norm)
    target_label = output.max(1, keepdim=True)[1][0]
    
    criterion = nn.CrossEntropyLoss()

    # Thanh progress bar
    progress_bar = st.progress(0)

    for i in range(num_steps):
        # 1. Chuẩn hóa ảnh tạm thời để model "nhìn"
        input_norm = normalize_for_model(adv_image)
        
        # 2. Tính toán Loss
        output = model(input_norm)
        loss = criterion(output, target_label)
        
        # 3. Tính Gradient
        model.zero_grad()
        loss.backward()
        data_grad = adv_image.grad.data
        
        # 4. Thêm nhiễu vào ảnh gốc (ảnh 0-1)
        # Cộng nhiễu vào ảnh
        adv_image = adv_image + alpha * data_grad.sign()
        
        # 5. Chiếu (Projection) để đảm bảo nhiễu không quá lớn
        eta = torch.clamp(adv_image - original_tensor_01, min=-epsilon, max=epsilon)
        
        # 6. Quan trọng: Clamp về 0-1 chứ không phải cắt normalize
        adv_image = torch.clamp(original_tensor_01 + eta, min=0, max=1).detach()
        adv_image.requires_grad = True
        
        progress_bar.progress((i + 1) / num_steps)

    progress_bar.empty()
    return adv_image

# --- Giao Diện Chính ---

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(original_image_pil, caption="Ảnh gốc", use_container_width=True)

    with st.expander("Cấu hình"):
        # Epsilon rất nhỏ để không đổi màu
        epsilon = st.slider("Độ nhiễu (Epsilon)", 0.005, 0.05, 0.01, step=0.001, format="%.3f")
        steps = st.slider("Số bước lặp", 5, 50, 20)
    
    # Alpha nhỏ để nhiễu mịn
    alpha = 2.5 * epsilon / steps

    if st.button("🛡️ Kích hoạt bảo vệ"):
        with st.spinner("Đang xử lý..."):
            # 1. Chuyển ảnh sang Tensor 0-1
            img_tensor = image_to_tensor(original_image_pil)
            
            # 2. Chạy tấn công
            adv_tensor = attack_logic(model, img_tensor, epsilon, alpha, steps)
            
            # 3. Xuất ảnh kết quả
            final_tensor = adv_tensor.squeeze(0)
            final_image = transforms.ToPILImage()(final_tensor)
            
            # Resize về kích thước gốc
            final_image = final_image.resize(original_image_pil.size)
            
            st.success("Xong!")
            
            # So sánh
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image_pil, caption="Gốc", use_container_width=True)
            with col2:
                st.image(final_image, caption="Đã bảo vệ", use_container_width=True)
            
            # Download
            from io import BytesIO
            buf = BytesIO()
            final_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Tải ảnh về",
                data=byte_im,
                file_name="protected.png",
                mime="image/png"
            )
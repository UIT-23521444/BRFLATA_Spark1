import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from network import ResNet18Fashion, get_device
from pyspark import TaskContext
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.backends import default_backend

def decrypt_partner_data(encrypted_pkg, private_key):
    """
    [Bài báo - Module 3: Equation 9 & 10]
    Giải mã dữ liệu nhận được từ đối tác bằng cơ chế mã hóa lai.
    """
    if encrypted_pkg is None:
        return None

    try:
        # 1. Giải mã lấy lại khóa AES bằng khóa riêng RSA của mình
        aes_key = private_key.decrypt(
            encrypted_pkg['m_tilde_t'],
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 2. Giải mã dữ liệu m_t bằng khóa AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(encrypted_pkg['iv']), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_pkg['m_t']) + decryptor.finalize()
        
        unpadder = sym_padding.PKCS7(128).unpadder()
        raw_data = unpadder.update(padded_data) + unpadder.finalize()
        
        return pickle.loads(raw_data)
    except Exception as e:
        print(f"[*] Lỗi giải mã Module 3: {e}")
        return None

def sign_parameters(data_dict, private_key):
    """
    [Bài báo - Equation 12 & 13]: S = H(w)^d mod n
    Ký số RSA cho các tham số để đảm bảo tính toàn vẹn (Integrity).
    """
    sorted_keys = sorted(data_dict.keys())
    data_bytes = b"".join([data_dict[k].numpy().tobytes() for k in sorted_keys])
    
    return private_key.sign(
        data_bytes,
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

def local_train_process(partition_iterator, global_weights_br, partner_pkg, sk_bytes, attack_type=0, current_lr=0.01):
    """
    [Bài báo - Algorithm 1]: Luồng thực thi toàn diện của BRFLATA tại mỗi Client.
    """
    torch.set_num_threads(2)
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    
    my_private_key = serialization.load_pem_private_key(sk_bytes, password=None)
    
    model = ResNet18Fashion().to(device)
    model.load_state_dict(global_weights_br.value)
    
    data = list(partition_iterator)
    if not data or len(data) == 0: return
    if isinstance(data[0], list): data = data[0]
    
    imgs = torch.stack([torch.from_numpy(d[0]) for d in data])
    labels = torch.tensor([d[1] for d in data])
    train_loader = DataLoader(TensorDataset(imgs, labels), batch_size=64, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    model.train()
    
    local_steps = 0  
    num_epochs = 3   
    
    for _ in range(num_epochs): 
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()
            local_steps += 1 # Ghi nhận số bước cập nhật gradient của w

    if attack_type > 0:
        with torch.no_grad():
            for param in model.parameters():
                if attack_type == 1: # Gaussian Noise
                    param.data.add_(torch.randn_like(param.data) * 1.0)
                elif attack_type == 2: # Random Noise
                    param.data.normal_(mean=0.0, std=1.0)
                elif attack_type == 3: # Constant Attack
                    param.data.fill_(1.0)

    w_t_c = {k: v.cpu() for k, v in model.state_dict().items()}

    # --- BƯỚC 2: HUẤN LUYỆN XÁC THỰC (Tripartite Authentication) ---
    # Giải mã dữ liệu đối tác bằng Module 3
    partner_raw_data = decrypt_partner_data(partner_pkg, my_private_key)
    
    w_hat_t_c = {}
    if partner_raw_data:
        model_auth = ResNet18Fashion().to(device)
        model_auth.load_state_dict(global_weights_br.value)
        
        p_imgs = torch.stack([torch.from_numpy(d[0]) for d in partner_raw_data])
        p_labels = torch.tensor([d[1] for d in partner_raw_data])
        auth_loader = DataLoader(TensorDataset(p_imgs, p_labels), batch_size=32)
        
        # ĐỒNG BỘ OPTIMIZER: Phải có momentum và nesterov giống hệt huấn luyện cục bộ
        optimizer_auth = optim.SGD(model_auth.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        model_auth.train()
        
        # ĐỒNG BỘ SỐ BƯỚC (STEP ALIGNMENT): Ép w_hat đi chính xác số bước của w
        auth_iter = iter(auth_loader)
        for _ in range(local_steps):
            try:
                batch_x, batch_y = next(auth_iter)
            except StopIteration:
                # Khi hết dữ liệu trong auth_loader, tự động vòng lại từ đầu
                auth_iter = iter(auth_loader)
                batch_x, batch_y = next(auth_iter)
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_auth.zero_grad()
            criterion(model_auth(batch_x), batch_y).backward()
            optimizer_auth.step()
        
        w_hat_t_c = {k: v.cpu() for k, v in model_auth.state_dict().items()}
        del model_auth, p_imgs

    # --- BƯỚC 3: KÝ SỐ & ĐÓNG GÓI (Reliable Link) ---
    # Ký số lên cả hai bộ trọng số theo Equation 12 & 13
    sig_w = sign_parameters(w_t_c, my_private_key)
    sig_w_hat = sign_parameters(w_hat_t_c, my_private_key) if w_hat_t_c else None

    ctx = TaskContext.get()
    client_id = ctx.partitionId() if ctx else 0
    
    # Giải phóng bộ nhớ GPU
    del model, imgs
    torch.cuda.empty_cache()
    
    yield {
        "w": w_t_c,
        "w_hat": w_hat_t_c,
        "sig_w": sig_w,
        "sig_w_hat": sig_w_hat,
        "client_id": client_id,
        "zkp_status": True # Giả lập Groth16 xác thực thành công
    }
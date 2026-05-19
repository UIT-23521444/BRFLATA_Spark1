import torch
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

# ====================================================================
# MODULE 1: ADAPTIVE CLIENT MATCHING (Ghép cặp Client thích ứng)
# ====================================================================

def get_adaptive_pairing(credibility_table, round_t, num_clients, p_history, lambda_val=10):
    """
    [Bài báo - Module 1]: Adaptive Client Matching Mechanism.
    Đảm bảo tạo ra một Permutation (Hoán vị) hoàn hảo để mọi Prover đều có Verifier.
    """
    # 1. Sắp xếp các Client theo độ tin cậy r_c giảm dần (Tính thích ứng - Equation 5)
    # Việc sắp xếp giúp các máy có r_c gần nhau đứng cạnh nhau trong danh sách
    sorted_items = sorted(credibility_table.items(), key=lambda x: x[1], reverse=True)
    sorted_ids = [item[0] for item in sorted_items]
    
    # Kiểm tra vòng phục hồi (Rehabilitation round - Equation 6)
    is_rehab = (round_t % lambda_val == 0) and (round_t > 0)

    # 2. Tìm Offset (Độ lệch) tối ưu để tránh lặp lại lịch sử P (Equation 3 & 4)
    # Bài báo yêu cầu tránh ghép cặp đã tồn tại trong P (thường lưu C/3 vòng gần nhất)
    best_offset = 1
    for offset in range(1, num_clients):
        valid_offset = True
        for i in range(num_clients):
            prover = sorted_ids[i]
            verifier = sorted_ids[(i + offset) % num_clients]
            
            # Kiểm tra xem cặp (Verifier, Prover) này có trong lịch sử không
            if (verifier, prover) in p_history:
                valid_offset = False
                break
        
        if valid_offset:
            best_offset = offset
            break

    # 3. Tạo kết quả ghép cặp (One-to-One Mapping)
    # Đảm bảo tính Tripartite: Mỗi Verifier xác thực cho đúng một Prover
    pairing = {}
    for i in range(num_clients):
        # Người xác thực (Verifier)
        verifier = sorted_ids[i]
        # Người được xác thực (Prover)
        prover = sorted_ids[(i + best_offset) % num_clients]
        
        # Lưu vào từ điển: Verifier 'i' sẽ kiểm tra cho Prover 'prover'
        pairing[verifier] = prover
        
    return pairing
# ====================================================================
# MODULE 4: INCENTIVE MECHANISM & AIMD (Cơ chế khuyến khích)
# ====================================================================

def calculate_euclidean_distance(w1, w2):
    """
    [Bài báo - Equation 18]: d(w, w_hat) = ||w - w_hat||_2[cite: 304].
    Tính khoảng cách giữa tham số huấn luyện và tham số xác thực[cite: 303].
    """
    with torch.no_grad():
        v1 = torch.cat([p.detach().cpu().flatten() for p in w1.values()])
        v2 = torch.cat([p.detach().cpu().flatten() for p in w2.values()])
        return torch.dist(v1, v2, p=2).item()

def update_aimd(r_prev, dist, d_th, is_link_secure=True):
    """
    [Bài báo - Equation 19, 20, 21]: r_t = theta_1 * theta_2.
    Cơ chế Additive Increase / Multiplicative Decrease (AIMD)[cite: 52].
    """
    # 1. Tính toán theta_1 dựa trên kết quả xác thực tham số [cite: 309]
    if dist < 1e-4:
        # Phát hiện Attack 3 (Constant Attack): Khoảng cách quá nhỏ [cite: 355]
        theta_1 = r_prev * 0.01 
    elif dist < d_th:
        # Thưởng: Tăng cộng 0.1 [cite: 310, 317]
        theta_1 = r_prev + 0.1 
    else:
        # Phạt: Giảm nhân 0.01 [cite: 310, 317]
        theta_1 = r_prev * 0.01 

    # 2. Tính toán theta_2 dựa trên xác thực đường truyền (Module 3) [cite: 311]
    # theta_2 = 1 nếu vượt qua RSA/ZKP, ngược lại là 0.01 [cite: 256, 311]
    theta_2 = 1.0 if is_link_secure else 0.01
    
    # Equation 21: Kết hợp kết quả 
    r_new = theta_1 * theta_2
    return max(0.0, min(1.0, r_new))

# ====================================================================
# MODULE 3: RELIABLE COMMUNICATION (Uplink & Downlink)
# ====================================================================

def verify_rsa_signature(public_key_bytes, weights, signature):
    """
    [Bài báo - Equation 14 & 15]: Xác thực chữ ký RSA tại Server[cite: 250, 251].
    """
    try:
        public_key = serialization.load_pem_public_key(public_key_bytes)
        
        # [QUAN TRỌNG]: Sắp xếp keys giống hệt như lúc ký ở Client
        sorted_keys = sorted(weights.keys())
        
        # Sử dụng .contiguous() để đảm bảo an toàn bộ nhớ khi dữ liệu đi qua Spark
        param_bytes = b"".join([weights[k].cpu().contiguous().numpy().tobytes() for k in sorted_keys])
        
        public_key.verify(
            signature,
            param_bytes,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True 
    except Exception as e:
        return False # Xác thực thất bại [cite: 255]
def verify_zkp_proof(proof_package):
    """
    [Bài báo - Equation 17]: Xác thực bằng chứng ZKP Groth16[cite: 275].
    Client kiểm tra tính đúng đắn của Public Key nhận từ Server[cite: 278].
    """
    if not proof_package:
        return False
    # r = 1: Chấp nhận khóa; r = 0: Truyền lại [cite: 278]
    return proof_package.get("zkp_status", False)

# ====================================================================
# THIẾT LẬP THAM SỐ THỰC NGHIỆM
# ====================================================================

def estimate_similarity_threshold(sc, client_updates_list):
    """
    [Bài báo - Section 3.2.1]: Quy trình ước tính ngưỡng d_th.
    Tính giá trị cực đại của khoảng cách giữa các lần cập nhật epoch[cite: 345].
    """
    distances = []
    num = len(client_updates_list)
    for i in range(num):
        for j in range(i + 1, num):
            d = calculate_euclidean_distance(client_updates_list[i], client_updates_list[j])
            distances.append(d)
    
    # Ngưỡng d_th = ceil(max(distances)) [cite: 345]
    return max(distances) if distances else 6.0
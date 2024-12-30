import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os
from datetime import datetime
import psutil

sys.path.append('./BDCN')
from bdcn import BDCN


def list_available_cameras():
    """
    列出所有可用的摄像头
    """
    available = []
    for i in range(8):  # 检查前8个摄像头索引
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def get_camera_info(camera_id):
    """
    获取摄像头信息
    """
    cap = cv2.VideoCapture(camera_id + cv2.CAP_DSHOW)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return f"摄像头 {camera_id}: {width}x{height} @{fps}fps"
    return f"摄像头 {camera_id}: 不可用"


def initialize_model(model_path):
    """
    初始化模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BDCN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def adjust_image(image, contrast=1.0, brightness=0, denoise_strength=10):
    """
    增强图像预处理
    """
    # 去噪
    denoised = cv2.fastNlMeansDenoisingColored(image, None, denoise_strength,
                                               denoise_strength, 7, 21)

    # 调整对比度和亮度
    adjusted = cv2.convertScaleAbs(denoised, alpha=contrast, beta=brightness)

    # 增强对比度
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def enhance_edges(edges, threshold=30):
    """
    边缘后处理增强
    """
    # 自适应阈值处理
    binary = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    # 先腐蚀后膨胀，去除噪点
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # 闭运算连接断开的边缘
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return closed


def save_images(frame, edges, save_dir):
    """
    保存原始图片和边缘检测结果
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_path = os.path.abspath(os.path.join(save_dir, f'original_{timestamp}.png'))
        edges_path = os.path.abspath(os.path.join(save_dir, f'edges_{timestamp}.png'))

        if frame is None or edges is None:
            raise ValueError("图像数据无效")

        if len(edges.shape) == 2:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        success1 = cv2.imwrite(original_path, frame)
        success2 = cv2.imwrite(edges_path, edges)

        if not (success1 and success2):
            try:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                edges_pil = Image.fromarray(edges if len(edges.shape) == 2 else cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
                frame_pil.save(original_path)
                edges_pil.save(edges_path)
            except Exception as e:
                raise IOError(f"保存失败: {str(e)}")

        return original_path, edges_path

    except Exception as e:
        print(f"\n保存图片时发生错误: {str(e)}")
        raise


def process_frame(frame, model, device, transform, params):
    """
    处理单帧图像
    """
    if frame is None:
        return None

    original_h, original_w = frame.shape[:2]

    # 预处理增强
    enhanced_frame = adjust_image(frame, params['contrast'],
                                  params['brightness'],
                                  params['denoise_strength'])

    # 转换为灰度图
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用传统边缘检测作为补充
    canny = cv2.Canny(blurred, 50, 150)

    # 深度学习模型处理
    frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    edges = output[-1].squeeze().cpu().numpy()
    edges = (edges - edges.min()) / (edges.max() - edges.min()) * 255
    edges = edges.astype(np.uint8)
    edges = cv2.resize(edges, (original_w, original_h))

    # 组合深度学习和传统边缘检测结果
    combined_edges = cv2.addWeighted(edges, 0.7, canny, 0.3, 0)

    # 后处理增强
    enhanced_edges = enhance_edges(combined_edges, params['threshold'])

    return enhanced_edges


def analyze_chladni_patterns(edges):
    """
    分析克拉尼图像中的波节数
    返回：(n, k) 水平和垂直方向的波节数
    """
    # 转换为二值图像
    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    # 获取图像尺寸
    height, width = binary.shape

    # 计算水平方向的波节
    horizontal_profile = np.sum(binary, axis=0) / 255  # 垂直投影
    horizontal_profile_smooth = cv2.GaussianBlur(horizontal_profile.reshape(-1, 1), (1, 5), 0).flatten()

    # 计算垂直方向的波节
    vertical_profile = np.sum(binary, axis=1) / 255  # 水平投影
    vertical_profile_smooth = cv2.GaussianBlur(vertical_profile.reshape(-1, 1), (1, 5), 0).flatten()

    # 查找波节（通过寻找局部最小值）
    def count_nodes(profile):
        # 计算一阶导数
        gradient = np.gradient(profile)
        # 找到过零点（符号变化）
        zero_crossings = np.where(np.diff(np.signbit(gradient)))[0]
        # 过滤掉太接近的过零点（认为是噪声）
        min_distance = len(profile) // 20  # 最小间距阈值
        filtered_crossings = []
        last_crossing = -min_distance
        for crossing in zero_crossings:
            if crossing - last_crossing >= min_distance:
                filtered_crossings.append(crossing)
                last_crossing = crossing
        return len(filtered_crossings)

    n = count_nodes(horizontal_profile_smooth)  # 水平方向波节数
    k = count_nodes(vertical_profile_smooth)  # 垂直方向波节数

    return n, k


def visualize_chladni_analysis(frame, edges, n, k):
    """
    可视化克拉尼图像分析结果
    """
    # 创建结果展示图像
    height, width = frame.shape[:2]
    result = frame.copy()

    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Wave nodes - Horizontal: {n}, Vertical: {k}"
    cv2.putText(result, text, (10, height - 20), font, 0.7, (0, 255, 0), 2)

    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'saved_images(by_photo)')

    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        save_dir = os.path.join(os.path.expanduser('~'), 'edge_detection_images')
        os.makedirs(save_dir, exist_ok=True)
        print(f"使用备用保存目录: {save_dir}")

    available_cameras = list_available_cameras()
    if not available_cameras:
        print("未找到可用摄像头")
        return

    print("\n可用摄像头列表:")
    for cam_id in available_cameras:
        print(get_camera_info(cam_id))

    while True:
        try:
            camera_id = int(input(f"\n请选择摄像头 (0-{len(available_cameras) - 1}): "))
            if camera_id in available_cameras:
                break
            print("无效的摄像头编号，请重新选择")
        except ValueError:
            print("请输入有效的数字")

    model_path = './BDCN_pretrained/model/bdcn_pretrained_on_bsds500.pth'
    model, device = initialize_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(camera_id + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n操作说明:")
    print("按 's' 保存当前画面")
    print("按 'q' 退出程序")
    print("按 '+/-' 调整边缘检测灵敏度")
    print("按 'c/v' 调整对比度")
    print("按 'b/n' 调整亮度")
    print("按 'd/f' 调整去噪强度")

    # 参数初始化
    params = {
        'threshold': 30,
        'contrast': 1.0,
        'brightness': 0,
        'denoise_strength': 10
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法获取画面")
                break

            edges = process_frame(frame, model, device, transform, params)
            if edges is None:
                continue

            # 分析克拉尼图像
            n, k = analyze_chladni_patterns(edges)

            # 可视化分析结果
            result = visualize_chladni_analysis(frame, edges, n, k)

            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((result, edges_colored))

            cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Edge Detection', combined)

            # 添加实时显示波节数
            print(f"\r波节数 - 水平: {n}, 垂直: {k}", end='', flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n程序退出")
                break
            elif key == ord('s'):
                try:
                    print("\n正在保存图片...")
                    original_path, edges_path = save_images(frame, edges, save_dir)
                    # 保存波节数据
                    with open(os.path.join(save_dir, f'nodes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'),
                              'w') as f:
                        f.write(f"Horizontal nodes (n): {n}\nVertical nodes (k): {k}")
                    print("图片和波节数据保存完成！")
                except Exception as e:
                    print(f"\n保存失败: {str(e)}")
                    try:
                        backup_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'edge_detection_backup')
                        os.makedirs(backup_dir, exist_ok=True)
                        original_path, edges_path = save_images(frame, edges, backup_dir)
                        print("使用备用位置保存成功！")
                    except Exception as backup_error:
                        print(f"备用保存也失败了: {str(backup_error)}")
            elif key == ord('+'):
                params['threshold'] = min(params['threshold'] + 5, 255)
                print(f"提高灵敏度: {params['threshold']}")
            elif key == ord('-'):
                params['threshold'] = max(params['threshold'] - 5, 0)
                print(f"降低灵敏度: {params['threshold']}")
            elif key == ord('c'):
                params['contrast'] = min(params['contrast'] + 0.1, 3.0)
                print(f"提高对比度: {params['contrast']:.1f}")
            elif key == ord('v'):
                params['contrast'] = max(params['contrast'] - 0.1, 0.1)
                print(f"降低对比度: {params['contrast']:.1f}")
            elif key == ord('b'):
                params['brightness'] = min(params['brightness'] + 10, 100)
                print(f"提高亮度: {params['brightness']}")
            elif key == ord('n'):
                params['brightness'] = max(params['brightness'] - 10, -100)
                print(f"降低亮度: {params['brightness']}")
            elif key == ord('d'):
                params['denoise_strength'] = min(params['denoise_strength'] + 1, 20)
                print(f"增加去噪强度: {params['denoise_strength']}")
            elif key == ord('f'):
                params['denoise_strength'] = max(params['denoise_strength'] - 1, 0)
                print(f"降低去噪强度: {params['denoise_strength']}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n程序已正常关闭")


if __name__ == "__main__":
    main()
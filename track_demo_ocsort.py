import os
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from trackers.ocsort_tracker.ocsort import OCSort
from tqdm import tqdm
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pickle
from visualization_new import start_visualization_thread
from typing import Dict, List, Tuple, Any
from scipy import stats
import time
import sys
DEFAULT_SPEEDS: Dict[int, float] = {}
DEFAULT_QUALITIES: Dict[int, float] = {}
# Configuration dictionary with all settings
CONFIG = {
    # Video Processing
    'video_path': r"C:\Users\intel\Desktop\1000-1.mp4",
    'output_dir': r'D:\pythonproject\yzh\yolo11\1',
    'model_weights': r'D:\pythonproject\yzh\yolo11\yolo11s_large\exp\weights\best.pt',
    'device': '0',

    # Detection Parameters
    'confidence_threshold': 0.6,

    # Slice Configuration
    'slice_params': {
        'slice_height': 220,
        'slice_width': 220,
        'overlap_height_ratio': 0.4,
        'overlap_width_ratio': 0.4
    },

    # Processing Limits
    'max_frames': 600,
    'num_threads': 15,

    # Tracking Configuration
    'det_thresh': 0.8,
    'max_age': 120,
    'min_hits': 3,
    'iou_threshold': 0.3,
    'delta_t': 3,
    'inertia': 0.4,
    'asso_func': "diou",
    'use_byte': True,
    'direction': (0, -1),

    # File Management
    'save_detections': True,
    'load_detections': False,
    'detections_path': r'D:\pythonproject\yzh\yolo11\2',
    'save_video': True,
    'filtered_video_dir': r'D:\pythonproject\yzh\yolo11\3',

    # Visualization Settings
    'visualize_area_calculation': False,
    'pixel_to_um': 50/24.75,
    'visualize_frequency': 10,
    'detection_colors': [(255, 0, 0)],  # Will be updated by generate_color_table
    'bbox_thickness': 1,
    'text_scale': 0.1,
    'text_thickness': 1
}


# Global variables
TrackedPoint = Tuple[int, float, float, float, float, float, float]
tracked_trajectories: Dict[int, List[TrackedPoint]] = {}
tracked_speeds: Dict[int, float] = {}
tracked_qualities: Dict[int, float] = {}
average_directions: Dict[int, Tuple[float, float]] = {}

# Thread locks
trajectories_lock = threading.Lock()
speeds_lock = threading.Lock()
qualities_lock = threading.Lock()
directions_lock = threading.Lock()

# Environment settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def generate_color_table(n_colors: int = 100) -> List[Tuple[int, int, int]]:
    """
    Generate a rich color table for visualization.

    Args:
        n_colors: Number of unique colors to generate

    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(n_colors):
        # Generate colors in HSV space for better distribution
        h = i / n_colors
        s = 0.8 + np.random.random() * 0.2
        v = 0.8 + np.random.random() * 0.2

        # Convert to RGB then BGR
        rgb = tuple(round(x * 255) for x in plt.cm.hsv(h)[:-1])
        bgr = (rgb[2], rgb[1], rgb[0])
        colors.append(bgr)

    return colors


def init_detection_model(
        weights: str = CONFIG['model_weights'],
        device: str = CONFIG['device']
) -> AutoDetectionModel:
    """
    Initialize the object detection model.

    Args:
        weights: Path to model weights
        device: Device to run the model on ('cpu' or GPU index)

    Returns:
        Initialized detection model
    """
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weights,
        confidence_threshold=CONFIG['confidence_threshold'],
        device=device,
        image_size=640
    )


def save_detections(detections: np.ndarray, path: str) -> None:
    """
    Save detection results to a pickle file.

    Args:
        detections: Array of detection results
        path: Path to save the pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(detections, f)


def load_detections(path: str) -> np.ndarray:
    """
    Load detection results from a pickle file.

    Args:
        path: Path to the pickle file

    Returns:
        Array of detection results
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_video(frame_folder: str, video_path: str, fps: int = 30) -> None:
    """
    Generate video from a sequence of frames.
    """
    images = sorted([img for img in os.listdir(frame_folder) if img.endswith(".jpg")])
    if not images:
        print("No images found in the directory.")
        return

    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 生成颜色映射
    colors = generate_color_table(100)
    id_color_map = {}

    for image in images:
        frame = cv2.imread(os.path.join(frame_folder, image))
        frame_idx = int(image.split('_')[-1].split('.')[0])

        for obj_id, trajectory in tracked_trajectories.items():
            if obj_id not in id_color_map:
                id_color_map[obj_id] = colors[obj_id % len(colors)]

            current_points = [p for p in trajectory if p[0] == frame_idx]
            if current_points:
                point = current_points[0]
                x1, y1, x2, y2 = map(int, point[1:5])
                color = id_color_map[obj_id]

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                # 减小字体大小并调整位置
                label = f'{obj_id}'  # 简化标签，去掉'ID:'前缀
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 字体大小改为0.3，粗细改为1

        video.write(frame)

    video.release()


def plot_detection_results(
        img: np.ndarray,
        detections: np.ndarray,
        output_path: str
) -> None:
    """
    Visualize detection results on the image.

    Args:
        img: Input image
        detections: Array of detection results (x1, y1, x2, y2, score)
        output_path: Path to save the output image
    """
    img_copy = img.copy()

    for i, (x1, y1, x2, y2, score) in enumerate(detections):
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Get color from color table
        color = CONFIG['detection_colors'][i % len(CONFIG['detection_colors'])]

        # Draw bounding box
        cv2.rectangle(
            img_copy,
            (x1, y1),
            (x2, y2),
            color,
            thickness=CONFIG['bbox_thickness']
        )

        # Add confidence score label
        label = f'{score:.2f}'
        t_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            CONFIG['text_scale'],
            CONFIG['text_thickness']
        )[0]

        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1),
            (x1 + t_size[0], y1 - t_size[1] - 3),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            CONFIG['text_scale'],
            (255, 255, 255),
            CONFIG['text_thickness']
        )

    # Add total detection count
    cv2.putText(
        img_copy,
        f'Detections: {len(detections)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imwrite(output_path, img_copy)


def plot_img(
        img: np.ndarray,
        results: List[Tuple[float, float, float, float, int]],
        output_path: str
) -> None:
    """
    Plot tracking results on the image.

    Args:
        img: Input image
        results: List of tracking results (x1, y1, x2, y2, track_id)
        output_path: Path to save the output image
    """
    img_copy = img.copy()

    for x1, y1, x2, y2, track_id in results:
        # Generate unique color for each track ID
        color = tuple(map(int, (
            (37 * track_id) % 255,
            (17 * track_id) % 255,
            (29 * track_id) % 255
        )))

        # Draw bounding box
        cv2.rectangle(
            img_copy,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=1
        )

    cv2.imwrite(output_path, img_copy)


def detect_objects(
        frame: np.ndarray,
        detection_model: AutoDetectionModel,
        slice_params: Dict[str, Any]
) -> np.ndarray:
    """
    Perform object detection on a frame using the SAHI sliced prediction approach.

    Args:
        frame: Input frame in BGR format
        detection_model: Initialized detection model
        slice_params: Parameters for sliced prediction

    Returns:
        Array of detection results [x1, y1, x2, y2, score]
    """
    # Convert BGR to RGB for model input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform sliced prediction
    result = get_sliced_prediction(frame_rgb, detection_model, **slice_params)

    # Log detection statistics
    detection_count = len(result.object_prediction_list)
    print(f"Number of detections in current frame: {detection_count}")

    if detection_count == 0:
        print("Warning: No objects detected in this frame!")

    # Convert predictions to numpy array
    return np.array([
        [
            obj.bbox.minx,
            obj.bbox.miny,
            obj.bbox.maxx,
            obj.bbox.maxy,
            obj.score.value
        ]
        for obj in result.object_prediction_list
    ], dtype=np.float32)


def visualize_area_calculation(
        frame: np.ndarray,
        bbox: np.ndarray,
        area: float
) -> None:
    """
    Visualize the area calculation process for a detected object.

    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        area: Calculated area value
    """
    # Extract ROI and convert to binary image
    x1, y1, x2, y2 = map(int, bbox[:4])
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Process connected components
    white_areas = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_areas, connectivity=8)
    largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create visualization plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Object Area Calculation (Area: {area:.2f})')

    # Original ROI
    axes[0, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original ROI')
    axes[0, 0].axis('off')

    # Grayscale image
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')

    # Binary image
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Image (Otsu)')
    axes[1, 0].axis('off')

    # Largest connected component
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    color_img[labels == largest_component_index] = [255, 0, 0]
    axes[1, 1].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Largest White Area (Blue)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_object_area(
        frame: np.ndarray,
        bbox: np.ndarray,
        visualize: bool = False
) -> float:
    """
    Calculate the area of a detected object using image processing.

    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        visualize: Whether to show visualization of the calculation process

    Returns:
        Calculated area of the object
    """
    try:
        # Convert coordinates to integers and ensure they're within image bounds
        x1, y1, x2, y2 = map(int, bbox[:4])
        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # Validate ROI coordinates
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bbox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return 0

        # Extract and validate ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"Warning: Empty ROI for bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return 0

        # Process image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_areas = (binary == 0).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_areas, connectivity=8)

        if num_labels <= 1:
            return 0

        # Get largest connected component area
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return 0

        largest_area = np.max(areas)

        # Visualize if requested
        if visualize:
            visualize_area_calculation(frame, bbox, largest_area)

        return largest_area

    except Exception as e:
        print(f"Error in calculate_object_area: {str(e)}")
        print(f"bbox: {bbox}")
        return 0


def calculate_most_common_area(areas: List[float]) -> float:
    """
    Calculate the most common object area using kernel density estimation.

    Args:
        areas: List of calculated areas

    Returns:
        Most common area value
    """
    # Estimate density distribution using KDE
    kde = stats.gaussian_kde(areas)

    # Create evaluation points
    x_range = np.linspace(min(areas), max(areas), 1000)

    # Calculate density values
    density = kde(x_range)

    # Find the area with maximum density
    most_common_area = x_range[np.argmax(density)]

    return most_common_area


def estimate_object_count(area: float, single_object_area: float) -> int:
    """
    Estimate the number of objects based on area comparison.

    Args:
        area: Total area to evaluate
        single_object_area: Reference area for a single object

    Returns:
        Estimated number of objects
    """
    if area < single_object_area:
        return 1
    else:
        return max(1, round(area / single_object_area))


def preprocess_trajectories(
        trajectories: Dict[int, List[TrackedPoint]],
        min_length: int = 5,
        smooth_window: int = 3
) -> Dict[int, List[TrackedPoint]]:
    """
    Preprocess trajectories with smoothing while maintaining complete bounding box information.

    Args:
        trajectories: Dictionary mapping object IDs to tracking points
        min_length: Minimum trajectory length to keep
        smooth_window: Window size for coordinate smoothing

    Returns:
        Dictionary of processed trajectories
    """
    processed_trajectories = {}

    for obj_id, points in trajectories.items():
        # Skip short trajectories
        if len(points) < min_length:
            continue

        processed_points = []
        for i in range(len(points)):
            # Define smoothing window
            window_start = max(0, i - smooth_window + 1)
            window_points = points[window_start:i + 1]

            # Get original frame index
            frame_idx = points[i][0]

            # Calculate smoothed coordinates
            smoothed_x1 = np.mean([p[1] for p in window_points])
            smoothed_y1 = np.mean([p[2] for p in window_points])
            smoothed_x2 = np.mean([p[3] for p in window_points])
            smoothed_y2 = np.mean([p[4] for p in window_points])

            # Calculate smoothed center coordinates
            smoothed_center_x = (smoothed_x1 + smoothed_x2) / 2
            smoothed_center_y = (smoothed_y1 + smoothed_y2) / 2

            # Create processed point with smoothed coordinates
            processed_point = (
                frame_idx,
                smoothed_x1,
                smoothed_y1,
                smoothed_x2,
                smoothed_y2,
                smoothed_center_x,
                smoothed_center_y
            )

            processed_points.append(processed_point)

        processed_trajectories[obj_id] = processed_points

    return processed_trajectories


def calculate_displacement_and_speed(
        trajectories: Dict[int, List[TrackedPoint]],
        direction: Tuple[float, float],
        frame_rate: float
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Calculate displacement, speed, and direction metrics for tracked objects.

    Args:
        trajectories: Dictionary mapping object IDs to tracking points
        direction: Reference direction vector for speed calculation
        frame_rate: Video frame rate for time calculations

    Returns:
        Tuple of (speeds, qualities, average_directions) dictionaries
    """
    speeds = {}
    qualities = {}
    average_directions = {}
    specified_dir_vector = np.array(direction) / np.linalg.norm(direction)
    pixel_to_um = CONFIG['pixel_to_um']

    for obj_id, points in trajectories.items():
        if len(points) > 1:
            # Extract positions and times
            positions = np.array([(p[5], p[6]) for p in points])
            times = np.array([p[0] for p in points]) / frame_rate

            # Convert positions to micrometers
            positions_um = positions * pixel_to_um

            # Calculate displacements and time differences
            displacements = np.diff(positions_um, axis=0)
            time_diffs = np.diff(times)

            # Calculate total displacement and time
            total_displacement = np.sum(displacements, axis=0)
            total_time = np.sum(time_diffs)

            if total_time > 0:
                # Calculate speed metrics (in μm/s)
                average_speed = np.linalg.norm(total_displacement) / total_time
                average_direction = (
                    total_displacement / np.linalg.norm(total_displacement)
                    if np.linalg.norm(total_displacement) > 0
                    else np.array([0, 0])
                )
                speed_in_specified_direction = (
                        average_speed * np.dot(average_direction, specified_dir_vector)
                )

                # Calculate instantaneous speeds (in μm/s)
                instant_speeds = np.linalg.norm(displacements, axis=1) / time_diffs
                speed_variance = np.var(instant_speeds) if len(instant_speeds) > 0 else 0

                # Store results
                speeds[obj_id] = speed_in_specified_direction
                average_directions[obj_id] = tuple(average_direction)

                # Calculate trajectory quality using positions in micrometers
                center_point_trajectory = [(p[0], p[5] * pixel_to_um, p[6] * pixel_to_um) for p in points]
                qualities[obj_id] = calculate_trajectory_quality(
                    center_point_trajectory,
                    speed_variance
                )
            else:
                speeds[obj_id] = 0
                average_directions[obj_id] = (0, 0)
                qualities[obj_id] = 0

    return speeds, qualities, average_directions


def calculate_trajectory_quality(
        trajectory: List[Tuple[int, float, float]],
        speed_variance: float,
        min_length: int = 5,
        max_length: int = 100
) -> float:
    """
    Calculate quality score for a trajectory based on multiple metrics.

    Args:
        trajectory: List of (frame_idx, center_x, center_y) tuples
        speed_variance: Variance of instantaneous speeds
        min_length: Minimum trajectory length for quality calculation
        max_length: Maximum trajectory length for normalization

    Returns:
        Quality score between 0 and 1
    """
    if len(trajectory) < 2:
        return 0.0

    # Extract positions for calculations
    positions = np.array([(p[1], p[2]) for p in trajectory])

    # Calculate displacement metrics
    displacements = np.diff(positions, axis=0)
    total_length = np.sum(np.linalg.norm(displacements, axis=1))
    start_end_distance = np.linalg.norm(positions[-1] - positions[0])

    # Calculate quality factors
    straightness = start_end_distance / total_length if total_length > 0 else 0
    length_factor = np.clip((len(trajectory) - min_length) / (max_length - min_length), 0, 1)
    consistency_factor = 1 / (1 + speed_variance)

    # Calculate acceleration-based factors
    accelerations = calculate_accelerations(trajectory)
    acceleration_factor = 1 / (1 + np.var(accelerations))

    # Calculate time-based factor
    time_duration = trajectory[-1][0] - trajectory[0][0]
    time_factor = min(time_duration / 300, 1)

    # Define factor weights
    weights = {
        'straightness': 0.3,
        'length': 0.2,
        'consistency': 0.2,
        'acceleration': 0.2,
        'time': 0.1
    }

    # Calculate weighted quality score
    quality = sum(
        weight * factor for weight, factor in zip(
            weights.values(),
            [straightness, length_factor, consistency_factor,
             acceleration_factor, time_factor]
        )
    )

    return quality


def calculate_accelerations(
        trajectory: List[Tuple[int, float, float]]
) -> np.ndarray:
    """
    Calculate acceleration values from trajectory points.

    Args:
        trajectory: List of (frame_idx, center_x, center_y) tuples

    Returns:
        Array of acceleration magnitudes
    """
    # Extract times and positions
    times = np.array([p[0] for p in trajectory])
    positions = np.array([(p[1], p[2]) for p in trajectory])

    # Calculate velocities
    velocities = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]

    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0) / np.diff(times[1:])[:, np.newaxis]

    return np.linalg.norm(accelerations, axis=1)


def get_background_color(video_path: str, sample_frames: int = 30) -> Tuple[int, int, int]:
    """
    获取视频的主要背景色

    Args:
        video_path: 视频路径
        sample_frames: 采样帧数

    Returns:
        BGR颜色元组
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // sample_frames)

    # 收集采样帧的平均颜色
    colors = []
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # 计算帧的平均颜色
            avg_color = np.mean(frame, axis=(0, 1))
            colors.append(avg_color)

    cap.release()

    # 返回中值颜色作为背景色
    if colors:
        median_color = np.median(colors, axis=0)
        return tuple(map(int, median_color))
    return (0, 0, 0)  # 默认黑色


def generate_filtered_video_full_trajectory(
        trajectories: Dict[int, List[TrackedPoint]],
        filtered_ids: List[int],
        output_dir: str,
        video_path: str,
        original_video_path: str
) -> None:
    """
    生成显示累积轨迹的视频版本（无ID）:
    1. 完整视频：显示所有累积轨迹和目标，无ID
    2. 目标视频：在新背景下只显示过滤后的累积轨迹和目标，无ID

    轨迹会随着每一帧逐渐累积显示，而不是一次性显示完整轨迹。

    Args:
        trajectories: 跟踪轨迹字典
        filtered_ids: 过滤后的目标ID列表
        output_dir: 输出目录
        video_path: 完整视频的保存路径
        original_video_path: 原始视频路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频属性
    cap = cv2.VideoCapture(original_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取背景色
    bg_color = get_background_color(original_video_path)

    # 创建新的文件名（累积轨迹版本）
    video_name = os.path.basename(video_path)
    params_part = video_name.replace('filtered_video_', '')
    full_traj_video_name = f'full_trajectory_complete_{params_part}'
    full_traj_target_video_name = f'full_trajectory_targets_only_{params_part}'

    full_traj_path = os.path.join(output_dir, full_traj_video_name)
    full_traj_target_path = os.path.join(output_dir, full_traj_target_video_name)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_full = cv2.VideoWriter(full_traj_path, fourcc, fps, (frame_width, frame_height))
    out_targets = cv2.VideoWriter(full_traj_target_path, fourcc, fps, (frame_width, frame_height))

    # 找到最大帧号
    max_frame = max(max(point[0] for point in traj) for traj in trajectories.values())

    print(f"正在生成累积轨迹视频（无ID）,包含 {len(filtered_ids)} 个目标...")

    # 创建累积轨迹字典
    accumulated_trajectories = {obj_id: [] for obj_id in filtered_ids}

    # 处理每一帧
    for frame_idx in tqdm(range(max_frame + 1), desc="处理帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # 创建两个帧
        full_frame = frame.copy()
        target_frame = np.full((frame_height, frame_width, 3), bg_color, dtype=np.uint8)

        # 创建遮罩
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # 更新和绘制轨迹
        for obj_id in filtered_ids:
            if obj_id in trajectories:
                # 获取当前帧之前的所有点（包括当前帧）
                current_and_previous_points = [
                    p for p in trajectories[obj_id]
                    if p[0] <= frame_idx
                ]

                if current_and_previous_points:
                    # 为每个目标生成唯一的颜色
                    color = tuple(map(int, (
                        (37 * obj_id) % 255,
                        (17 * obj_id) % 255,
                        (29 * obj_id) % 255
                    )))

                    # 更新累积轨迹
                    latest_point = current_and_previous_points[-1]
                    accumulated_trajectories[obj_id].append(
                        (int(latest_point[5]), int(latest_point[6]))
                    )

                    # 绘制累积的轨迹
                    points = accumulated_trajectories[obj_id]
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            pt1 = points[i - 1]
                            pt2 = points[i]
                            cv2.line(full_frame, pt1, pt2, color, 2)
                            cv2.line(target_frame, pt1, pt2, color, 2)

                    # 处理当前帧的边界框
                    if current_and_previous_points[-1][0] == frame_idx:
                        current_point = current_and_previous_points[-1]
                        x1, y1, x2, y2 = map(int, current_point[1:5])

                        # 确保坐标在有效范围内
                        x1 = max(0, min(x1, frame_width - 1))
                        x2 = max(0, min(x2, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        y2 = max(0, min(y2, frame_height - 1))

                        # 在遮罩上标记目标区域
                        mask[y1:y2, x1:x2] = 255

                        # 只绘制边界框，不添加ID标签
                        cv2.rectangle(full_frame, (x1, y1), (x2, y2), color, 1)
                        cv2.rectangle(target_frame, (x1, y1), (x2, y2), color, 1)

        # 使用遮罩只复制过滤后目标的区域
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        target_frame = np.where(mask[:, :, np.newaxis] > 0, masked_frame, target_frame)

        # 写入两个视频
        out_full.write(full_frame)
        out_targets.write(target_frame)

    # 释放资源
    cap.release()
    out_full.release()
    out_targets.release()

    print(f"累积轨迹完整视频（无ID）已保存到: {full_traj_path}")
    print(f"累积轨迹目标视频（无ID）已保存到: {full_traj_target_path}")


def generate_filtered_video(
        trajectories: Dict[int, List[TrackedPoint]],
        filtered_ids: List[int],
        output_dir: str,
        video_path: str,
        original_video_path: str,
        speeds: Dict[int, float] = None
) -> None:
    """
    生成两种过滤后的视频:
    1. 完整视频：显示所有轨迹和目标，无标签
    2. 目标视频：在新背景下只显示过滤后的目标，无标签

    Args:
        trajectories: 跟踪轨迹字典
        filtered_ids: 过滤后的目标ID列表
        output_dir: 输出目录
        video_path: 完整视频的保存路径
        original_video_path: 原始视频路径
        speeds: 对象速度字典（可选）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频属性
    cap = cv2.VideoCapture(original_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取背景色
    bg_color = get_background_color(original_video_path)

    # 从video_path中提取参数部分
    video_name = os.path.basename(video_path)
    params_part = video_name.replace('filtered_video_', '')  # 移除前缀
    target_video_name = f'targets_only_{params_part}'  # 添加targets_only前缀
    target_only_path = os.path.join(output_dir, target_video_name)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_full = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    out_targets = cv2.VideoWriter(target_only_path, fourcc, fps, (frame_width, frame_height))

    # 找到最大帧号
    max_frame = max(max(point[0] for point in traj) for traj in trajectories.values())

    print(f"正在生成过滤后的视频（无标签）,包含 {len(filtered_ids)} 个目标...")

    # 创建轨迹历史记录字典
    trajectory_history = {obj_id: [] for obj_id in filtered_ids}

    # 处理每一帧
    for frame_idx in tqdm(range(max_frame + 1), desc="处理帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # 创建两个帧
        full_frame = frame.copy()
        target_frame = np.full((frame_height, frame_width, 3), bg_color, dtype=np.uint8)

        # 创建遮罩
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # 绘制过滤后的轨迹和目标
        for obj_id in filtered_ids:
            if obj_id in trajectories:
                points = [p for p in trajectories[obj_id] if p[0] <= frame_idx]

                if points:
                    # 为每个目标生成唯一的颜色
                    color = tuple(map(int, (
                        (37 * obj_id) % 255,
                        (17 * obj_id) % 255,
                        (29 * obj_id) % 255
                    )))

                    # 更新轨迹历史
                    current_point = points[-1]
                    trajectory_history[obj_id].append((
                        int(current_point[5]),
                        int(current_point[6])
                    ))

                    # 保持轨迹历史的最大长度
                    if len(trajectory_history[obj_id]) > 100:
                        trajectory_history[obj_id] = trajectory_history[obj_id][-100:]

                    # 在两个帧上绘制轨迹线
                    for i in range(1, len(trajectory_history[obj_id])):
                        pt1 = trajectory_history[obj_id][i - 1]
                        pt2 = trajectory_history[obj_id][i]
                        cv2.line(full_frame, pt1, pt2, color, 1)
                        cv2.line(target_frame, pt1, pt2, color, 1)

                    # 获取当前边界框
                    x1, y1, x2, y2 = map(int, current_point[1:5])

                    # 确保坐标在有效范围内
                    x1 = max(0, min(x1, frame_width - 1))
                    x2 = max(0, min(x2, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))
                    y2 = max(0, min(y2, frame_height - 1))

                    # 在遮罩上标记目标区域
                    mask[y1:y2, x1:x2] = 255

                    # 在两个视频中绘制边界框（无标签）
                    cv2.rectangle(full_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.rectangle(target_frame, (x1, y1), (x2, y2), color, 1)

                    # 注释掉所有标签相关代码
                    # # 添加目标ID和速度标签
                    # speed_text = f" S:{speeds[obj_id]:.1f}" if speeds and obj_id in speeds else ""
                    # label = f'ID:{obj_id}{speed_text}'
                    #
                    # # 调整标签位置和字体大小以适应新的内容
                    # font_scale = 0.4  # 稍微减小字体大小以适应更多文本
                    # font_thickness = 1
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    #
                    # # 计算文本大小以更好地放置标签
                    # (text_width, text_height), baseline = cv2.getTextSize(
                    #     label, font, font_scale, font_thickness
                    # )
                    #
                    # # 在边界框上方绘制标签背景
                    # label_bg_x1 = x1
                    # label_bg_y1 = y1 - text_height - baseline - 4
                    # label_bg_x2 = x1 + text_width + 4
                    # label_bg_y2 = y1
                    #
                    # # 确保标签在图像范围内
                    # if label_bg_y1 >= 0:
                    #     # 绘制半透明背景
                    #     overlay = full_frame.copy()
                    #     cv2.rectangle(overlay,
                    #                   (label_bg_x1, label_bg_y1),
                    #                   (label_bg_x2, label_bg_y2),
                    #                   color, -1)
                    #     cv2.addWeighted(overlay, 0.5, full_frame, 0.5, 0, full_frame)
                    #
                    #     # 绘制文本
                    #     cv2.putText(full_frame, label,
                    #                 (x1 + 2, y1 - 4),
                    #                 font, font_scale, (255, 255, 255), font_thickness)
                    #
                    #     # 对目标视频做同样的处理
                    #     overlay = target_frame.copy()
                    #     cv2.rectangle(overlay,
                    #                   (label_bg_x1, label_bg_y1),
                    #                   (label_bg_x2, label_bg_y2),
                    #                   color, -1)
                    #     cv2.addWeighted(overlay, 0.5, target_frame, 0.5, 0, target_frame)
                    #     cv2.putText(target_frame, label,
                    #                 (x1 + 2, y1 - 4),
                    #                 font, font_scale, (255, 255, 255), font_thickness)

        # 使用遮罩只复制过滤后目标的区域
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        target_frame = np.where(mask[:, :, np.newaxis] > 0, masked_frame, target_frame)

        # 写入两个视频
        out_full.write(full_frame)
        out_targets.write(target_frame)

    # 释放资源
    cap.release()
    out_full.release()
    out_targets.release()

    print(f"完整轨迹视频（无标签）已保存到: {video_path}")
    print(f"目标视频（无标签）已保存到: {target_only_path}")

def plot_trajectories(
        trajectories: Dict[int, List[TrackedPoint]],
        filtered_ids: List[int],
        save_dir: str,
        speeds: Dict[int, float] = None,
        qualities: Dict[int, float] = None,
        dpi_setting: int = 150
) -> None:
    """
    优化后的轨迹可视化函数，无标签。

    Args:
        trajectories: 将对象 ID 映射到跟踪点列表的字典。
        filtered_ids: 通过了过滤条件的对象 ID 列表。
        save_dir: 可视化结果将保存到的目录。
        speeds: 对象速度字典（可选）。
        qualities: 轨迹质量字典（可选）。
        dpi_setting: 保存图像文件的 DPI 设置。
    """
    # 如果传入 None，则使用默认的空字典
    if speeds is None:
        speeds = DEFAULT_SPEEDS
    if qualities is None:
        qualities = DEFAULT_QUALITIES

    try:
        plt.figure(figsize=(15, 10))
        ax = plt.gca()
        # 使用稍浅的背景色，可能与灰色线条对比更佳
        ax.set_facecolor('#fafafa')
        # 使用 ':' 点状网格线
        plt.grid(True, linestyle=':', alpha=0.6, color='grey')

        x_coords_all = []
        y_coords_all = []

        # --- 绘图循环 ---
        for obj_id, points in trajectories.items():
            # 至少需要两个点才能绘制线条
            if not points or len(points) < 2:
                continue

            try:
                # 在访问索引前，确保点具有预期的结构
                if len(points[0]) < 7:
                    print(f"警告：跳过轨迹 {obj_id}，因为数据点结构异常。")
                    continue

                # 使用 numpy 高效提取坐标
                points_arr = np.array(points)
                x_coords = points_arr[:, 5] # 中心 X
                y_coords = points_arr[:, 6] # 中心 Y

                x_coords_all.extend(x_coords)
                y_coords_all.extend(y_coords)

                if obj_id in filtered_ids:
                    # --- 过滤后的轨迹 (高亮显示) ---
                    # 使用标准颜色映射表
                    color = plt.cm.tab20(obj_id % 20)

                    # 优化：移除了每个点的 'o' 标记，线条稍细
                    plt.plot(x_coords, y_coords, '-',
                             color=color, alpha=0.9,
                             linewidth=1.5, label=f'ID {obj_id}') # 不再有 marker='o'

                    # 保留数量较少的起点和终点标记
                    plt.plot(x_coords[0], y_coords[0], '^', color='green', # 简化标记调用
                             markersize=8, alpha=0.8, markeredgecolor='black', markeredgewidth=0.5)
                    plt.plot(x_coords[-1], y_coords[-1], 's', color='red', # 简化标记调用
                             markersize=8, alpha=0.8, markeredgecolor='black', markeredgewidth=0.5)

                    # 注释掉标签相关代码 - 去掉目标右上角的标签
                    # # 添加注释 (仅对过滤后的轨迹)
                    # if obj_id in speeds and obj_id in qualities:
                    #     speed = speeds[obj_id]
                    #     quality = qualities[obj_id]
                    #     # 将注释放置在终点稍微偏移的位置
                    #     plt.annotate(
                    #         f'ID:{obj_id} S:{speed:.1f} Q:{quality:.2f}',
                    #         (x_coords[-1], y_coords[-1]),
                    #         xytext=(8, -8), # 向右下方偏移一点
                    #         textcoords='offset points',
                    #         ha='left', va='top', # 调整对齐方式
                    #         bbox=dict(boxstyle='round,pad=0.3', # 简化填充
                    #                   fc='yellow',
                    #                   alpha=0.6), # 更透明一些
                    #         fontsize=7 # 更小的字体
                    #     )
                else:
                    # --- 未过滤的轨迹 (淡化显示) ---
                    # 优化：简化绘图 - 更细、更透明的实线
                    # 实线通常比虚线渲染更快
                    plt.plot(x_coords, y_coords, '-',
                             color='grey',
                             alpha=0.15, # 更透明
                             linewidth=0.5) # 更细

            except IndexError:
                 print(f"警告：处理轨迹 {obj_id} 时出现 IndexError。数据点结构可能不一致。")
                 continue
            except Exception as e:
                print(f"绘制轨迹 {obj_id} 时出错: {str(e)}")
                continue
        # --- 绘图循环结束 ---

        # 设置带有边距的绘图范围
        if x_coords_all and y_coords_all:
            x_min, x_max = min(x_coords_all), max(x_coords_all)
            y_min, y_max = min(y_coords_all), max(y_coords_all)
            x_padding = (x_max - x_min) * 0.05 # 稍微减少边距
            y_padding = (y_max - y_min) * 0.05
            plt.xlim(x_min - x_padding, x_max + x_padding)
            # 假设是典型的图像坐标系 (Y=0 在顶部)
            plt.ylim(y_max + y_padding, y_min - y_padding) # 反转 Y 轴
        else:
            # 如果没有绘制数据，则使用默认范围
            plt.xlim(0, 100)
            plt.ylim(100, 0)


        # 自定义图表外观
        plt.title(f'对象轨迹 ({len(filtered_ids)} 条已过滤)', fontsize=14, pad=15)
        plt.xlabel('X 坐标 (像素)', fontsize=10)
        plt.ylabel('Y 坐标 (像素)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8) # 更小的刻度标签

        # 仅当有需要显示的已过滤 ID 时才添加图例
        if filtered_ids:
            # 将图例放在绘图区域外部以避免重叠
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8)

        # 统计信息注释
        plt.text(
            0.01, 0.99, # 位置稍微向内
            f'总轨迹数: {len(trajectories)}\n已过滤: {len(filtered_ids)}',
            transform=ax.transAxes,
            ha='left', va='top', # 左上对齐
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7),
            fontsize=8 # 更小的字体
        )

        # 调整布局以防止标签/图例被截断 (保存前)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # 在右侧为图例留出空间

        # 保存图像
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'enhanced_trajectories.png')

        # 优化：使用降低的 DPI 和可能更简单的保存参数
        plt.savefig(
            save_path,
            dpi=dpi_setting, # 使用参数值
            # bbox_inches='tight', # 'tight' 会增加计算量，如果仍然很慢可以尝试去掉
            pad_inches=0.1,
            facecolor=ax.get_facecolor(), # 匹配坐标轴背景色
            format='png'
        )
        plt.close() # 关闭图形以释放内存
        print(f"已将优化后的轨迹图（无标签）保存至 {save_path} (DPI: {dpi_setting})")

    except Exception as e:
        print(f"生成或保存轨迹图失败: {str(e)}")
        traceback.print_exc()
        plt.close() # 即使出错也要确保关闭图形


def process_video(
        video_path: str,
        output_dir: str,
        model_weights: str,
        device: str,
        slice_params: Dict[str, Any],
        visualize_area: bool = False,
        visualize_frequency: int = 100
) -> Tuple[Dict[int, float], float]:
    """
    Process video for object detection and tracking.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        model_weights: Path to model weights
        device: Device for model inference
        slice_params: Parameters for sliced prediction
        visualize_area: Flag to enable area calculation visualization
        visualize_frequency: Frequency of visualization updates

    Returns:
        Tuple of object areas dictionary and most common area value
    """
    print("Initializing detection model...")
    detection_model = init_detection_model(model_weights, device)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames_to_process = min(
        CONFIG['max_frames'] if CONFIG['max_frames'] is not None else total_frames,
        total_frames
    )
    print(f"Processing {frames_to_process} frames...")

    # Initialize tracker
    tracker = OCSort(
        det_thresh=CONFIG['det_thresh'],
        max_age=CONFIG['max_age'],
        min_hits=CONFIG['min_hits'],
        iou_threshold=CONFIG['iou_threshold'],
        delta_t=CONFIG['delta_t'],
        asso_func=CONFIG['asso_func'],
        inertia=CONFIG['inertia'],
        use_byte=CONFIG['use_byte']
    )

    # Initialize tracking variables
    object_areas = {}
    all_areas = []
    total_detections = 0
    frames_with_detections = 0
    tracking_count = 0

    # Step 1: Parallel Detection
    with ThreadPoolExecutor(max_workers=CONFIG['num_threads']) as executor:
        futures = []
        for frame_idx in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                print(f"Can't receive frame (stream end?). Exiting at frame {frame_idx}")
                break

            if frame is None or frame.size == 0:
                print(f"Invalid frame at index {frame_idx}")
                continue

            detection_path = os.path.join(CONFIG['detections_path'], f'detection_{frame_idx}.pkl')

            if CONFIG['load_detections'] and os.path.exists(detection_path):
                future = executor.submit(load_detections, detection_path)
                print(f"Loading detections from {detection_path}")
            else:
                future = executor.submit(detect_objects, frame, detection_model, slice_params)

            futures.append((frame_idx, future, frame))

        # Process detection results
        detections_list = []
        for frame_idx, future, frame in tqdm(futures, total=len(futures), desc="Detecting objects"):
            try:
                detections = future.result()
                if detections is None or len(detections) == 0:
                    print(f"No detections in frame {frame_idx}")
                    continue

                num_detections = len(detections)
                total_detections += num_detections
                if num_detections > 0:
                    frames_with_detections += 1
                print(f"Frame {frame_idx}: {num_detections} objects detected")

                if CONFIG['save_detections']:
                    os.makedirs(CONFIG['detections_path'], exist_ok=True)
                    save_detections(detections, os.path.join(CONFIG['detections_path'],
                                                             f'detection_{frame_idx}.pkl'))

                output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.jpg')
                plot_detection_results(frame, detections, output_path)

                detections_list.append((frame_idx, detections, frame))

            except Exception as e:
                print(f"Error detecting objects in frame {frame_idx}: {str(e)}")
                traceback.print_exc()

    # Print detection statistics
    print("\n=== Detection Statistics ===")
    print(f"Total frames processed: {len(futures)}")
    print(f"Total objects detected: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Average detections per frame: {total_detections / max(len(futures), 1):.2f}")
    print(f"Detection rate: {(frames_with_detections / max(len(futures), 1)) * 100:.2f}%")
    print("=========================\n")

    # Step 2: Sequential Tracking
    for frame_idx, detections, frame in tqdm(detections_list, desc="Tracking objects"):
        try:
            if frame is None or frame.size == 0:
                print(f"Invalid frame at tracking stage, frame_idx: {frame_idx}")
                continue

            # Update tracker
            height, width = frame.shape[:2]

            # Update OCSort tracker with consistent dimensions
            tracked_objects = tracker.update(
                detections,  # shape: (N, 5) [x1,y1,x2,y2,score]
                (height, width),  # original image size
                (height, width)  # target image size - keeping consistent
            )

            print(f"Frame {frame_idx}: Tracking {len(tracked_objects)} objects")
            tracking_count += len(tracked_objects)

            # Update trajectories
            with trajectories_lock:
                for obj in tracked_objects:
                    try:
                        x1, y1, x2, y2, obj_id = obj
                        obj_id = int(obj_id - 1)  # Convert 1-based to 0-based ID

                        # Simple coordinate validation
                        if not (0 <= x1 < width and 0 <= y1 < height and
                                0 <= x2 < width and 0 <= y2 < height and
                                x1 < x2 and y1 < y2):
                            continue

                        # Calculate center points
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        # Update trajectory
                        tracked_trajectories.setdefault(obj_id, []).append(
                            (frame_idx, x1, y1, x2, y2, x_center, y_center)
                        )

                        # Calculate object area
                        should_visualize = visualize_area and frame_idx % visualize_frequency == 0
                        try:
                            area = calculate_object_area(
                                frame,
                                [x1, y1, x2, y2],
                                visualize=should_visualize
                            )
                            # 如果新检测的面积为0，使用原来的面积
                            if area == 0:
                                area = object_areas.get(obj_id, 1.5)  # 如果原来也没有，使用默认值1.5

                            object_areas[obj_id] = area
                            all_areas.append(area)
                        except Exception as e:
                            print(f"Error calculating area for object {obj_id}: {str(e)}")
                            traceback.print_exc()
                            # 发生异常时也使用原来的面积或默认值
                            area = object_areas.get(obj_id, 1.5)
                            object_areas[obj_id] = area
                            all_areas.append(area)

                    except Exception as e:
                        print(f"Error processing tracked object: {str(e)}")
                        traceback.print_exc()
                        continue

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            traceback.print_exc()
            continue

    # Print tracking statistics
    print("\n=== Tracking Statistics ===")
    if detections_list:
        print(f"Average tracked objects per frame: {tracking_count / len(detections_list):.2f}")
    print(f"Total unique object IDs: {len(tracked_trajectories)}")
    print(f"Total object areas calculated: {len(object_areas)}")
    print("=========================\n")

    cap.release()

    # Calculate most common area
    most_common_area = 0
    try:
        most_common_area = calculate_most_common_area(all_areas) if all_areas else 0
        print(f"Most common object area: {most_common_area:.2f}")
    except Exception as e:
        print(f"Error calculating most common area: {str(e)}")

    # Save output video
    if CONFIG['save_video']:
        try:
            save_video(output_dir, os.path.join(output_dir, 'output_video.mp4'), fps)
            print(f"Video saved to {os.path.join(output_dir, 'output_video.mp4')}")
        except Exception as e:
            print(f"Error saving output video: {str(e)}")

    return object_areas, most_common_area


def main() -> None:
    """Main function to run the object tracking pipeline."""
    try:
        # 创建所需的目录
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        os.makedirs(CONFIG['filtered_video_dir'], exist_ok=True)
        os.makedirs(CONFIG['detections_path'], exist_ok=True)

        # 第一步：处理视频和计算轨迹
        print("\n=== 开始处理视频和计算轨迹 ===")
        object_areas, single_object_area = process_video(
            CONFIG['video_path'],
            CONFIG['output_dir'],
            CONFIG['model_weights'],
            CONFIG['device'],
            CONFIG['slice_params'],
            visualize_area=CONFIG['visualize_area_calculation'],
            visualize_frequency=CONFIG['visualize_frequency']
        )

        print(f"\n估计的单个目标面积: {single_object_area:.2f}")

        # 第二步：处理轨迹数据
        print("\n=== 开始处理轨迹数据 ===")
        print("预处理轨迹...")
        preprocessed_trajectories = preprocess_trajectories(tracked_trajectories)

        print("计算速度、质量和平均方向...")
        speeds, qualities, average_directions = calculate_displacement_and_speed(
            preprocessed_trajectories,
            CONFIG['direction'],
            frame_rate=30
        )

        print("启动可视化线程...")
        start_visualization_thread(
            preprocessed_trajectories,
            speeds,
            qualities,
            average_directions,
            object_areas,
            single_object_area
        )

        # 第三步：交互式过滤和视频生成循环
        while True:
            print("\n=== 生成过滤后的视频 ===")
            print("提示: 输入 'q' 退出程序，或输入新的阈值继续生成视频")
            print("当前参数说明:")
            print("- 速度阈值: 用于过滤运动速度低于该值的目标")
            print("- 质量阈值: 取值范围0-1，用于过滤轨迹质量低于该值的目标")

            # 获取速度阈值
            speed_input = input("\n请输入速度阈值 (或 'q' 退出): ")
            if speed_input.lower() == 'q':
                break

            # 获取质量阈值
            quality_input = input("请输入质量阈值 (0-1) (或 'q' 退出): ")
            if quality_input.lower() == 'q':
                break

            try:
                speed_threshold = float(speed_input)
                quality_threshold = float(quality_input)

                # 验证输入值的有效性
                if quality_threshold < 0 or quality_threshold > 1:
                    raise ValueError("质量阈值必须在0到1之间")

            except ValueError as e:
                print(f"错误: {str(e)}")
                print("请输入有效的数值\n")
                continue

            # 基于阈值过滤轨迹
            filtered_ids = [
                obj_id for obj_id, speed in speeds.items()
                if speed > speed_threshold and qualities.get(obj_id, 0) > quality_threshold
            ]

            if filtered_ids:
                print(f"\n符合过滤条件的对象数量: {len(filtered_ids)}")

                # 绘制轨迹图
                plot_trajectories(
                    preprocessed_trajectories,
                    filtered_ids,
                    CONFIG['output_dir'],
                    speeds,
                    qualities
                )

                # 显示过滤后对象的详细信息
                print("\n=== 过滤后目标的详细信息 ===")
                for obj_id in filtered_ids:
                    area = object_areas.get(obj_id, 0)
                    estimated_count = estimate_object_count(area, single_object_area)

                    print(
                        f"ID {obj_id:3d}: "
                        f"速度={speeds[obj_id]:6.2f}, "
                        f"质量={qualities[obj_id]:5.2f}, "
                        f"方向=({average_directions[obj_id][0]:6.2f}, {average_directions[obj_id][1]:6.2f}), "
                        f"面积={area:6.0f}, "
                        f"估计物体数={estimated_count}"
                    )

                # 生成视频文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_name = (
                    f'filtered_video_speed{speed_threshold:.1f}_'
                    f'quality{quality_threshold:.2f}_{timestamp}.mp4'
                )
                video_path = os.path.join(CONFIG['filtered_video_dir'], video_name)

                print("\n=== 开始生成所有版本的视频 ===")

                # 1. 生成原始版本（无标签）
                print("\n1. 正在生成原始版本（无标签）...")
                generate_filtered_video(
                    preprocessed_trajectories,
                    filtered_ids,
                    CONFIG['filtered_video_dir'],
                    video_path,
                    CONFIG['video_path'],
                    speeds=speeds
                )

                # 2. 生成完整轨迹版本（无ID）
                print("\n2. 正在生成完整轨迹版本（无ID）...")
                generate_filtered_video_full_trajectory(
                    preprocessed_trajectories,
                    filtered_ids,
                    CONFIG['filtered_video_dir'],
                    video_path,
                    CONFIG['video_path']
                )

                print("\n=== 视频生成完成 ===")
                print("已生成以下四个版本的视频（无标签）：")
                print(f"1. 原始完整版本（无标签）: filtered_video_{timestamp}.mp4")
                print(f"2. 原始目标版本（无标签）: targets_only_{timestamp}.mp4")
                print(f"3. 完整轨迹完整版本（无ID）: full_trajectory_complete_{timestamp}.mp4")
                print(f"4. 完整轨迹目标版本（无ID）: full_trajectory_targets_only_{timestamp}.mp4")
                print(f"\n所有视频文件已保存到目录: {CONFIG['filtered_video_dir']}")

            else:
                print("\n警告: 没有满足条件的轨迹，请尝试调整阈值。")
                print("建议：")
                print("- 降低速度阈值以包含更多慢速运动的目标")
                print("- 降低质量阈值以包含更多不太理想的轨迹")

        print("\n=== 程序执行完成 ===")
        print("1. 请检查输出目录中的结果")
        print("2. 请关闭可视化窗口以完全退出程序")
        print("感谢使用！")

    except Exception as e:
        print(f"\n错误: 程序执行过程中遇到异常: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # 初始化颜色表
    CONFIG['detection_colors'] = generate_color_table()

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行主函数
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        traceback.print_exc()
    finally:
        # 计算并显示总运行时间
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        print(f"\n总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")

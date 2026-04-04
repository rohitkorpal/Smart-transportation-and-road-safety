import os
import gdown

folder_id = "1ojfHWoPK0U7pbhLZ2CiMx06yloqkZrGL"

# Create models directory
if not os.path.exists("models"):
    os.makedirs("models")

# Download models only if not already present
if len(os.listdir("models")) == 0:
    gdown.download_folder(
        id=folder_id,
        output="models",
        quiet=False
    )
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "opencv-python", "numpy", "ultralytics"])

import logging
import warnings
import os
import sys

# Suppress warnings
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
import cv2
import tempfile
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import json

def model_path(name):
    return os.path.join("models", name)
# Setup relative imports for Accident_AI module
current_dir = os.path.dirname(os.path.abspath(__file__))
accident_ai_path = os.path.join(current_dir, 'Accident_AI')
if accident_ai_path not in sys.path:
    sys.path.append(accident_ai_path)

from main import AccidentDetectionSystem

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart City AI Suite", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- GLOBAL MODEL LOADING FOR TRAFFIC CONTROLLER ---
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8m.pt")

try:
    yolo_model = load_yolo_model()
except Exception as e:
    # st.error(f"Error loading model: {e}")
    yolo_model = None

vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# --- APP 1: ADAPTIVE TRAFFIC WARNING SYSTEM ---
def render_traffic_control():
    # Professional CSS for Traffic Control
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            .main-header-traffic {
                background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
                padding: 2rem;
                border-radius: 12px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .main-header-traffic h1 {
                font-weight: 700;
                margin-bottom: 0.5rem;
                letter-spacing: -1px;
            }

            /* Traffic Light Component Styling */
            .traffic-light-container {
                background-color: #2c2c2c;
                width: 80px;
                height: 200px;
                border-radius: 20px;
                padding: 15px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                align-items: center;
                margin: 0 auto 15px auto;
                border: 4px solid #111;
                box-shadow: 0 10px 20px rgba(0,0,0,0.5);
            }

            .light {
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background-color: #444; /* Off state */
                transition: all 0.3s ease;
                border: 2px solid rgba(0,0,0,0.3);
            }

            .light.red.active {
                background-color: #ff3b30;
                box-shadow: 0 0 20px #ff3b30, inset 0 0 10px rgba(255, 255, 255, 0.5);
            }

            .light.yellow.active {
                background-color: #ffcc00;
                box-shadow: 0 0 20px #ffcc00, inset 0 0 10px rgba(255, 255, 255, 0.5);
            }

            .light.green.active {
                background-color: #34c759;
                box-shadow: 0 0 20px #34c759, inset 0 0 10px rgba(255, 255, 255, 0.5);
            }
            
            /* Lane Card Styling */
            .lane-card {
                background: #1f2937;
                border-radius: 15px;
                padding: 1.5rem;
                border: 1px solid #374151;
                transition: transform 0.2s;
            }
            
            .lane-card:hover {
                transform: translateY(-2px);
                border-color: #4B5563;
            }

            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                text-align: center;
                color: #F3F4F6;
            }
            
            .metric-label {
                text-align: center;
                color: #9CA3AF;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            /* Custom Badges */
            .badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
                text-align: center;
            }
            
            .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; }
            .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #10B981; }

        </style>
    """, unsafe_allow_html=True)

    # Header
    with st.container():
        current_time = datetime.now().strftime("%Y-%m-%d | %H:%M:%S")
        st.markdown(f"""
            <div class='main-header-traffic'>
                <h1>🚦 Adaptive Traffic Warning System</h1>
                <p style='color: #cbd5e1; font-size: 1.1rem;'>Dynamic Signal Optimization Powered by Computer Vision</p>
                <div style='margin-top: 1rem; display: inline-block; background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;'>
                    ⏱️ System Time: <b>{current_time}</b> | 🟢 Active Nodes: <b>4</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Main Interface
    st.markdown("### 📡 Feed Inputs & Live Monitoring")
    st.markdown("---")

    lane_names = ["North Avenue", "East Boulevard", "South Street", "West Way"]
    uploaded_videos = [None] * 4

    # Inputs
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.markdown(f"**{lane_names[i]}**")
            uploaded_videos[i] = st.file_uploader(
                f"Input Feed {i+1}",
                type=["mp4", "avi", "mov"],
                key=f"lane{i+1}",
                label_visibility="collapsed"
            )

    # Logic Function
    def process_lane_density(video_file):
        if not video_file or not yolo_model:
            return 0, None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            path = tmp.name
        
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        count = 0
        processed_frame = None
        
        if ret:
            # Resize for performance
            frame = cv2.resize(frame, (640, 360))
            results = yolo_model(frame, verbose=False)
            
            # Count vehicle classes
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) in vehicle_ids:
                        count += 1
                        # Draw for preview
                        p1, p2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        cap.release()
        return count, processed_frame

    # Processing Trigger
    if st.button("⚡ Analyze Traffic & Optimize Signals", type="primary", use_container_width=True):
        if all(v is None for v in uploaded_videos):
            st.warning("⚠️ Please provide video feeds to analyze.")
        else:
            if not yolo_model:
                st.error("Error loading YOLO model. Please check yolov8m.pt file.")
                return

            with st.spinner("🔄 Analyzing traffic density across all nodes..."):
                lane_data = []
                
                # Process all lanes
                for i, video in enumerate(uploaded_videos):
                    count, frame = process_lane_density(video)
                    lane_data.append({
                        "name": lane_names[i],
                        "count": count,
                        "frame": frame
                    })
                
                # Determine logic
                max_density = max(d["count"] for d in lane_data)
                # Default to North if all 0, otherwise Green to max density
                green_idx = 0 
                if max_density > 0:
                    # Find index of max
                    for idx, d in enumerate(lane_data):
                        if d["count"] == max_density:
                            green_idx = idx
                            break
                
                # Display Results
                st.markdown("### 🚥 Signal Optimization Results")
                st.markdown("---")
                
                res_cols = st.columns(4)
                
                for i in range(4):
                    is_green = (i == green_idx)
                    data = lane_data[i]
                    
                    # HTML for Traffic Light
                    red_class = "active" if not is_green else ""
                    green_class = "active" if is_green else ""
                    # Yellow usually for transition, keeping off for static state
                    
                    traffic_light_html = f"""
                    <div class="traffic-light-container">
                        <div class="light red {red_class}"></div>
                        <div class="light yellow"></div>
                        <div class="light green {green_class}"></div>
                    </div>
                    """
                    
                    with res_cols[i]:
                        with st.container():
                            st.markdown(f"""<div class="lane-card">""", unsafe_allow_html=True)
                            
                            st.markdown(f"<h4 style='text-align:center;margin-bottom:1rem;'>{data['name']}</h4>", unsafe_allow_html=True)
                            
                            # Render Light
                            st.markdown(traffic_light_html, unsafe_allow_html=True)
                            
                            # Metrics
                            density_color = "#10B981" if data['count'] < 5 else ("#F59E0B" if data['count'] < 10 else "#EF4444")
                            
                            st.markdown(f"""
                                <div style='margin-top:1rem;'>
                                    <div class='metric-value' style='color:{density_color}'>{data['count']}</div>
                                    <div class='metric-label'>Vehicles Detected</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Status Badge
                            status_html = f"<span class='badge badge-success'>OPEN</span>" if is_green else f"<span class='badge badge-danger'>STOP</span>"
                            st.markdown(f"<div style='text-align:center;margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)

                            # Preview Image
                            if data['frame'] is not None:
                                st.image(data['frame'], use_container_width=True, channels="RGB")
                            else:
                                st.info("No Feed")
                                
                            st.markdown("</div>", unsafe_allow_html=True)

    # Footer for Traffic Controller
    st.markdown("""
        <div style='margin-top: 50px; text-align: center; color: #666; padding: 20px; border-top: 1px solid #333;'>
            Smart City Traffic OS v2.2 | Integrated with YOLOv8 Neural Network
        </div>
    """, unsafe_allow_html=True)


# --- APP 2: SMART TRAFFIC & ACCIDENT DETECTION AI ---
def render_accident_detection():
    # Custom CSS for Accident Detection
    st.markdown("""
        <style>
            .main-header-accident {
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 20px;
            }
            .alert-box {
                background-color: #ff4444;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #1e3c72;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for AI module
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    # Header
    st.markdown("""
        <div class="main-header-accident">
            <h1>🚨 Smart Traffic & Accident Detection AI</h1>
            <p>Real-time Road Safety Monitoring System</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Options Specific to Detection Module
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Detection Configuration")
    
    # Input source selection
    input_option = st.sidebar.radio(
        "Input Source",
        ["Upload Video", "Upload Image (JPEG)", "RTSP Stream", "File Path"],
        key="accident_input_source"
    )
    
    video_source = None
    image_source = None
    
    if input_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload_acc"
        )
        if uploaded_file:
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_source = tfile.name
    
    elif input_option == "Upload Image (JPEG)":
        uploaded_image = st.sidebar.file_uploader(
            "Upload image file (JPEG/JPG)",
            type=["jpg", "jpeg", "png"],
            key="image_upload_acc"
        )
        if uploaded_image:
            # Save to temporary file
            suffix = ".jpg" if uploaded_image.name.lower().endswith(('.jpg', '.jpeg')) else ".png"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(uploaded_image.read())
            image_source = tfile.name
    
    elif input_option == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("RTSP URL", value="rtsp://example.com/stream", key="rtsp_acc")
        if rtsp_url:
            video_source = rtsp_url
    
    elif input_option == "File Path":
        file_path = st.sidebar.text_input("Video/Image File Path", key="filepath_acc")
        if file_path and os.path.exists(file_path):
            # Check if it's an image or video
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_source = file_path
            else:
                video_source = file_path
    
    # Options
    st.sidebar.subheader("Detection Options")
    enable_pose = st.sidebar.checkbox("Enable Human Detection (Phase-2)", value=False)
    enable_fire = st.sidebar.checkbox("Enable Fire Detection (Phase-3)", value=True)
    enable_road_condition = st.sidebar.checkbox("Enable Road Condition Detection", value=True)
    enable_helmet = st.sidebar.checkbox("Enable Helmet Violation Detection", value=True)
    enable_overload = st.sidebar.checkbox("Enable Overload Vehicle Detection", value=True)
    fps = st.sidebar.slider("FPS", min_value=15, max_value=60, value=30)
    
    # Start button
    if st.sidebar.button("🚀 Start Detection", type="primary", use_container_width=True):
        if video_source or image_source:
            with st.spinner("Initializing system..."):
                try:
                    # Reset state
                    if 'video_cap' in st.session_state:
                        if st.session_state.video_cap is not None:
                            st.session_state.video_cap.release()
                        del st.session_state.video_cap
                    if 'image_processed' in st.session_state:
                        del st.session_state.image_processed
                    
                    # Store source type
                    st.session_state.is_image = image_source is not None
                    st.session_state.source_path = image_source if image_source else video_source
                    
                    if image_source:
                        # For images, create system with dummy video source
                        st.session_state.system = AccidentDetectionSystem(
                            video_source="dummy",
                            fps=fps,
                            enable_pose=enable_pose,
                            enable_fire=enable_fire,
                            enable_road_condition=enable_road_condition,
                            enable_helmet=enable_helmet,
                            enable_overload=enable_overload
                        )
                        # Mark as static image for collision detection
                        st.session_state.system.is_static_image = True
                    else:
                        st.session_state.system = AccidentDetectionSystem(
                            video_source=video_source,
                            fps=fps,
                            enable_pose=enable_pose,
                            enable_fire=enable_fire,
                            enable_road_condition=enable_road_condition,
                            enable_helmet=enable_helmet,
                            enable_overload=enable_overload
                        )
                    st.session_state.video_processed = True
                    st.sidebar.success("System initialized!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                    import traceback
                    st.sidebar.code(traceback.format_exc())
        else:
            st.sidebar.warning("Please provide a video or image source")

    # Main content
    if st.session_state.system and st.session_state.video_processed:
        # Create columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Check if processing image or video
            if st.session_state.get('is_image', False):
                st.subheader("🖼️ Image Detection")
                image_placeholder = st.empty()
                
                # Process image
                if 'image_processed' not in st.session_state:
                    try:
                        # Read image
                        frame = cv2.imread(st.session_state.source_path)
                        if frame is not None:
                            # Process frame
                            tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                            
                            # Draw detections
                            frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                            
                            # Update alerts
                            if alerts:
                                st.session_state.alerts.extend(alerts)
                            
                            # Convert to RGB for Streamlit
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            
                            st.session_state.image_processed = True
                            st.success("✅ Image processed successfully!")
                        else:
                            st.error("Could not read image file")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    # Re-display processed image
                    frame = cv2.imread(st.session_state.source_path)
                    if frame is not None:
                        tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                        frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            else:
                # Video processing
                st.subheader("📹 Live Video Feed")
                video_placeholder = st.empty()
                
                # Process video frame by frame
                if 'video_cap' not in st.session_state:
                    st.session_state.video_cap = cv2.VideoCapture(st.session_state.source_path)
                    st.session_state.frame_count = 0
                    st.session_state.video_finished = False
                
                cap = st.session_state.video_cap
                
                if cap is not None and cap.isOpened() and not st.session_state.video_finished:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Process frame
                        tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                        
                        # Draw detections
                        frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                        
                        # Update alerts
                        if alerts:
                            st.session_state.alerts.extend(alerts)
                        
                        # Convert to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        st.session_state.frame_count += 1
                        
                        # Limit processing for demo or check if video ended
                        if st.session_state.frame_count >= 100:  # Process first 100 frames
                            st.session_state.video_finished = True
                            cap.release()
                            st.info("✅ Video processing complete! Processed 100 frames.")
                    else:
                        st.session_state.video_finished = True
                        cap.release()
                        st.info("✅ Video processing complete!")
                elif st.session_state.video_finished:
                    st.info("✅ Video processing complete!")
                else:
                    st.error("Could not open video source")
        
        with col2:
            st.subheader("📊 Statistics")
            
            # Get alert summary
            if st.session_state.system:
                summary = st.session_state.system.alert_manager.get_alert_summary()
                
                st.metric("Total Alerts", summary['total_alerts'])
                st.metric("Vehicles Tracked", len(st.session_state.system.track_history))
                
                # Alert counts
                st.subheader("Alert Breakdown")
                for alert_type, count in summary['alert_counts'].items():
                    st.metric(alert_type.replace("_", " ").title(), count)
            
            # Recent alerts
            st.subheader("🚨 Recent Alerts")
            if st.session_state.alerts:
                for alert in st.session_state.alerts[-10:]:  # Last 10 alerts
                    alert_type = alert[0]
                    st.markdown(f"""
                        <div class="alert-box">
                            <strong>{alert_type.replace('_', ' ').title()}</strong>
                            <br>{datetime.now().strftime('%H:%M:%S')}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts yet")
            
            # Download logs
            if st.button("📥 Download Alert Logs", use_container_width=True):
                log_dir = st.session_state.system.alert_manager.log_dir
                if os.path.exists(log_dir):
                    # Create zip of logs
                    import zipfile
                    zip_path = "alerts_logs.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file in os.listdir(log_dir):
                            zipf.write(os.path.join(log_dir, file), file)
                    
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            "Download",
                            f.read(),
                            file_name="alerts_logs.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
    else:
        st.info("👈 Please configure and start the system from the sidebar")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>Smart Traffic & Accident Detection AI System | Microsoft Hackathon 2025</p>
            <p>Built with YOLOv8, DeepSORT, and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


# --- MAIN NAVIGATION ---
def main():
    st.sidebar.title("Smart City Suite")
    st.sidebar.markdown("Choose a module to monitor:")
    app_mode = st.sidebar.radio(
        "Navigation",
        ["🚦 Traffic Controller", "🚨 Accident Detection AI"],
        label_visibility="collapsed"
    )

    if app_mode == "🚦 Traffic Controller":
        render_traffic_control()
    elif app_mode == "🚨 Accident Detection AI":
        render_accident_detection()

if __name__ == "__main__":
    main()

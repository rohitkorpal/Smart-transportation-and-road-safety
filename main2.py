import logging
# silence the Streamlit ScriptRunContext warnings
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

try:
    yolo_model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    yolo_model = None

vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Streamlit Page Config
st.set_page_config(
    page_title="Smart City Traffic Control", 
    layout="wide", 
    page_icon="🚦",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .stApp {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-header h1 {
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
        
        /* Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3rem;
            font-weight: 600;
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
        <div class='main-header'>
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
if st.button("⚡ Analyze Traffic & Optimize Signals", type="primary"):
    if all(v is None for v in uploaded_videos):
        st.warning("⚠️ Please provide video feeds to analyze.")
    else:
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

# Footer
st.markdown("""
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; color: #666; padding: 20px; background: #0e1117; border-top: 1px solid #333;'>
        Smart City Traffic OS v2.2 | Integrated with YOLOv8 Neural Network
    </div>
""", unsafe_allow_html=True)

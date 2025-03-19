import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import os

def get_streamlit_url():
    """Get the URL for the Streamlit dashboard."""
    return "http://localhost:8501"  # Default Streamlit port

def load_latest_results():
    """Load the latest comparison results from the results directory."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None
    
    # Get the latest results file
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        return None
    
    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file) as f:
        return json.load(f)

def main():
    st.set_page_config(
        page_title="Cloud Vision Services Comparison",
        page_icon="📊",
        layout="wide"
    )

    st.title("Cloud Vision Services Comparison Dashboard")

    results = load_latest_results()
    if not results:
        st.warning("No comparison results available yet. Please run a comparison first.")
        return

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Processing Time", "Detections", "Cost Analysis"])

    with tab1:
        st.header("Processing Time Comparison")
        time_df = pd.DataFrame({
            'Service': ['YOLO', 'AWS Rekognition', 'Azure Computer Vision'],
            'Time (seconds)': [
                results['processing_time']['yolo'],
                results['processing_time']['aws'],
                results['processing_time']['azure']
            ]
        })
        fig = px.bar(time_df, x='Service', y='Time (seconds)',
                    title='Processing Time by Service',
                    color='Service')
        st.plotly_chart(fig)

    with tab2:
        st.header("Object Detections")
        detections_df = pd.DataFrame({
            'Service': ['YOLO', 'AWS Rekognition', 'Azure Computer Vision'],
            'Objects Detected': [
                results['detections']['yolo'],
                results['detections']['aws'],
                results['detections']['azure']
            ]
        })
        fig = px.bar(detections_df, x='Service', y='Objects Detected',
                    title='Number of Objects Detected by Service',
                    color='Service')
        st.plotly_chart(fig)

    with tab3:
        st.header("Cost Analysis")
        cost_df = pd.DataFrame({
            'Service': ['YOLO', 'AWS Rekognition', 'Azure Computer Vision'],
            'Cost (USD)': [
                results['costs']['yolo'],
                results['costs']['aws'],
                results['costs']['azure']
            ]
        })
        fig = px.bar(cost_df, x='Service', y='Cost (USD)',
                    title='Cost Comparison by Service',
                    color='Service')
        st.plotly_chart(fig)

        # Add cost efficiency metrics
        st.subheader("Cost Efficiency")
        cost_efficiency = pd.DataFrame({
            'Service': ['YOLO', 'AWS Rekognition', 'Azure Computer Vision'],
            'Cost per Detection (USD)': [
                results['costs']['yolo'] / results['detections']['yolo'] if results['detections']['yolo'] > 0 else 0,
                results['costs']['aws'] / results['detections']['aws'] if results['detections']['aws'] > 0 else 0,
                results['costs']['azure'] / results['detections']['azure'] if results['detections']['azure'] > 0 else 0
            ]
        })
        fig = px.bar(cost_efficiency, x='Service', y='Cost per Detection (USD)',
                    title='Cost per Detection by Service',
                    color='Service')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main() 
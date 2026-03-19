import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

st.set_page_config(page_title="Music Genre AI", layout="wide")

# Initialize Spark and Model
@st.cache_resource
def load_resources():
    spark = SparkSession.builder.appName("MusicAppInference").getOrCreate()
    model = PipelineModel.load("models/genre_xgb_model")
    labels = model.stages[0].labels
    return spark, model, labels

spark, model, labels = load_resources()

# Sidebar for visualizations
st.sidebar.title("📊 Analytics Dashboard")
st.sidebar.info("The charts below show the model's confidence for each music genre.")

# Main UI
st.title("🎵 Lyric Genre Classifier")
st.write("Enter song lyrics to identify the genre.")

lyrics_input = st.text_area("Paste Lyrics Here:", height=300, placeholder="Type or paste lyrics...")

if st.button("Classify Genre") and lyrics_input:
    # 1. Create Spark DataFrame
    input_df = spark.createDataFrame([(lyrics_input,)], ["lyrics"])
    
    # 2. Run Inference
    prediction = model.transform(input_df)
    
    # 3. Extract Probabilities and Map to Labels
    prob_vector = prediction.select("probability").collect()[0][0].toArray()
    results_df = pd.DataFrame({
        "Genre": labels,
        "Confidence": prob_vector
    }).sort_values(by="Confidence", ascending=False)

    # 4. Define Threshold (e.g., 0.45 means 45% confidence)
    THRESHOLD = 0.25 
    top_confidence = results_df.iloc[0]['Confidence']
    top_genre = results_df.iloc[0]['Genre'].upper()
    
    # 5. Requirement Logic: Check if it belongs to trained classes
    if top_confidence < THRESHOLD:
        st.error("### Result: Unknown Genre")
        st.warning(f"The system is only {top_confidence*100:.1f}% confident. This song likely belongs to a genre outside of the 8 trained classes.")
    else:
        st.success(f"### Result: This sounds like **{top_genre}**")
        st.balloons()
    
    # Update Sidebar Charts (Visualizing the low confidence helps prove your point)
    with st.sidebar:
        st.write(f"**Max Confidence Score:** {top_confidence:.4f}")
        
        # Pie Chart
        fig_pie = px.pie(results_df, values='Confidence', names='Genre', 
                         title="Confidence Distribution", hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar Chart
        fig_bar = px.bar(results_df, x='Confidence', y='Genre', orientation='h',
                         title="Probability Score by Genre", color='Confidence',
                         color_continuous_scale='Magma')
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.sidebar.warning("Awaiting input lyrics...")
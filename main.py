import streamlit as st
from arcgis.features import GeoAccessor
from arcgis.gis import GIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

item_id_dict = {
        "psu": "e185caf63fbd452aa7b3d1e6396404a9",
        "ssu": "543d31deb07c4a4ab4ae9d59b429508d",
}
ceo_legend = {
    "Trees_CanopyCover": "#007500",
    "bush/scrub": "#DBED00",
    "grass": "#00F100",
    "cultivated vegetation": "#FF00D6",
    "Water>treated pool": "#00F1DE",
    "Water>lake/ponded/container": "#00B7F2",
    "Water>rivers/stream": "#1527F6",
    "Water>irrigation ditch": "#007570",
    "shadow": "#000000",
    "unknown": "#C8D2D3",
    "Bare Ground": "#AA7941",
    "Building": "#FF8080",
    "Impervious Surface (no building)": "#FF0000",
}
formatted_legend = {
    key[:11].replace(">", " ").replace("_", " ").replace("/", " ") : key for key in ceo_legend.keys()
}

@st.cache
def get_data(data_type):
    gis = GIS()
    item = gis.content.get(itemid=item_id_dict[data_type])
    df = GeoAccessor.from_layer(item.layers[0])

    if "SHAPE" in df:
        df.drop(["SHAPE"], axis=1, inplace=True)
    
    return df

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if 'psu_data' not in st.session_state:
    st.session_state['psu_data'] = get_data("psu")
if 'ssu_data' not in st.session_state:
    st.session_state['ssu_data'] = get_data("ssu")
if "selected_psu" not in st.session_state:
    st.session_state["selected_psu"] = pd.DataFrame()
if "selected_ssu" not in st.session_state:
    st.session_state["selected_ssu"] = pd.DataFrame()

st.title("Collect Earth AOI Preview and Download")
aoi = st.selectbox("Choose you AOI Number", pd.unique(st.session_state['psu_data']["AOI_Number"]))

st.session_state["selected_psu"] = st.session_state["psu_data"][st.session_state["psu_data"]["AOI_Number"] == aoi]
st.session_state["selected_ssu"] = st.session_state["ssu_data"][st.session_state["ssu_data"]["AOI_Number"] == aoi]

st.header("Data Summary")
col1, col2, col3=  st.columns(3)
col1.metric("Plots completed", len(st.session_state["selected_psu"]))
col2.metric("Plots left", 37 - len(st.session_state["selected_psu"]))
#st.dataframe(st.session_state["selected_psu"])
lc_prefix = "Land_Cover_Elements_"
lc_classifications = [col for col in st.session_state["selected_psu"] if col.startswith(lc_prefix)]
classification_names = []
average_percent_cover = []
colors = []
for classification in lc_classifications:
    mean = np.mean(st.session_state["selected_psu"][classification])
    if mean > 0:
        formatted_name = classification.replace(lc_prefix, "").replace("_", " ")
        ceo_name = formatted_legend[formatted_name]
        classification_names.append(ceo_name)
        average_percent_cover.append(mean)
        colors.append(ceo_legend[ceo_name])
largest_percent = max(average_percent_cover)
primary_classification = classification_names[average_percent_cover.index(largest_percent)]
col3.metric("Primary Classification", primary_classification)

fig, ax = plt.subplots()
ax.pie(average_percent_cover, labels=classification_names, colors=colors)
ax.axis("equal")

st.subheader("Land Cover Percentage Pie Chart")
st.pyplot(fig)

st.subheader("Analyzed AOI Map")
st.map(st.session_state["selected_psu"].rename({"center_lat" : "lat", "center_lon" : "lon"}, axis=1))

st.header("Data Download")
psu_download, ssu_download =  st.columns(2)

selected_psu_data = convert_df(st.session_state["selected_psu"])
selected_ssu_data = convert_df(st.session_state["selected_ssu"])
psu_download.download_button(label="Download Primary Sampling Unit CSV", data = selected_psu_data, file_name=f"CEO PSU-{aoi}.csv", mime='text/csv')
ssu_download.download_button(label="Download Secondary Sampling Unit CSV", data = selected_ssu_data, file_name=f"CEO SSU-{aoi}.csv", mime='text/csv')
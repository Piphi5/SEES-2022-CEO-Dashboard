import streamlit as st
from arcgis.features import GeoAccessor
from arcgis.gis import GIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ee
from functools import partial
from matplotlib.patches import Rectangle as MPLRect
import math
import geemap.foliumap as geemap
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ipyleaflet import Rectangle, LayerGroup
import folium

import go_utils
from go_utils.constants import landcover_protocol

st.set_page_config(layout="wide")
Map = geemap.Map()
ee.Initialize()
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
    key[:11].replace(">", " ").replace("_", " ").replace("/", " "): key
    for key in ceo_legend.keys()
}

harmonized_classes = [
    "Grassland",
    "Shrubland",
    "Built Up",
    "Barren",
    "Trees",
    "Cropland",
    "Water Bodies",
    "Wetland",
    "Snow",
]

harmonized_classes_vis = {
    "Grassland": "#00F100",
    "Shrubland": "#DBED00",
    "Built Up": "#FF0000",
    "Barren": "#AA7941",
    "Trees": "#007500",
    "Cropland": "#FF00D6",
    "Water Bodies": "#00B7F2",
    "Wetland": "#0096A0",
    "Snow": "#F0F0F0",
}

ceo_to_harmonized_lookup = {
    "Trees_CanopyCover": "Trees",
    "bush/scrub": "Shrubland",
    "grass": "Grassland",
    "cultivated vegetation": "Cropland",
    "Water>lake/ponded/container": "Water Bodies",
    "Water>rivers/stream": "Water Bodies",
    "Water>irrigation ditch": "Wetland",
    "Water>treated pool": "Water Bodies",
    "Bare Ground": "Barren",
    "Building": "Built Up",
    "Impervious Surface (no building)": "Built Up",
}

wc_to_harmonized_lookup = {
    "Trees": "Trees",
    "Shrubland": "Shrubland",
    "Grassland": "Grassland",
    "Cropland": "Cropland",
    "Built-up": "Built Up",
    "Barren / Sparse Vegetation": "Barren",
    "Snow and Ice": "Snow",
    "Open Water": "Water Bodies",
    "Herbaceous Wetland": "Wetland",
    "Mangroves": "Wetland",
    "Moss and Lichen": "Grassland",  # don't know if this is a good class to put this in
}


wc_id_classification_map = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Barren / Sparse Vegetation",
    70: "Snow and Ice",
    80: "Open Water",
    90: "Herbaceous Wetland",
    95: "Mangroves",
    100: "Moss and Lichen",
}


@st.cache(ttl=3600)
def get_data(data_type):
    gis = GIS()
    item = gis.content.get(itemid=item_id_dict[data_type])
    df = GeoAccessor.from_layer(item.layers[0])

    if "SHAPE" in df:
        df.drop(["SHAPE"], axis=1, inplace=True)

    return df


def to_gee(lat, lon, classification, objectid):
    return ee.Feature(
        ee.Geometry.Point([lon, lat]),
        {"ObjectId": objectid},
    )


gee_converter = np.vectorize(to_gee)


def wc_to_str(value):
    return wc_id_classification_map[value]


worldcover_converter = np.vectorize(wc_to_str)


def convert_to_harmonized(lookup, value):
    if value in lookup:
        return lookup[value]
    else:
        return ""


ceo_converter = np.vectorize(partial(convert_to_harmonized, ceo_to_harmonized_lookup))
wc_converter = np.vectorize(partial(convert_to_harmonized, wc_to_harmonized_lookup))


def get_latlon_spacing_constants(grid_distance, latitude):
    # Calculate grid constants
    r_earth = 6.371 * 10**6

    # See Theoretical Basis for the derivations
    latitude_const = (
        360 / math.pi * math.asin((math.sin(grid_distance / (2 * r_earth))))
    )
    longitude_const = (
        360
        / math.pi
        * math.asin(
            math.sin(grid_distance / (2 * r_earth)) / math.cos(latitude * math.pi / 180)
        )
    )

    return latitude_const, longitude_const


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)

    From: https://stackoverflow.com/a/4913653
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000  # put into meters


@st.cache
def enrich_ceo_data(df, image):
    ceo_payload = gee_converter(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        df["Land_Cover_Elements"].to_numpy(),
        df["ObjectId"].to_numpy(),
    ).tolist()
    ceo_fc = ee.FeatureCollection(ceo_payload)
    enriched = image.reduceRegions(
        **{
            "collection": ceo_fc,
            "reducer": ee.Reducer.median(),
            "scale": 10,
        }
    )
    wc_df = geemap.ee_to_df(enriched)
    wc_df["wc_classification"] = worldcover_converter(wc_df["median"].to_numpy())
    combined = pd.merge(df, wc_df, on="ObjectId")
    combined["harmonized_ceo"] = ceo_converter(
        combined["Land_Cover_Elements"].to_numpy()
    )
    combined["harmonized_wc"] = wc_converter(combined["wc_classification"].to_numpy())
    return combined[
        (combined["harmonized_ceo"] != "") & (combined["harmonized_wc"] != "")
    ]

def generate_confusion_matrix(df):
    cf_matrix = confusion_matrix(
        df["harmonized_ceo"],
        df["harmonized_wc"],
        labels=harmonized_classes,
    )

    user_total = []

    for row in cf_matrix:
        user_total.append(np.sum(row))

    producer_total = []

    for col in cf_matrix.T:
        producer_total.append(np.sum(col))

    agreement = []
    for i, row in enumerate(cf_matrix):
        agreement.append((row[i] / producer_total[i], harmonized_classes[i]))

    confusion = []
    for i, row in enumerate(cf_matrix):
        for j, col in enumerate(row):
            if j == i:
                continue
            else:
                percent_error = (
                    col / producer_total[j] if producer_total[j] > 0.0 else 0.0
                )
                confusion.append((percent_error, col, (i, j)))

    producer_total = np.array(producer_total, dtype="float64")
    # (np.sum(producer_total))
    producer_total /= np.sum(producer_total)
    user_total = np.array(user_total, dtype="float64")
    user_total /= np.sum(user_total)

    agreed_tuple = max(agreement)  # Sorted by percent error and then by raw error
    number_confused, _, confused_values = max((confusion))
    ceo, wc = confused_values
    confused_str = f"{harmonized_classes[ceo]} for {harmonized_classes[wc]}"
    confused_tuple = (number_confused, confused_str)
    cf_matrix = confusion_matrix(
        df["harmonized_ceo"],
        df["harmonized_wc"],
        labels=harmonized_classes,
        normalize="pred",
    )
    cf_matrix = np.append(cf_matrix, np.array([user_total]).T, axis=1)
    producer_total = np.append(producer_total, np.sum(producer_total))
    cf_matrix = np.append(cf_matrix, [producer_total], axis=0)

    fig, ax = plt.subplots(figsize=(20, 10))
    mask = np.zeros(cf_matrix.shape)
    mask[-1:] = True
    mask[:, -1:] = True

    sns.heatmap(cf_matrix, mask=mask, cmap="Blues", linewidths=0.25, linecolor="black")
    sns.heatmap(cf_matrix, alpha=0, cbar=False, annot=True, annot_kws={"color": "k"})
    for i in range(len(harmonized_classes)):
        ax.add_patch(MPLRect((i, i), 1, 1, fill=False, edgecolor="black", lw=3))
    ax.add_patch(MPLRect((wc, ceo), 1, 1, fill=False, edgecolor="red", lw=3))
    ax.yaxis.set_ticklabels(harmonized_classes + ["WC Percentage"])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_ticklabels(harmonized_classes + ["User Percentage"])
    ax.set_xlabel("\nWorld Cover")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Collect Earth Online ")

    return fig, agreed_tuple, confused_tuple


def get_accuracy(df):
    agreed = df["harmonized_ceo"] == df["harmonized_wc"]
    return len(df[agreed]) / len(df)


def sentinel_cloud_mask(image):
    qa = image.select("QA60")
    cloudBit = 1 << 10
    cirrusBit = 1 << 1

    mask = qa.bitwiseAnd(cloudBit) and (qa.bitwiseAnd(cirrusBit).eq(0))

    return image.updateMask(mask).divide(10000)


sentinel_image = (
    ee.ImageCollection("COPERNICUS/S2_SR")
    .filterDate("2020-01-01", "2020-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(sentinel_cloud_mask)
    .reduce(ee.Reducer.median())
)
sentinel_vis = {
    "min": 0.0,
    "max": 0.3,
    "bands": ["B4_median", "B3_median", "B2_median"],
}
world_cover = ee.ImageCollection("ESA/WorldCover/v100").first()
harmonized_wc_vis = [
    harmonized_classes_vis[wc_to_harmonized_lookup[classification]]
    for classification in wc_to_harmonized_lookup.keys()
]


def in_bounding_box(ne, sw, lat, lon):
    max_lat, max_lon = ne
    min_lat, min_lon = sw
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def get_globe_photos(lc_df, lat, lon):

    lat_const, lon_const = get_latlon_spacing_constants(50, lat)
    ne = (lat + lat_const, lon + lon_const)
    sw = (lat - lat_const, lon - lon_const)
    bounding_filter = np.vectorize(partial(in_bounding_box, ne, sw))
    mask = bounding_filter(
        lc_df["lc_Latitude"].to_numpy(), lc_df["lc_Longitude"].to_numpy()
    )
    if not np.any(mask):
        return [], ()
    intersected = lc_df[mask]
    vec_dist = np.vectorize(partial(haversine, lat, lon))
    intersected["Dist"] = vec_dist(
        intersected["lc_Latitude"].to_numpy(), intersected["lc_Longitude"].to_numpy()
    )
    photo_cols = [col for col in intersected.columns if "Url" in col]
    intersected_entry = intersected.sort_values(by="Dist").iloc[0]
    urls = intersected_entry[photo_cols]
    cleaned_urls = [
        (url, direction)
        for url, direction in zip(urls, photo_cols)
        if not pd.isna(url) and "https" in url
    ]
    return cleaned_urls, intersected_entry[["lc_Latitude", "lc_Longitude"]]


def display_latlon_coords(center_lat, center_lon, chip_size):
    with st.expander("View Coordinates for External Enrichment"):
        st.subheader("Link to Time Series explorer")
        link = f"https://jstnbraaten.users.earthengine.app/view/landsat-timeseries-explorer#run=true;lon={center_lon};lat={center_lat};from=06-10;to=09-20;index=NBR;rgb=SWIR1%2FNIR%2FGREEN;chipwidth={chip_size};"
        st.markdown(f"[Link to Time Series Explorer]({link})")
        st.subheader("Plot Center Coords (Latitude, Longitude):")
        st.code(f"{center_lat}, {center_lon}")
        st.subheader("Plot Center Coords (Latitude):")
        st.code(f"{center_lat}")
        st.subheader("Plot Center Coords (Longitude):")
        st.code(f"{center_lon}")


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache
def get_lc_data(ttl=3600):
    return go_utils.get_api_data(landcover_protocol)


if "psu_data" not in st.session_state:
    st.session_state["psu_data"] = get_data("psu")
if "ssu_data" not in st.session_state:
    st.session_state["ssu_data"] = get_data("ssu")
if "selected_psu" not in st.session_state:
    st.session_state["selected_psu"] = pd.DataFrame()
if "selected_ssu" not in st.session_state:
    st.session_state["selected_ssu"] = pd.DataFrame()
if "analysis_ssu" not in st.session_state:
    st.session_state["analysis_ssu"] = pd.DataFrame()
if "lc_data" not in st.session_state:
    st.session_state["lc_data"] = get_lc_data()


entire_aoi_option = "Analyze entire AOI"

st.title("Collect Earth AOI Preview and Download")
st.header("AOI Selection")
aoi_list = pd.unique(st.session_state["psu_data"]["AOI_Number"]).tolist()
aoi_list.sort()
aoi = st.selectbox("Choose your AOI Number", aoi_list)
st.session_state["selected_psu"] = st.session_state["psu_data"][
    st.session_state["psu_data"]["AOI_Number"] == aoi
]
st.session_state["selected_ssu"] = st.session_state["ssu_data"][
    st.session_state["ssu_data"]["AOI_Number"] == aoi
]
plotid_list = pd.unique(st.session_state["selected_ssu"]["plotid"]).tolist()
plotid_list.sort()
plotid_list.insert(0, entire_aoi_option)

plotid = st.selectbox("Choose your ploid", plotid_list)

if plotid == entire_aoi_option:
    st.session_state["analysis_ssu"] = st.session_state["selected_ssu"]
else:
    st.session_state["analysis_ssu"] = st.session_state["selected_ssu"][
        st.session_state["selected_ssu"]["plotid"] == plotid
    ]
psu_col, ssu_col = st.columns(2)
with psu_col:
    st.header("PSU Summary")

    st.metric("Plots completed", len(st.session_state["selected_psu"]))
    st.metric("Plots left", 37 - len(st.session_state["selected_psu"]))
    # st.dataframe(st.session_state["selected_psu"])
    lc_prefix = "Land_Cover_Elements_"
    lc_classifications = [
        col for col in st.session_state["selected_psu"] if col.startswith(lc_prefix)
    ]
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
    primary_classification = classification_names[
        average_percent_cover.index(largest_percent)
    ]
    st.metric("Primary Classification", primary_classification)

    fig, ax = plt.subplots()
    ax.pie(average_percent_cover, labels=classification_names, colors=colors)
    ax.axis("equal")

    st.subheader("Land Cover Percentage Pie Chart")
    st.pyplot(fig)

with ssu_col:

    st.session_state["analysis_ssu"] = enrich_ceo_data(
        st.session_state["analysis_ssu"], world_cover
    )
    cf_fig, agreed, confused = generate_confusion_matrix(
        st.session_state["analysis_ssu"]
    )

    agreement = get_accuracy(st.session_state["analysis_ssu"])

    agreed_percent, agreed_class = agreed
    ssu_amount = 3600 if plotid == entire_aoi_option else 100

    agreed_percent = f"{(agreed_percent * 100):.02f}%"
    confused_percent, confused_str = confused
    confused_percent = f"-{(confused_percent*100):.02f}%"
    st.header("SSU Summary")
    st.metric("Agreement", f"{(agreement*100):.02f}%")
    st.metric(
        "Most agreed landcover class",
        agreed_class,
        delta=agreed_percent,
        delta_color="off",
    )
    st.metric(
        "Most confused landcover class (Collect Earth Classification for Worldcover Prediction)",
        confused_str,
        delta=confused_percent,
        delta_color="off",
    )

    st.pyplot(cf_fig)

    # Map.add_legend(title="ESA Land Cover", builtin_legend="ESA_WorldCover")
    Map.add_legend(legend_dict=harmonized_classes_vis)
    Map.add_basemap("SATELLITE")
    Map.addLayer(sentinel_image, sentinel_vis, "Sentinel View")
    Map.addLayer(
        world_cover,
        vis_params={"min": 10, "max": 100, "palette": harmonized_wc_vis},
        name="World Cover",
    )
    if plotid != entire_aoi_option:
        center_lat, center_lon = st.session_state["selected_psu"][
            st.session_state["selected_psu"]["plotid"] == plotid
        ].iloc[0][["center_lat", "center_lon"]]
        display_latlon_coords(center_lat, center_lon, 0.1)

        if st.button("Find nearby GLOBE Pictures"):
            with st.expander("View pictures"):
                photo_list, coords = get_globe_photos(
                    st.session_state["lc_data"],
                    center_lat,
                    center_lon,
                )
                if len(photo_list) != 0:
                    lc_lat, lc_lon = coords
                    point = folium.FeatureGroup(name="GLOBE Observation")
                    folium.CircleMarker(
                        location=(lc_lat, lc_lon),
                        name="GLOBE Observation",
                        radius=10,
                        color="#000000",
                        fill_color="white",
                    ).add_to(point)
                    point.add_to(Map)
                    for photo in photo_list:
                        url, direction = photo
                        st.text(direction.replace("lc_", "").replace("PhotoUrl", ""))
                        st.image(url)
                else:
                    st.write("No images found.")

        x = []
        y = []
        grid = folium.FeatureGroup(name="100m Grid")
        for _, data in st.session_state["analysis_ssu"][
            st.session_state["analysis_ssu"]["plotid"] == plotid
        ].iterrows():
            lat_const, lon_const = get_latlon_spacing_constants(4.0, data["lat"])
            sw = (data["lat"] - lat_const, data["lon"] - lon_const)
            ne = (data["lat"] + lat_const, data["lon"] + lon_const)
            bounds = [sw, ne]
            y.extend([sw[0], ne[0]])
            x.extend([sw[1], ne[1]])
            folium.Rectangle(
                bounds=bounds,
                fill=True,
                color="#000000",
                fill_color=harmonized_classes_vis[data["harmonized_ceo"]],
                fill_opacity=0.75,
            ).add_to(grid)

        centerlat = (min(y) + max(y)) / 2
        centerlon = (min(x) + max(x)) / 2

        Map.setCenter(centerlon, centerlat, zoom=19)
        grid.add_to(Map)

        Map.addLayerControl()
        Map.to_streamlit(height=750)
    else:
        center_lat, center_lon = st.session_state["selected_psu"][
            st.session_state["selected_psu"]["plotid"] == aoi * 100
        ].iloc[0][["center_lat", "center_lon"]]
        display_latlon_coords(center_lat, center_lon, 3.5)


with st.sidebar:
    st.header("Data Download")

    selected_psu_data = convert_df(st.session_state["selected_psu"])
    selected_ssu_data = convert_df(st.session_state["selected_ssu"])
    analysis_ssu_data = convert_df(st.session_state["analysis_ssu"])
    st.download_button(
        label="Download Primary Sampling Unit CSV",
        data=selected_psu_data,
        file_name=f"CEO PSU-{aoi}.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Secondary Sampling Unit CSV",
        data=selected_ssu_data,
        file_name=f"CEO SSU-{aoi}.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Enriched Secondary Sampling Unit Data",
        data=selected_ssu_data,
        file_name=f"CEO Enriched SSU-{aoi}.csv",
        mime="text/csv",
    )

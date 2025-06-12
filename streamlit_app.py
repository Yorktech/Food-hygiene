import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="FHRS Hygiene Map", layout="wide")
st.title("\U0001F374 Food Hygiene Ratings by Postcode (UK)")

# Input
postcode_input = st.text_input("Enter a UK postcode or postcode area:", "SW1A 1AA")
st.caption("💡 Enter a full postcode (e.g., SW1A 1AA) or postcode area (e.g., YO1, M1, SW1) to see all establishments in that area")
min_rating = st.slider("Minimum Rating", 0, 5, 0)

# Slider to control zoom threshold for clustering
cluster_zoom_threshold = st.slider("Cluster Zoom Threshold", 10, 20, 15)

# Fetch + Parse XML on button press
if st.button("Get Ratings and Map"):
    url = "https://ratings.food.gov.uk/api/open-data-files/FHRS868en-GB.xml"
    st.write("\U0001F4E6 Downloading and parsing data...")

    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to fetch data. Check the URL or try again later.")
    else:
        root = ET.fromstring(response.content)
        establishments = []

        for est in root.findall(".//EstablishmentDetail"):
            name = est.findtext("BusinessName")
            postcode = est.findtext("PostCode")
            rating = est.findtext("RatingValue")
            address = est.findtext("AddressLine1")
            authority = est.findtext("LocalAuthorityName")

            lat_elem = est.find(".//Geocode/Latitude")
            lon_elem = est.find(".//Geocode/Longitude")
            lat = lat_elem.text.strip() if lat_elem is not None else None
            lon = lon_elem.text.strip() if lon_elem is not None else None

            try:
                lat_float = float(lat)
                lon_float = float(lon)
            except (TypeError, ValueError):
                lat_float, lon_float = None, None

            if name and postcode:
                establishments.append({
                    "BusinessName": name,
                    "Postcode": postcode,
                    "Rating": rating,
                    "Address": address,
                    "Authority": authority,
                    "Latitude": lat_float,
                    "Longitude": lon_float
                })

        # Build DataFrame
        df = pd.DataFrame(establishments)
        df.dropna(subset=["Latitude", "Longitude", "Rating"], inplace=True)
        df["PostcodeClean"] = df["Postcode"].str.replace(" ", "").str.upper()
        df = df[df["Rating"].apply(lambda x: x.isdigit() and int(x) >= min_rating)]

        postcode_cleaned = postcode_input.replace(" ", "").upper()
        
        # Check if it's a postcode area search (e.g., "YO1") or full postcode
        if len(postcode_cleaned) <= 4:  # Postcode area search
            # For area search, match against original postcodes with spaces
            # This way "YO1" matches "YO1 9NA" but not "YO19 4TA"
            area_pattern = postcode_cleaned + " "  # Add space after area
            filtered_df = df[df["Postcode"].str.upper().str.startswith(area_pattern)].copy()
            search_type = "postcode area"
        else:  # Full postcode search
            filtered_df = df[df["PostcodeClean"] == postcode_cleaned].copy()
            search_type = "postcode"

        def rating_to_color(rating):
            try:
                val = int(rating)
                if val == 5: return [0, 200, 0, 180]
                elif val == 4: return [100, 200, 0, 180]
                elif val == 3: return [255, 200, 0, 180]
                elif val == 2: return [255, 100, 0, 180]
                elif val == 1: return [200, 0, 0, 180]
                elif val == 0: return [120, 0, 0, 180]
            except:
                return [128, 128, 128, 160]

        filtered_df["Color"] = filtered_df["Rating"].apply(rating_to_color)

        if not filtered_df.empty:
            st.success(f"Found {len(filtered_df)} establishments in {search_type} {postcode_input.upper()}")
            st.dataframe(filtered_df[["BusinessName", "Rating", "Address", "Postcode", "Latitude", "Longitude"]])

            # Round coordinates to 3 decimal places for clustering
            filtered_df["Latitude_Rounded"] = filtered_df["Latitude"].round(3)
            filtered_df["Longitude_Rounded"] = filtered_df["Longitude"].round(3)

            # Create clustered data for points with same rounded coordinates
            clustered_data = filtered_df.groupby(["Latitude_Rounded", "Longitude_Rounded"]).agg({
                "BusinessName": list,
                "Rating": list,
                "Address": list,
                "Color": "first",
                "Postcode": "first"
            }).reset_index()
            
            # Add cluster info
            clustered_data["Count"] = clustered_data["BusinessName"].apply(len)
            clustered_data["IsSingleVenue"] = clustered_data["Count"] == 1
            
            # Create tooltip text for clusters
            def create_cluster_tooltip(row):
                if row["Count"] == 1:
                    return f"<b>{row['BusinessName'][0]}</b><br/>Rating: {row['Rating'][0]}<br/>{row['Address'][0]}"
                else:
                    venues_text = "<br/>".join([f"• {name} (Rating: {rating})" 
                                              for name, rating in zip(row['BusinessName'], row['Rating'])])
                    return f"<b>{row['Count']} venues at this location:</b><br/>{venues_text}<br/><i>Zoom in to see individual venues</i>"
            
            clustered_data["TooltipText"] = clustered_data.apply(create_cluster_tooltip, axis=1)
            
            # Create expanded data for multi-venue clusters with individual nodes
            import math
            
            expanded_venues = []
            connecting_lines = []
            cluster_centers = []  # Track cluster centers for when zoomed in
            
            for _, cluster in clustered_data.iterrows():
                if cluster["Count"] > 1:
                    center_lat = cluster["Latitude_Rounded"]
                    center_lon = cluster["Longitude_Rounded"]
                    
                    # Add cluster center point for connecting lines
                    cluster_centers.append({
                        "Latitude": center_lat,
                        "Longitude": center_lon,
                        "Color": [200, 200, 200, 100]  # Gray center point
                    })
                    
                    # Create individual nodes in a circle around the center
                    for i, (name, rating, address) in enumerate(zip(
                        cluster["BusinessName"], cluster["Rating"], cluster["Address"]
                    )):
                        # Calculate position in circle around center
                        # Start from a small offset to avoid center overlap
                        angle = (2 * math.pi * i) / cluster["Count"]
                        radius = 0.0012  # Increased radius to ensure separation from center
                        
                        venue_lat = center_lat + radius * math.cos(angle)
                        venue_lon = center_lon + radius * math.sin(angle)
                        
                        expanded_venues.append({
                            "BusinessName": name,
                            "Rating": rating,
                            "Address": address,
                            "Latitude": venue_lat,
                            "Longitude": venue_lon,
                            "Color": rating_to_color(rating),
                            "TooltipText": f"<b>{name}</b><br/>Rating: {rating}<br/>{address}"
                        })
                        
                        # Create connecting line from center to venue
                        connecting_lines.append({
                            "start": [center_lon, center_lat],
                            "end": [venue_lon, venue_lat],
                            "color": [128, 128, 128, 180]
                        })
            
            expanded_venues_df = pd.DataFrame(expanded_venues)
            connecting_lines_df = pd.DataFrame(connecting_lines)
            cluster_centers_df = pd.DataFrame(cluster_centers)
            
            # Cluster layer (shows when zoomed out) - sized by number of venues
            cluster_layer = pdk.Layer(
                "ScatterplotLayer",
                data=clustered_data,
                get_position="[Longitude_Rounded, Latitude_Rounded]",
                get_fill_color="Color",
                get_radius="Count * 30 + 80",  # Size based on venue count
                radius_min_pixels=10,
                radius_max_pixels=60,
                pickable=True,
                auto_highlight=True,
                visible=True,
                max_zoom=cluster_zoom_threshold - 1
            )
            
            # Text layer for cluster counts (only show for multi-venue clusters)
            cluster_text_layer = pdk.Layer(
                "TextLayer",
                data=clustered_data[clustered_data["Count"] > 1],
                get_position="[Longitude_Rounded, Latitude_Rounded]",
                get_text="Count",
                get_size=14,
                get_color=[255, 255, 255, 255],
                get_alignment_baseline="center",
                visible=True,
                max_zoom=cluster_zoom_threshold - 1
            )

            # Connecting lines layer (shows when zoomed in)
            lines_layer = pdk.Layer(
                "LineLayer",
                data=connecting_lines_df,
                get_source_position="start",
                get_target_position="end",
                get_color="color",
                get_width=2,
                width_min_pixels=1,
                width_max_pixels=3,
                pickable=False,
                visible=True,
                min_zoom=cluster_zoom_threshold
            ) if not connecting_lines_df.empty else None

            # Expanded individual venues layer (shows when zoomed in)
            expanded_layer = pdk.Layer(
                "ScatterplotLayer",
                data=expanded_venues_df,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=60,
                radius_min_pixels=8,
                radius_max_pixels=20,
                pickable=True,
                auto_highlight=True,
                visible=True,
                min_zoom=cluster_zoom_threshold
            ) if not expanded_venues_df.empty else None

            # Cluster center points layer (small gray dots when zoomed in)
            center_points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=cluster_centers_df,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=15,
                radius_min_pixels=3,
                radius_max_pixels=8,
                pickable=False,
                auto_highlight=False,
                visible=True,
                min_zoom=cluster_zoom_threshold
            ) if not cluster_centers_df.empty else None

            # Individual venue layer for single venues (shows when zoomed in)
            single_venues = filtered_df.merge(
                clustered_data[clustered_data["Count"] == 1][["Latitude_Rounded", "Longitude_Rounded"]], 
                on=["Latitude_Rounded", "Longitude_Rounded"], 
                how="inner"
            )
            
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=single_venues,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=80,
                radius_min_pixels=6,
                radius_max_pixels=25,
                pickable=True,
                auto_highlight=True,
                visible=True,
                min_zoom=cluster_zoom_threshold
            )

            view_state = pdk.ViewState(
                latitude=filtered_df["Latitude"].mean(),
                longitude=filtered_df["Longitude"].mean(),
                zoom=14,
                pitch=30
            )

            # Enhanced tooltip for both individual venues and clusters
            tooltip = {
                "html": "{TooltipText}",
                "style": {"backgroundColor": "black", "color": "white", "fontSize": "12px"}
            }

            layers = [cluster_layer, cluster_text_layer, scatter_layer]
            
            # Add expanded venue layers when they exist
            if lines_layer is not None:
                layers.append(lines_layer)
            if expanded_layer is not None:
                layers.append(expanded_layer)
            if center_points_layer is not None:
                layers.append(center_points_layer)

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/streets-v12",
                initial_view_state=view_state,
                layers=layers,
                tooltip=tooltip
            ))
        else:
            st.warning("No establishments found for this postcode or rating range.")

    st.write("---")
                                    
    st.caption("Data: Food Standards Agency | STAT Data Intelligence Platform (STAT.co.uk)")

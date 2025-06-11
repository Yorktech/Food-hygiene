import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
import pydeck as pdk 
# https://ratings.food.gov.uk/api/open-data-files/FHRS868en-GB.xml



st.title("🍽️ Food Hygiene Ratings by Postcode")

# User input
postcode_input = st.text_input("Enter postcode to search:", "SW1A 1AA")

# Button to trigger API call and parse
if st.button("Get Ratings"):
    # URL for XML file
    url = "https://ratings.food.gov.uk/api/open-data-files/FHRS868en-GB.xml"

    # Get XML data
    st.write("📦 Downloading XML data...")
    response = requests.get(url)

    if response.status_code == 200:
        # Parse XML content
        st.write("🔍 Parsing XML data...")
        root = ET.fromstring(response.content)

        # Extract relevant data
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

            # Safe float conversion
            try:
                lat_float = float(lat)
            except (TypeError, ValueError):
                lat_float = None

            try:
                lon_float = float(lon)
            except (TypeError, ValueError):
                lon_float = None

            # Append only if essential fields exist
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
        df = pd.DataFrame(establishments)
        # Filter by postcode
        filtered_df = df[df["Postcode"] == postcode_input.upper()]
  # Center map on the mean location
        midpoint = (filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean())

            # Pydeck layer for markers
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position='[Longitude, Latitude]',
            get_radius=50,
            get_fill_color='[200, 30, 0, 160]',
            pickable=True
        )

        # Pydeck view
        view_state = pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=14,
            pitch=0
        )

        # Pydeck tooltip
        tooltip = {
            "html": "<b>{BusinessName}</b><br/>Rating: {Rating}<br/>{Address}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

        # Render map
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v12",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ))
        if not filtered_df.empty:
            st.write(f"✅ Found {len(filtered_df)} establishments in **{postcode_input.upper()}**:")
            st.dataframe(filtered_df)
        else:
            st.warning("No establishments found for this postcode. Try another.")

    else:
        st.error("❌ Couldn’t download the XML data. Please check the URL or try again.")

# Footer
st.write("---")
st.write("🔗 Data from the UK Food Standards Agency ([link](https://ratings.food.gov.uk/open-data/en-GB))")
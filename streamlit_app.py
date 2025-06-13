import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="FHRS Hygiene Map", layout="wide")
st.title("Food Hygiene Ratings by Postcode and Postcode Area (UK)")

# Input
postcode_input = st.text_input("Enter a UK postcode or postcode area:", "SW1A 1AA")
st.caption("Enter a full postcode (e.g., SW1A 1AA) or postcode area (e.g., YO1, M1, SW1) to see all establishments in that area")
min_rating = st.slider("Minimum Rating", 0, 5, 0)

# Initialize session state for establishment types if not exists
if 'establishment_types' not in st.session_state:
    st.session_state.establishment_types = []

# Fetch establishment types without triggering full data processing
if not st.session_state.establishment_types:
    st.write("Loading establishment types...")
    url = "https://ratings.food.gov.uk/api/open-data-files/FHRS868en-GB.xml"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            types = set()
            for est in root.findall(".//EstablishmentDetail"):
                est_type = est.findtext("BusinessType")
                if est_type:
                    types.add(est_type)
            st.session_state.establishment_types = sorted(list(types))
    except Exception as e:
        st.error(f"Failed to load establishment types: {str(e)}")
        st.session_state.establishment_types = []

# Add establishment type dropdown
selected_type = st.selectbox(
    "Select establishment type:",
    ["All Types"] + st.session_state.establishment_types
)

# Slider to control zoom threshold for clustering
cluster_zoom_threshold = st.slider("Cluster Zoom Threshold", 10, 20, 15)

# Fetch + Parse XML on button press
if st.button("Get Ratings and Map"):
    url = "https://ratings.food.gov.uk/api/open-data-files/FHRS868en-GB.xml"
    status_message = st.empty()
    status_message.write(" Downloading and parsing data....")

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
            business_type = est.findtext("BusinessType")

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
                    "BusinessType": business_type,
                    "Latitude": lat_float,
                    "Longitude": lon_float
                })

        # Build DataFrame
        df = pd.DataFrame(establishments)
        df.dropna(subset=["Latitude", "Longitude", "Rating"], inplace=True)
        df["PostcodeClean"] = df["Postcode"].str.replace(" ", "").str.upper()
        df = df[df["Rating"].apply(lambda x: x.isdigit() and int(x) >= min_rating)]

        # Calculate national failure rate
        df['NumericRating'] = pd.to_numeric(df['Rating'], errors='coerce')
        national_failures = (df['NumericRating'] <= 2).sum()
        national_total = len(df['NumericRating'])
        national_fail_percentage = round((national_failures / national_total) * 100, 1) if national_total > 0 else 0

        # Filter by establishment type if selected
        if selected_type != "All Types":
            df = df[df["BusinessType"] == selected_type]

        postcode_cleaned = postcode_input.replace(" ", "").upper()
        
        # Function to extract postcode area using string operations
        def extract_postcode_area(postcode):
            try:
                # Convert to uppercase and split by space if exists
                postcode = postcode.upper()
                outward_code = postcode.split()[0] if ' ' in postcode else postcode
                
                # Find where the numbers start in the outward code
                for i, char in enumerate(outward_code):
                    if char.isdigit():
                        letters = outward_code[:i]
                        numbers = outward_code[i:]
                        return letters, numbers
                return None, None
            except:
                return None, None

        # Function to normalize postcode area
        def normalize_postcode_area(postcode):
            letters, numbers = extract_postcode_area(postcode)
            if letters and numbers:
                return letters + numbers[0]
            return None

        # Function to parse postcode area (for YO1 vs YO19 distinction)
        def parse_postcode_area(postcode):
            letters, numbers = extract_postcode_area(postcode)
            if letters and numbers:
                # Check if it's a double digit district (like YO19)
                if len(numbers) >= 2 and numbers[1].isdigit():
                    area = letters + numbers[:2]
                    return (area, True)
                else:
                    # Single digit district (like YO1)
                    area = letters + numbers[0]
                    return (area, False)
            return (None, False)

        # Get the standardized area and whether it's a double-digit district
        input_area, is_double_digit = parse_postcode_area(postcode_cleaned)
        
        # If it's a valid postcode area format or short enough to be one, treat as area search
        if input_area or len(postcode_cleaned) <= 4:
            def process_postcode_for_comparison(postcode):
                area, is_double = parse_postcode_area(postcode)
                return (area, is_double)
            
            # Process all postcodes
            temp = df['Postcode'].apply(process_postcode_for_comparison)
            df['ProcessedArea'] = temp.apply(lambda x: x[0])
            df['IsDoubleDigit'] = temp.apply(lambda x: x[1])
            
            if input_area:
                if is_double_digit:
                    # For YO19, only match exact YO19 patterns
                    filtered_df = df[
                        (df['ProcessedArea'] == input_area) & 
                        (df['IsDoubleDigit'] == True)
                    ].copy()
                else:
                    # For YO1, match YO1 but not YO19
                    filtered_df = df[
                        (df['ProcessedArea'] == input_area) & 
                        (df['IsDoubleDigit'] == False)
                    ].copy()
            else:
                # For very short inputs (like 'YO'), use prefix matching
                filtered_df = df[df['PostcodeClean'].str.startswith(postcode_cleaned)].copy()
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
            status_message.empty()
            st.success(f"Found {len(filtered_df)} establishments in {search_type} {postcode_input.upper()}")
            
            # Calculate average rating for current search and sub-areas all at once
            postcode_region = postcode_cleaned[:2].upper()  # Take first two characters of postcode
            
            # Use parse_postcode_area for sub-areas
            region_df = df[df["Postcode"].str.upper().str.startswith(postcode_region)].copy()
            # Get both the area and whether it's double digit
            area_info = region_df['Postcode'].apply(parse_postcode_area)
            region_df['SubArea'] = area_info.apply(lambda x: x[0])  # Get just the area part
            region_df['NumericRating'] = pd.to_numeric(region_df['Rating'], errors='coerce')
            region_df = region_df[region_df['NumericRating'].notna() & 
                                region_df['SubArea'].notna()]

            # Calculate and display current search statistics
            search_ratings = pd.to_numeric(filtered_df['Rating'], errors='coerce')
            search_ratings = search_ratings[search_ratings.notna()]
            if not search_ratings.empty:
                avg_rating = round(search_ratings.mean(), 2)
                st.info(f"Average hygiene rating for this search: {avg_rating}/5 ({len(search_ratings)} establishments)")
                
                # Calculate percentage of failed establishments (rating 0-2)
                search_ratings_array = search_ratings.to_numpy()
                failed_count = len(search_ratings_array[search_ratings_array <= 2])
                fail_percentage = round((failed_count / len(search_ratings_array)) * 100, 1)

                # Calculate area (e.g., YO) failure rate
                area_postcode = postcode_cleaned[:2].upper()
                area_df = df[df["Postcode"].str.upper().str.startswith(area_postcode)]
                area_failures = (area_df['NumericRating'] <= 2).sum()
                area_total = len(area_df['NumericRating'])
                area_fail_percentage = round((area_failures / area_total) * 100, 1) if area_total > 0 else 0

                # Compare with national and area averages
                if fail_percentage > 0:
                    message_parts = [f"{fail_percentage}% of establishments failed hygiene standards in your search"]
                    
                    # Calculate percentage differences
                    if national_fail_percentage > 0:
                        nat_diff_percent = round(((fail_percentage - national_fail_percentage) / national_fail_percentage) * 100, 1)
                        if nat_diff_percent != 0:
                            comp_text = f"{abs(nat_diff_percent)}% {'higher' if nat_diff_percent > 0 else 'lower'} than the national average of {national_fail_percentage}%"
                            message_parts.append(comp_text)
                    
                    if area_fail_percentage > 0 and area_fail_percentage != fail_percentage:
                        area_diff_percent = round(((fail_percentage - area_fail_percentage) / area_fail_percentage) * 100, 1)
                        if area_diff_percent != 0:
                            comp_text = f"{abs(area_diff_percent)}% {'higher' if area_diff_percent > 0 else 'lower'} than the {area_postcode} region average of {area_fail_percentage}%"
                            message_parts.append(comp_text)
                    
                    message = " - ".join(message_parts)
                    
                    if fail_percentage > 4:
                        st.error(message)
                    else:
                        st.warning(message)
                else:
                    st.success("No establishments failed hygiene standards (rated 0-2)")

            # Calculate and display region statistics
            if not region_df.empty:
                region_avg = round(region_df['NumericRating'].mean(), 2)
                st.info(f"Average hygiene rating for Your search in the {postcode_region} region: {region_avg}/5 (Total establishments: {len(region_df)})")

                # Calculate sub-area statistics
                sub_area_stats = []
                for sub_area, group in region_df.groupby('SubArea'):
                    if len(group) >= 5:  # Only include areas with at least 5 establishments
                        stats = {
                            'sub_area': sub_area,
                            'average': round(group['NumericRating'].mean(), 2),
                            'count': len(group)
                        }
                        sub_area_stats.append(stats)

                # Display best and worst areas
                if len(sub_area_stats) >= 2:
                    st.write("---")
                    st.write("Across all postcode areas in this region:")
                    
                    # Sort by average rating for consistent results in case of ties
                    sub_area_stats.sort(key=lambda x: (x['average'], x['count']), reverse=True)
                    best_area = sub_area_stats[0]
                    worst_area = sub_area_stats[-1]
                    
                    st.info(f"Best performing area: {best_area['sub_area']} - Average rating: {best_area['average']}/5 ({best_area['count']} establishments)")
                    st.info(f"Lowest performing area: {worst_area['sub_area']} - Average rating: {worst_area['average']}/5 ({worst_area['count']} establishments)")
                elif len(sub_area_stats) == 1:
                    st.info("Only one valid postcode area found in this region")

            st.dataframe(filtered_df[["BusinessName", "Rating", "Address", "Postcode", "Latitude", "Longitude"]])

            # Create bar chart of ratings distribution
            st.write("---")
            st.write("Distribution of Hygiene Ratings in Search Results:")
            
            # Calculate the count of each rating
            rating_counts = filtered_df['Rating'].value_counts().sort_index()
            
            # Create bar chart using Plotly
            import plotly.express as px
            
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Hygiene Rating', 'y': 'Number of Establishments'},
                color=rating_counts.values,
                color_continuous_scale=['red', 'orange', 'yellow', 'yellowgreen', 'green', 'darkgreen']
            )
            
            # Update layout for better appearance
            fig.update_layout(
                showlegend=False,
                xaxis_title="Hygiene Rating",
                yaxis_title="Number of Establishments",
                coloraxis_showscale=False,
                height=300,  # Compact height
                margin=dict(l=0, r=0, t=20, b=0)  # Minimal margins
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
                if cluster["Count"] > 1:  # Only create wheel layout for multiple venues
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
            
            # Cluster layer for multiple venues only (shows when zoomed out)
            multi_venue_clusters = clustered_data[clustered_data["Count"] > 1].copy()
            cluster_layer = pdk.Layer(
                "ScatterplotLayer",
                data=multi_venue_clusters,
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
                data=multi_venue_clusters,
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
            single_venues = clustered_data[clustered_data["Count"] == 1].copy()
            single_venues["Latitude"] = single_venues["Latitude_Rounded"]
            single_venues["Longitude"] = single_venues["Longitude_Rounded"]
            
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=single_venues,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=60,
                radius_min_pixels=8,
                radius_max_pixels=15,
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

            # Initialize layers list based on what we're showing
            layers = []
            
            # Add cluster-related layers for multi-venue locations
            if not multi_venue_clusters.empty:
                layers.extend([cluster_layer, cluster_text_layer])
                
                # Add expanded venue layers when they exist
                if lines_layer is not None:
                    layers.append(lines_layer)
                if expanded_layer is not None:
                    layers.append(expanded_layer)
                if center_points_layer is not None:
                    layers.append(center_points_layer)
            
            # Add single venue layer
            layers.append(scatter_layer)

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/streets-v12",
                initial_view_state=view_state,
                layers=layers,
                tooltip=tooltip
            ))
        else:
            status_message.empty()  # Clear the downloading message
            st.warning("No establishments found for this postcode or rating range.")

    st.write("---")
                                    
    st.caption("Data: Food Standards Agency | STAT Data Intelligence Platform (STAT.co.uk)")

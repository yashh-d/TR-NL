import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import io
import imageio
from PIL import Image
import base64
import numpy as np
from scipy.interpolate import interp1d
import math

# At the beginning of your file, after the imports, add this custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: bold;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .section-header {
        font-size: 24px !important;
        font-weight: bold;
        margin-top: 10px !important;
        margin-bottom: 5px !important;
        padding-top: 10px !important;
        padding-bottom: 0px !important;
    }
    .subsection-header {
        font-size: 20px !important;
        font-weight: bold;
        margin-top: 5px !important;
        margin-bottom: 0px !important;
        padding-top: 5px !important;
        padding-bottom: 0px !important;
    }
    /* Reduce spacing between elements */
    .stSlider, .stCheckbox, .stRadio, .stSelectbox {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

def format_y_tick(value, pos, use_dollar=False):
    """Format y-axis ticks with K, M, B suffixes and optional dollar sign."""
    if value == 0:
        return '$0' if use_dollar else '0'
    magnitude = 0
    while abs(value) >= 1000:
        magnitude += 1
        value /= 1000.0
    prefix = '$' if use_dollar else ''
    return f'{prefix}{value:.1f}{["", "K", "M", "B", "T"][magnitude]}'

def save_frames_as_gif(fig, frames, speed, width=1200, height=800):
    images = []
    try:
        for frame in frames:
            fig.update(data=frame.data)
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=1.0)
            img_array = imageio.imread(io.BytesIO(img_bytes))
            images.append(img_array)
        
        gif_buffer = io.BytesIO()
        imageio.mimsave(gif_buffer, images, format='gif', duration=speed, loop=0)
        gif_buffer.seek(0)
        return gif_buffer
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return None

def create_animated_graph(df, date_col, y_cols, colors, title, subtitle="", speed=0.1, theme="dark", 
                         xaxis_title="", yaxis_title="", logo_file=None, logo_x=0.95, logo_y=0.05, 
                         show_legend=True, legend_x=0.9, legend_y=0.9, format_numbers=False, use_dollar=False, 
                         legend_labels=None, title_size=18, subtitle_size=14, axis_label_size=14, 
                         tick_label_size=12, legend_font_size=10):
    fig = go.Figure()
    frames = []
    for i in range(len(df)):
        frame_data = []
        for col in y_cols:
            # Use custom legend label if provided, otherwise use column name
            display_name = legend_labels.get(col, col) if legend_labels else col
            
            frame_data.append(go.Scatter(
                x=df[date_col][:i+1], 
                y=df[col][:i+1], 
                mode='lines',  # Remove markers from the line
                name=display_name, 
                line=dict(color=colors[col]),
                showlegend=True
            ))
            frame_data.append(go.Scatter(
                x=[df[date_col][i]], 
                y=[df[col][i]], 
                mode='markers',  # Only show marker for last point
                marker=dict(color=colors[col], size=8, symbol='circle'),
                showlegend=False  # Prevents duplicate legend items
            ))
        frames.append(go.Figure(data=frame_data))
    
    layout_settings = {
        "plot_bgcolor": "black" if theme == "dark" else "white",
        "paper_bgcolor": "black" if theme == "dark" else "white",
        "font": {"color": "white" if theme == "dark" else "black"},
        "title": {
            "text": title, 
            "x": 0.5,
            "font": {"size": title_size}
        },
        "annotations": [] if not subtitle else [
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "bottom",
                "font": {
                    "size": subtitle_size,
                    "color": "white" if theme == "dark" else "black"
                },
                "showarrow": False
            }
        ],
        "xaxis_title": xaxis_title,
        "xaxis": {
            "range": [df[date_col].min(), df[date_col].max()],
            "title": {"font": {"size": axis_label_size}},
            "tickfont": {"size": tick_label_size}
        },
        "yaxis_title": yaxis_title,
        "yaxis": {
            "title": {"font": {"size": axis_label_size}},
            "tickfont": {"size": tick_label_size}
        },
        "showlegend": show_legend,
        "legend": {
            "x": legend_x, 
            "y": legend_y,
            "font": {"size": legend_font_size}
        },
        "margin": {"r": 150},
    }
    
    if format_numbers:
        tickprefix = '$' if use_dollar else ''
        layout_settings["yaxis"] = {
            "tickformat": ".2s",  # Use SI prefix formatting (k, M, G, etc.)
            "tickprefix": tickprefix,  # Add dollar sign prefix if requested
            "ticksuffix": ""  # This removes the default SI suffix so we can customize
        }
    
    fig.update_layout(**layout_settings)
    
    if logo_file:
        logo_image = Image.open(logo_file)
        logo_bytes = io.BytesIO()
        logo_image.save(logo_bytes, format='PNG')
        logo_base64 = base64.b64encode(logo_bytes.getvalue()).decode()
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{logo_base64}',
                xref="paper",
                yref="paper",
                x=logo_x,
                y=logo_y,
                sizex=0.1,
                sizey=0.1,
                xanchor="right",
                yanchor="bottom"
            )
        )
    
    gif_buffer = save_frames_as_gif(fig, frames, speed)
    
    if gif_buffer:
        gif_data = base64.b64encode(gif_buffer.read()).decode("utf-8")
        st.markdown(f'<img src="data:image/gif;base64,{gif_data}" alt="Animated Graph">', unsafe_allow_html=True)
        st.download_button("Download GIF", gif_buffer, file_name="animated_graph.gif", mime="image/gif")

def create_static_graph(plot_data, date_col, y_cols, colors, title, subtitle="", xaxis_title="", yaxis_title="", theme="dark", 
                       logo_file=None, logo_x=0.95, logo_y=0.05, show_legend=True, legend_x=0.9, legend_y=0.9, 
                       format_numbers=False, use_dollar=False, legend_labels=None, title_size=18, subtitle_size=14,
                       axis_label_size=14, tick_label_size=12, legend_font_size=10):
    fig, ax = plt.subplots(figsize=(12, 8))
    if theme == "dark":
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        text_color = "white"
        legend_facecolor = "black"
    else:
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        text_color = "black"
        legend_facecolor = "white"
    
    for col in y_cols:
        # Use custom legend label if provided, otherwise use column name
        display_name = legend_labels.get(col, col) if legend_labels else col
        ax.plot(plot_data[date_col], plot_data[col], color=colors[col], linewidth=2, label=display_name)
    
    ax.set_title(title, fontsize=title_size, color=text_color)
    
    # Add subtitle if provided
    if subtitle:
        # Add subtitle with smaller font below the title
        fig.text(0.5, 0.95, subtitle, ha='center', fontsize=subtitle_size, color=text_color)
        # Adjust the figure to make room for subtitle
        plt.subplots_adjust(top=0.85)
    
    ax.set_xlabel(xaxis_title, fontsize=axis_label_size, color=text_color)
    ax.set_ylabel(yaxis_title, fontsize=axis_label_size, color=text_color)
    ax.tick_params(colors=text_color, labelsize=tick_label_size)
    
    # Apply number formatting if requested
    if format_numbers:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_y_tick(x, pos, use_dollar)))
    
    if show_legend:
        ax.legend(loc='upper left', fontsize=legend_font_size, facecolor=legend_facecolor, 
                 edgecolor=text_color, labelcolor=text_color, bbox_to_anchor=(legend_x, legend_y))
    
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if logo_file:
        logo_image = Image.open(logo_file)
        newax = fig.add_axes([logo_x, logo_y, 0.1, 0.1], anchor='NE', zorder=1)
        newax.imshow(logo_image)
        newax.axis('off')
    
    st.pyplot(fig)

def create_bar_graph(plot_data, date_col, y_cols, colors, title, subtitle="", xaxis_title="", yaxis_title="", theme="dark", 
                    logo_file=None, logo_x=0.95, logo_y=0.05, show_legend=True, legend_x=0.9, legend_y=0.9, 
                    format_numbers=False, use_dollar=False, legend_labels=None, title_size=18, subtitle_size=14,
                    axis_label_size=14, tick_label_size=12, legend_font_size=10):
    """Create a bar graph with the selected data."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set theme colors
    if theme == "dark":
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        text_color = "white"
        legend_facecolor = "black"
        grid_color = 'gray'
        spine_color = '#333333'  # Darker gray for spines in dark mode
    else:
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        text_color = "black"
        legend_facecolor = "white"
        grid_color = 'lightgray'
        spine_color = '#e0e0e0'  # Very light gray for spines in light mode
    
    # Make the border (spines) lighter
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.5)  # Thinner border
    
    # For bar charts, we need to handle the x-axis differently
    # Convert dates to strings for better display
    if pd.api.types.is_datetime64_any_dtype(plot_data[date_col]):
        x_values = plot_data[date_col].dt.strftime('%Y-%m-%d')
    else:
        x_values = plot_data[date_col]
    
    # Calculate bar width based on number of columns
    bar_width = 0.8 / len(y_cols)
    
    # Plot each column as a set of bars
    for i, col in enumerate(y_cols):
        # Calculate position for this set of bars
        positions = np.arange(len(x_values)) - (0.4 - bar_width/2) + i * bar_width
        
        # Use custom legend label if provided, otherwise use column name
        display_name = legend_labels.get(col, col) if legend_labels else col
        
        # Create bars with lighter edge color
        ax.bar(positions, plot_data[col], width=bar_width, color=colors[col], 
               label=display_name, edgecolor=spine_color, linewidth=0.5)
    
    # Set x-axis ticks at the center of each group
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels(x_values, rotation=45, ha='right')
    
    # Set title and labels
    ax.set_title(title, fontsize=title_size, color=text_color)
    
    # Add subtitle if provided
    if subtitle:
        # Add subtitle with smaller font below the title
        fig.text(0.5, 0.95, subtitle, ha='center', fontsize=subtitle_size, color=text_color)
        # Adjust the figure to make room for subtitle
        plt.subplots_adjust(top=0.85)
    
    ax.set_xlabel(xaxis_title, fontsize=axis_label_size, color=text_color)
    ax.set_ylabel(yaxis_title, fontsize=axis_label_size, color=text_color)
    ax.tick_params(colors=text_color, labelsize=tick_label_size)
    
    # Apply number formatting if requested
    if format_numbers:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_y_tick(x, pos, use_dollar)))
    
    # Add legend
    if show_legend:
        ax.legend(loc='upper left', fontsize=legend_font_size, facecolor=legend_facecolor, 
                 edgecolor=spine_color, labelcolor=text_color, bbox_to_anchor=(legend_x, legend_y))
    
    # Add grid (only horizontal) with lighter color
    ax.grid(axis='y', color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add logo if provided
    if logo_file:
        logo_image = Image.open(logo_file)
        newax = fig.add_axes([logo_x, logo_y, 0.1, 0.1], anchor='NE', zorder=1)
        newax.imshow(logo_image)
        newax.axis('off')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Display the figure
    st.pyplot(fig)

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Replace the title with custom HTML
st.markdown('<p class="main-title">Animated and Static Graph Plotter with GIF Download</p>', unsafe_allow_html=True)

# Modified file upload section to handle multiple CSVs
num_files = st.number_input("Number of CSV files to upload", min_value=1, max_value=10, value=1)
uploaded_files = []
dataframes = []
file_names = []

# First, collect all files
for i in range(num_files):
    file = st.file_uploader(f"Upload CSV #{i+1}", type=["csv"], key=f"file_{i}")
    if file:
        uploaded_files.append(file)
        try:
            # Read with pandas explicitly specifying encoding
            df = pd.read_csv(file, encoding='utf-8-sig')
            
            # Debug information
            st.write(f"File {i+1} columns: {df.columns.tolist()}")
            st.write(f"First few rows:")
            st.write(df.head())
            
            # Store in dataframes list
            dataframes.append(df)
            file_names.append(file.name)
        except Exception as e:
            st.error(f"Error reading file #{i+1}: {str(e)}")

# Add logo and theme options
logo_file = st.file_uploader("Upload a Logo (Optional, PNG/JPG)", type=["png", "jpg", "jpeg"])
theme = st.radio("Select Theme", ["dark", "light"], index=0)

# Add text formatting options
st.markdown('<p class="section-header">Text Formatting</p>', unsafe_allow_html=True)
title_size = st.slider("Title Font Size", 10, 36, 18, 1)
axis_label_size = st.slider("Axis Label Font Size", 8, 24, 14, 1)
tick_label_size = st.slider("Tick Label Font Size", 8, 20, 12, 1)
legend_font_size = st.slider("Legend Font Size", 8, 20, 10, 1)

# Add animation and logo options
animation_speed = st.slider("Select Animation Speed (seconds per frame)", 0.05, 1.0, 0.1, 0.05)
logo_x = st.slider("Logo X Position", 0.0, 1.0, 0.95, 0.01)
logo_y = st.slider("Logo Y Position", 0.0, 1.0, 0.05, 0.01)

# Add legend options
show_legend = st.checkbox("Show Legend", value=True)
legend_x = st.slider("Legend X Position", 0.0, 1.0, 0.9, 0.01)
legend_y = st.slider("Legend Y Position", 0.0, 1.0, 0.9, 0.01)

# Add title and subtitle options
st.markdown('<p class="section-header">Title and Subtitle</p>', unsafe_allow_html=True)
custom_title = st.text_input("Graph Title", "Blockchain Comparison")
custom_subtitle = st.text_input("Graph Subtitle (Optional)", "")
subtitle_size = st.slider("Subtitle Font Size", 8, 24, 14, 1)

# Add this after the "Plot Settings" section, before the "Generate Static Graph" button
graph_type = st.radio("Select Graph Type", ["Line", "Bar"], index=0)

# Now process the dataframes if we have any
if len(dataframes) > 0:
    try:
        # Create a container for date column selection
        date_cols = {}
        st.markdown('<p class="subsection-header">Date Column Selection</p>', unsafe_allow_html=True)
        
        # Process each dataframe
        for i, df in enumerate(dataframes):
            date_cols[i] = st.selectbox(f"Select date column for {file_names[i]}", 
                                      df.columns, key=f"date_col_{i}")
            
            # Convert date column to datetime
            dataframes[i][date_cols[i]] = pd.to_datetime(dataframes[i][date_cols[i]], errors="coerce")
            
            # Convert numeric columns after date column is selected
            for col in df.columns:
                # Skip date column
                if col != date_cols[i]:
                    # Try to convert to numeric, replacing commas
                    try:
                        # First remove commas and convert to numeric
                        dataframes[i][col] = dataframes[i][col].astype(str).str.replace(',', '').astype(float)
                        st.success(f"Converted column {col} to numeric")
                    except Exception as e:
                        # If conversion fails, it might not be a numeric column
                        pass
            
            # Sort the dataframe by the selected date column
            dataframes[i] = dataframes[i].sort_values(by=date_cols[i])
            st.success(f"Data in {file_names[i]} sorted by {date_cols[i]}")
            
            # Show the date range
            min_date = dataframes[i][date_cols[i]].min()
            max_date = dataframes[i][date_cols[i]].max()
            st.info(f"Date range: {min_date} to {max_date}")
            
            # After showing the date range, add date range filter
            if st.checkbox(f"Filter date range for {file_names[i]}", key=f"date_filter_{i}"):
                # Create date range selector
                date_range = st.date_input(
                    f"Select date range for {file_names[i]}",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key=f"date_range_{i}"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    
                    # Check if the datetime column has timezone info
                    has_tz = pd.notnull(dataframes[i][date_cols[i]].iloc[0]) and dataframes[i][date_cols[i]].iloc[0].tzinfo is not None
                    
                    # Convert to datetime for filtering with appropriate timezone handling
                    if has_tz:
                        # Get the timezone from the data
                        sample_tz = dataframes[i][date_cols[i]].iloc[0].tzinfo
                        start_datetime = pd.Timestamp(start_date).tz_localize('UTC')
                        end_datetime = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize('UTC')
                    else:
                        # No timezone in the data
                        start_datetime = pd.Timestamp(start_date)
                        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    
                    # Filter the dataframe
                    filtered_df = dataframes[i][
                        (dataframes[i][date_cols[i]] >= start_datetime) & 
                        (dataframes[i][date_cols[i]] <= end_datetime)
                    ]
                    
                    # Replace the dataframe with filtered version
                    dataframes[i] = filtered_df
                    
                    st.success(f"Filtered {file_names[i]} to date range: {start_date} to {end_date}")
                    st.write(f"Rows after filtering: {len(filtered_df)}")
            
            # Add segmentation by string values in columns
            st.markdown('<p class="subsection-header">Data Segmentation (Optional)</p>', unsafe_allow_html=True)
            segment_data = {}
            segment_columns = {}
            segment_values = {}
            
            for i, df in enumerate(dataframes):
                # Check if user wants to segment this dataframe
                if st.checkbox(f"Segment data in {file_names[i]}", key=f"segment_checkbox_{i}"):
                    # Get string columns (object type)
                    string_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if string_cols:
                        # Let user select which column to use for segmentation
                        segment_columns[i] = st.selectbox(
                            f"Select column to segment by in {file_names[i]}", 
                            string_cols,
                            key=f"segment_col_{i}"
                        )
                        
                        # Get unique values in the selected column
                        unique_values = df[segment_columns[i]].unique().tolist()
                        
                        # Let user select which values to include
                        segment_values[i] = st.multiselect(
                            f"Select values to include from {segment_columns[i]} in {file_names[i]}", 
                            unique_values,
                            default=unique_values[:min(3, len(unique_values))],  # Default to first 3 values
                            key=f"segment_values_{i}"
                        )
                        
                        # Create segmented dataframes
                        segment_data[i] = {}
                        for value in segment_values[i]:
                            segment_data[i][value] = df[df[segment_columns[i]] == value].copy()
                            
                        # Display segmented data in tabs
                        segment_tabs = st.tabs([f"{value}" for value in segment_values[i]])
                        for j, (value, tab) in enumerate(zip(segment_values[i], segment_tabs)):
                            with tab:
                                st.write(f"Data for {segment_columns[i]} = {value}")
                                st.dataframe(segment_data[i][value])
                    else:
                        st.warning(f"No string columns found in {file_names[i]} for segmentation")
            
            # Create sliders for row ranges for each dataframe
            row_ranges = {}
            st.markdown('<p class="subsection-header">Row Range Selection</p>', unsafe_allow_html=True)
            for i, df in enumerate(dataframes):
                row_ranges[i] = st.slider(f"Row range for {file_names[i]}", 
                                         0, len(df) - 1, 
                                         (0, min(50, len(df) - 1)), 
                                         key=f"row_range_{i}")
            
            # Create sub-dataframes based on selected row ranges
            sub_dfs = {}
            for i, df in enumerate(dataframes):
                rs, re = row_ranges[i]
                # If segmentation is active for this dataframe, use the segmented data
                if i in segment_data:
                    # Create a combined dataframe with all selected segments
                    combined_segments = pd.DataFrame()
                    for value in segment_values[i]:
                        segment_df = segment_data[i][value]
                        # Apply row range to segmented data
                        segment_df = segment_df.iloc[min(rs, len(segment_df)-1):min(re+1, len(segment_df))]
                        # Add a new column to identify the segment
                        segment_df = segment_df.copy()
                        segment_df[f'_segment_{segment_columns[i]}'] = value
                        combined_segments = pd.concat([combined_segments, segment_df])
                    
                    sub_dfs[i] = combined_segments
                else:
                    # Use the original dataframe with row range
                    sub_dfs[i] = df.iloc[rs:re + 1].copy()
            
            # Display the sub-dataframes in tabs
            tabs = st.tabs([f"Data {i+1}: {name}" for i, name in enumerate(file_names)])
            for i, tab in enumerate(tabs):
                with tab:
                    st.dataframe(sub_dfs[i])
            
            # Column selection for each dataframe
            st.markdown('<p class="subsection-header">Column Selection</p>', unsafe_allow_html=True)
            y_cols_by_file = {}
            for i, df in enumerate(sub_dfs.values()):
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                y_cols_by_file[i] = st.multiselect(f"Select numeric columns to plot from {file_names[i]}", 
                                                  numeric_cols, 
                                                  key=f"y_cols_{i}")
            
            # Check if any columns are selected
            any_cols_selected = any(len(cols) > 0 for cols in y_cols_by_file.values())
            
            if any_cols_selected:
                normalize = st.checkbox("Normalize Values", value=False)
                format_numbers = st.checkbox("Format Y-axis (1K, 1M, 1B)", value=False)
                use_dollar = st.checkbox("Add Dollar Sign ($) to Y-axis", value=False)
                
                # Color picker for each selected column in each file
                colors = {}
                st.markdown('<p class="section-header">Color Selection</p>', unsafe_allow_html=True)
                for file_idx, cols in y_cols_by_file.items():
                    for col in cols:
                        # If segmentation is active, create color pickers for each segment
                        if file_idx in segment_data:
                            for value in segment_values[file_idx]:
                                color_key = f"{file_idx}_{col}_{value}"
                                colors[color_key] = st.color_picker(
                                    f"Pick color for {file_names[file_idx]} - {col} - {value}", 
                                    "#" + ''.join([hex(hash(color_key + str(i)) % 256)[2:].zfill(2) for i in range(3)])
                                )
                        else:
                            color_key = f"{file_idx}_{col}"
                            colors[color_key] = st.color_picker(
                                f"Pick color for {file_names[file_idx]} - {col}", 
                                "#" + ''.join([hex(hash(color_key + str(i)) % 256)[2:].zfill(2) for i in range(3)])
                            )
                
                # Custom legend labels section
                use_custom_labels = st.checkbox("Use Custom Legend Labels", value=False)
                legend_labels = {}
                if use_custom_labels:
                    st.write("Enter custom legend labels:")
                    for file_idx, cols in y_cols_by_file.items():
                        for col in cols:
                            # If segmentation is active, create label inputs for each segment
                            if file_idx in segment_data:
                                for value in segment_values[file_idx]:
                                    label_key = f"{file_idx}_{col}_{value}"
                                    legend_labels[label_key] = st.text_input(
                                        f"Custom label for {file_names[file_idx]} - {col} - {value}", 
                                        value=f"{file_names[file_idx]}: {col} ({value})",
                                        key=f"label_{label_key}"
                                    )
                            else:
                                label_key = f"{file_idx}_{col}"
                                legend_labels[label_key] = st.text_input(
                                    f"Custom label for {file_names[file_idx]} - {col}", 
                                    value=f"{file_names[file_idx]}: {col}",
                                    key=f"label_{label_key}"
                                )
                
                custom_xaxis_title = st.text_input("Enter the X-axis title", "Date")
                
                y_axis_title_default = "Normalized Values" if normalize else "Values"
                if use_dollar and not normalize:
                    y_axis_title_default = "Dollar Values"
                custom_yaxis_title = st.text_input("Enter the Y-axis title", y_axis_title_default)
                
                # Normalize data if requested
                if normalize:
                    for file_idx, df in sub_dfs.items():
                        if y_cols_by_file[file_idx]:
                            sub_dfs[file_idx][y_cols_by_file[file_idx]] = normalize_data(df[y_cols_by_file[file_idx]])
                
                # Prepare combined dataframe for plotting
                if st.button("Generate Static Graph"):
                    if graph_type == "Line":
                        # Existing line graph code
                        fig, ax = plt.subplots(figsize=(12, 8))
                        if theme == "dark":
                            ax.set_facecolor("black")
                            fig.patch.set_facecolor("black")
                            text_color = "white"
                            legend_facecolor = "black"
                        else:
                            ax.set_facecolor("white")
                            fig.patch.set_facecolor("white")
                            text_color = "black"
                            legend_facecolor = "white"
                        
                        # Plot each selected column from each file
                        for file_idx, cols in y_cols_by_file.items():
                            df = sub_dfs[file_idx]
                            date_col = date_cols[file_idx]
                            
                            # Check if this dataframe has segmentation
                            if file_idx in segment_data:
                                segment_col = f'_segment_{segment_columns[file_idx]}'
                                
                                for col in cols:
                                    # Plot each segment separately
                                    for value in segment_values[file_idx]:
                                        # Filter data for this segment
                                        segment_df = df[df[segment_col] == value]
                                        if not segment_df.empty:
                                            color_key = f"{file_idx}_{col}_{value}"
                                            # Use custom legend label if provided
                                            display_name = legend_labels.get(color_key, f"{file_names[file_idx]}: {col} ({value})") if use_custom_labels else f"{file_names[file_idx]}: {col} ({value})"
                                            ax.plot(segment_df[date_col], segment_df[col], color=colors[color_key], linewidth=2, label=display_name)
                            else:
                                # Plot without segmentation
                                for col in cols:
                                    color_key = f"{file_idx}_{col}"
                                    # Use custom legend label if provided
                                    display_name = legend_labels.get(color_key, f"{file_names[file_idx]}: {col}") if use_custom_labels else f"{file_names[file_idx]}: {col}"
                                    ax.plot(df[date_col], df[col], color=colors[color_key], linewidth=2, label=display_name)
                        
                        ax.set_title(custom_title, fontsize=title_size, color=text_color)
                        ax.set_xlabel(custom_xaxis_title, fontsize=axis_label_size, color=text_color)
                        ax.set_ylabel(custom_yaxis_title, fontsize=axis_label_size, color=text_color)
                        ax.tick_params(colors=text_color, labelsize=tick_label_size)
                        
                        # Apply number formatting if requested
                        if format_numbers:
                            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_y_tick(x, pos, use_dollar)))
                        
                        if show_legend:
                            ax.legend(loc='upper left', fontsize=legend_font_size, facecolor=legend_facecolor, 
                                     edgecolor=text_color, labelcolor=text_color, bbox_to_anchor=(legend_x, legend_y))
                        
                        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                        
                        if logo_file:
                            logo_image = Image.open(logo_file)
                            newax = fig.add_axes([logo_x, logo_y, 0.1, 0.1], anchor='NE', zorder=1)
                            newax.imshow(logo_image)
                            newax.axis('off')
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            # Add subtitle with smaller font below the title
                            fig.text(0.5, 0.95, custom_subtitle, ha='center', fontsize=subtitle_size, color=text_color)
                            # Adjust the figure to make room for subtitle
                            plt.subplots_adjust(top=0.85)
                        
                        st.pyplot(fig)
                    else:  # Bar graph
                        for file_idx, cols in y_cols_by_file.items():
                            df = sub_dfs[file_idx]
                            date_col = date_cols[file_idx]
                            
                            # Create a color dictionary for this dataframe's columns
                            plot_colors = {}
                            
                            # Check if this dataframe has segmentation
                            if file_idx in segment_data:
                                segment_col = f'_segment_{segment_columns[file_idx]}'
                                
                                for col in cols:
                                    # Process each segment separately
                                    for value in segment_values[file_idx]:
                                        # Filter data for this segment
                                        segment_df = df[df[segment_col] == value]
                                        if not segment_df.empty:
                                            color_key = f"{file_idx}_{col}_{value}"
                                            plot_colors[col] = colors[color_key]
                                            # Use custom legend label if provided
                                            if use_custom_labels:
                                                legend_labels[col] = legend_labels.get(color_key, f"{file_names[file_idx]}: {col} ({value})")
                                            
                                            create_bar_graph(
                                                plot_data=segment_df,
                                                date_col=date_col,
                                                y_cols=cols,
                                                colors=plot_colors,
                                                title=f"{custom_title} - {value}",
                                                subtitle=custom_subtitle,
                                                xaxis_title=custom_xaxis_title,
                                                yaxis_title=custom_yaxis_title,
                                                theme=theme,
                                                logo_file=logo_file,
                                                logo_x=logo_x,
                                                logo_y=logo_y,
                                                show_legend=show_legend,
                                                legend_x=legend_x,
                                                legend_y=legend_y,
                                                format_numbers=format_numbers,
                                                use_dollar=use_dollar,
                                                legend_labels=legend_labels if use_custom_labels else None,
                                                title_size=title_size,
                                                subtitle_size=subtitle_size,
                                                axis_label_size=axis_label_size,
                                                tick_label_size=tick_label_size,
                                                legend_font_size=legend_font_size
                                            )
                            else:
                                # Create bar graph without segmentation
                                for col in cols:
                                    color_key = f"{file_idx}_{col}"
                                    plot_colors[col] = colors[color_key]
                                    # Use custom legend label if provided
                                    if use_custom_labels:
                                        legend_labels[col] = legend_labels.get(color_key, f"{file_names[file_idx]}: {col}")
                                    
                                    create_bar_graph(
                                        plot_data=df,
                                        date_col=date_col,
                                        y_cols=cols,
                                        colors=plot_colors,
                                        title=custom_title,
                                        subtitle=custom_subtitle,
                                        xaxis_title=custom_xaxis_title,
                                        yaxis_title=custom_yaxis_title,
                                        theme=theme,
                                        logo_file=logo_file,
                                        logo_x=logo_x,
                                        logo_y=logo_y,
                                        show_legend=show_legend,
                                        legend_x=legend_x,
                                        legend_y=legend_y,
                                        format_numbers=format_numbers,
                                        use_dollar=use_dollar,
                                        legend_labels=legend_labels if use_custom_labels else None,
                                        title_size=title_size,
                                        subtitle_size=subtitle_size,
                                        axis_label_size=axis_label_size,
                                        tick_label_size=tick_label_size,
                                        legend_font_size=legend_font_size
                                    )
                
                if st.button("Generate Animated Graph"):
                    # For animation, we need to align the dates across all dataframes
                    all_dates = set()
                    for file_idx, df in sub_dfs.items():
                        all_dates.update(df[date_cols[file_idx]].tolist())
                    
                    all_dates = sorted(all_dates)
                    
                    # Create a figure
                    fig = go.Figure()
                    frames = []
                    
                    # Create frames for animation
                    for i in range(len(all_dates)):
                        frame_data = []
                        current_date = all_dates[i]
                        
                        for file_idx, cols in y_cols_by_file.items():
                            df = sub_dfs[file_idx]
                            date_col = date_cols[file_idx]
                            
                            # Check if this dataframe has segmentation
                            if file_idx in segment_data:
                                segment_col = f'_segment_{segment_columns[file_idx]}'
                                
                                for col in cols:
                                    # Process each segment separately
                                    for value in segment_values[file_idx]:
                                        # Filter data for this segment and up to current date
                                        segment_df = df[df[segment_col] == value]
                                        mask = segment_df[date_col] <= current_date
                                        if not mask.any():
                                            continue
                                            
                                        filtered_df = segment_df[mask]
                                        if not filtered_df.empty:
                                            color_key = f"{file_idx}_{col}_{value}"
                                            # Use custom legend label if provided
                                            display_name = legend_labels.get(color_key, f"{file_names[file_idx]}: {col} ({value})") if use_custom_labels else f"{file_names[file_idx]}: {col} ({value})"
                                            
                                            frame_data.append(go.Scatter(
                                                x=filtered_df[date_col], 
                                                y=filtered_df[col], 
                                                mode='lines',
                                                name=display_name, 
                                                line=dict(color=colors[color_key]),
                                                showlegend=True
                                            ))
                                            
                                            # Add marker for the last point
                                            frame_data.append(go.Scatter(
                                                x=[filtered_df[date_col].iloc[-1]], 
                                                y=[filtered_df[col].iloc[-1]], 
                                                mode='markers',
                                                marker=dict(color=colors[color_key], size=8, symbol='circle'),
                                                showlegend=False
                                            ))
                            else:
                                # Filter data up to current date without segmentation
                                mask = df[date_col] <= current_date
                                if not mask.any():
                                    continue
                                    
                                for col in cols:
                                    color_key = f"{file_idx}_{col}"
                                    # Use custom legend label if provided
                                    display_name = legend_labels.get(color_key, f"{file_names[file_idx]}: {col}") if use_custom_labels else f"{file_names[file_idx]}: {col}"
                                    
                                    filtered_df = df[mask]
                                    if not filtered_df.empty:
                                        frame_data.append(go.Scatter(
                                            x=filtered_df[date_col], 
                                            y=filtered_df[col], 
                                            mode='lines',
                                            name=display_name, 
                                            line=dict(color=colors[color_key]),
                                            showlegend=True
                                        ))
                                        
                                        # Add marker for the last point
                                        frame_data.append(go.Scatter(
                                            x=[filtered_df[date_col].iloc[-1]], 
                                            y=[filtered_df[col].iloc[-1]], 
                                            mode='markers',
                                            marker=dict(color=colors[color_key], size=8, symbol='circle'),
                                            showlegend=False
                                        ))
                        
                        if frame_data:  # Only add frame if there's data to show
                            frames.append(go.Figure(data=frame_data))
                    
                    if frames:  # Only proceed if we have frames to animate
                        layout_settings = {
                            "plot_bgcolor": "black" if theme == "dark" else "white",
                            "paper_bgcolor": "black" if theme == "dark" else "white",
                            "font": {"color": "white" if theme == "dark" else "black"},
                            "title": {
                                "text": custom_title, 
                                "x": 0.5,
                                "font": {"size": title_size}
                            },
                            "xaxis_title": custom_xaxis_title,
                            "xaxis": {
                                "range": [min(all_dates), max(all_dates)],
                                "title": {"font": {"size": axis_label_size}},
                                "tickfont": {"size": tick_label_size}
                            },
                            "yaxis_title": custom_yaxis_title,
                            "yaxis": {
                                "title": {"font": {"size": axis_label_size}},
                                "tickfont": {"size": tick_label_size}
                            },
                            "showlegend": show_legend,
                            "legend": {
                                "x": legend_x, 
                                "y": legend_y,
                                "font": {"size": legend_font_size}
                            },
                            "margin": {"r": 150},
                        }
                        
                        if format_numbers:
                            tickprefix = '$' if use_dollar else ''
                            layout_settings["yaxis"] = {
                                "tickformat": ".2s",
                                "tickprefix": tickprefix,
                                "ticksuffix": ""
                            }
                        
                        fig.update_layout(**layout_settings)
                        
                        if logo_file:
                            logo_image = Image.open(logo_file)
                            logo_bytes = io.BytesIO()
                            logo_image.save(logo_bytes, format='PNG')
                            logo_base64 = base64.b64encode(logo_bytes.getvalue()).decode()
                            fig.add_layout_image(
                                dict(
                                    source=f'data:image/png;base64,{logo_base64}',
                                    xref="paper",
                                    yref="paper",
                                    x=logo_x,
                                    y=logo_y,
                                    sizex=0.1,
                                    sizey=0.1,
                                    xanchor="right",
                                    yanchor="bottom"
                                )
                            )
                        
                        gif_buffer = save_frames_as_gif(fig, frames, animation_speed)
                        
                        if gif_buffer:
                            gif_data = base64.b64encode(gif_buffer.read()).decode("utf-8")
                            st.markdown(f'<img src="data:image/gif;base64,{gif_data}" alt="Animated Graph">', unsafe_allow_html=True)
                            st.download_button("Download GIF", gif_buffer, file_name="animated_graph.gif", mime="image/gif")
                    else:
                        st.error("No data available to create animation frames.")
                
        # Check if 'blockchain' column exists in any of the dataframes
        for i, df in enumerate(dataframes):
            if 'blockchain' in df.columns:
                st.success(f"Found blockchain column in file {file_names[i]}")
                # Display unique blockchain names to verify they're being read correctly
                st.write("Available blockchains:")
                st.write(df['blockchain'].unique())
            else:
                st.warning(f"No 'blockchain' column found in file {file_names[i]}")
                # Show all columns so user can identify which one contains blockchain names
                st.write("Available columns:")
                st.write(df.columns.tolist())
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
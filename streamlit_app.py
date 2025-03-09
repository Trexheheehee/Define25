import streamlit as st
import pandas as pd
import os
from io import BytesIO

# Custom processing functions (from flight_processor.py)
def process_data(df):
    df = df.drop(columns=['Duplicate_ID', 'Raw_Timestamp'], errors='ignore')
    df['Route'] = df['Departure_Station'] + '-' + df['Arrival_Station']
    df['Aircraft_Config'] = df['Aircraft_Type'] + '_' + df['Seating_Capacity'].astype(str)
    df['Flight_Key'] = df['Flight_ID'] + '_' + df['Flight_Date'].dt.strftime('%Y%m%d')
    return df

def prepare_ai_data(df):
    df['Departure_Time'] = df['Departure_Time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    categorical_cols = ['Route', 'Aircraft_Config']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def optimize_data(df):
    for col in df.select_dtypes(include=['int64', 'float64']):
        df[col] = pd.to_numeric(df[col], downcast='integer' if 'int' in str(df[col].dtype) else 'float')
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].astype('category')
    return df

def clean_flight_data(df):
    """Process flight data with column removal"""
    # Remove unwanted flight columns
    keep_columns = [
        'Flight_ID', 
        'Departure_Station', 
        'Arrival_Station',
        'Scheduled_Departure',
        'Scheduled_Arrival',
        'Aircraft_Type_Req'
    ]
    df = df[keep_columns].copy()
    
    # Rest of processing remains the same
    df['Scheduled_Departure'] = pd.to_datetime(df['Scheduled_Departure'])
    df['Scheduled_Arrival'] = pd.to_datetime(df['Scheduled_Arrival'])
    df['Flight_Duration'] = (df['Scheduled_Arrival'] - df['Scheduled_Departure']).dt.total_seconds()/60
    df['Route'] = df['Departure_Station'] + '-' + df['Arrival_Station']
    
    return df[['Flight_ID', 'Route', 'Scheduled_Departure', 
              'Flight_Duration', 'Aircraft_Type_Req']]

def clean_aircraft_data(df):
    """Process aircraft data with column removal"""
    # Remove unwanted aircraft columns
    keep_columns = [
        'Aircraft_ID',
        'Aircraft_Type',
        'Current_Location',
        'Seating_Capacity',
        'Maintenance_Due',
        'Operational_Status'
    ]
    df = df[keep_columns].copy()
    
    # Rest of processing remains the same
    df['Maintenance_Due'] = pd.to_datetime(df['Maintenance_Due'])
    df['Days_Until_Maintenance'] = (df['Maintenance_Due'] - pd.Timestamp.now()).dt.days
    df['Available'] = df['Operational_Status'].isin(['Ready', 'Standby'])
    
    return df[['Aircraft_ID', 'Aircraft_Type', 'Current_Location',
              'Seating_Capacity', 'Available', 'Days_Until_Maintenance']]

def clean_dataset(df, unwanted_columns=[]):
    """Cleans dataset by removing empty columns and unwanted variables"""
    # Remove empty columns
    df = df.dropna(axis=1, how='all')
    
    # Remove specified unwanted columns
    existing_cols = [col for col in unwanted_columns if col in df.columns]
    df = df.drop(columns=existing_cols, errors='ignore')
    
    return df

def main():
    st.title("Flight and Aircraft Data Cleaner")
    
    # File upload section
    st.header("Upload Datasets")
    col1, col2 = st.columns(2)
    with col1:
        flight_file = st.file_uploader("Upload Flight Data (CSV)", type=["csv"])
    with col2:
        aircraft_file = st.file_uploader("Upload Aircraft Data (CSV)", type=["csv"])
    
    if flight_file and aircraft_file:
        try:
            # Load datasets
            flights_df = pd.read_csv(flight_file)
            aircraft_df = pd.read_csv(aircraft_file)
            
            # Define unwanted columns (customize as needed)
            unwanted_flight_cols = ['Unused1', 'Temp', 'Notes']
            unwanted_aircraft_cols = ['OldID', 'Comments']
            
            # Clean datasets
            with st.spinner("Cleaning data..."):
                clean_flights = clean_dataset(flights_df, unwanted_flight_cols)
                clean_aircraft = clean_dataset(aircraft_df, unwanted_aircraft_cols)
            
            # Show cleaned data preview
            st.subheader("Cleaned Flight Data Preview")
            st.dataframe(clean_flights.head())
            
            st.subheader("Cleaned Aircraft Data Preview")
            st.dataframe(clean_aircraft.head())
            
            # Download section
            st.header("Download Cleaned Data")
            col1, col2 = st.columns(2)
            
            with col1:
                flight_csv = clean_flights.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Flight Data",
                    data=flight_csv,
                    file_name="cleaned_flight_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                aircraft_csv = clean_aircraft.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Aircraft Data",
                    data=aircraft_csv,
                    file_name="cleaned_aircraft_data.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
    else:
        st.info("Please upload both flight and aircraft data files to begin.")

if __name__ == "__main__":
    main() 
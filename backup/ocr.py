import easyocr
import cv2
import pandas as pd
from datetime import datetime
import os

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the saved image of the number plate
img_path = "plates/scaned_img.jpg"
img = cv2.imread(img_path)

# Ensure the image loaded correctly
if img is None:
    print(f"Error: Image at path {img_path} not found.")
else:
    # Extract text from the image
    results = reader.readtext(img)
    extracted_text = " ".join([res[1] for res in results])
    print("Extracted Text:", extracted_text)

    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Load existing Excel file or create a new one if it doesn't exist
    try:
        df = pd.read_excel("db.xlsx", engine="openpyxl")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Plate Number", "Date", "Time"])

    # Append the new data
    new_entry = {"Plate Number": extracted_text, "Date": current_date, "Time": current_time}
    df = df.append(new_entry, ignore_index=True)

    # Save to Excel
    df.to_excel("db.xlsx", index=False, engine="openpyxl")
    print("Entry added to db.xlsx")

    # Delete the image file after text extraction and entry
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f"Deleted image file: {img_path}")
    else:
        print(f"File {img_path} does not exist or has already been deleted.")

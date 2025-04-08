# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 21:56:09 2025
@author: kenji
"""

import cv2
import statistics
import pytesseract
import pandas as pd
import os

def get_image_files(folder, extension=(".jpg", ".jpeg", ".png")):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extension)
    ]

def extract_text(infile):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    return pytesseract.image_to_string(infile)

def load_image(infile):
    image = cv2.imread(infile)
    if image is None:
        raise FileNotFoundError(f"Could not find file {infile}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_contours(gray_image, threshold=100):
    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask


def analyze_contours(contours,area_threshold = 2):
    areas, sizes, centers, curvatures = [], [], [], []
    for contour in contours:
        if (area := cv2.contourArea(contour)) > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            centers.append((x + w // 2, y + h // 2))
            sizes.append(w * h)
            areas.append(int(area))

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            curv_value = len(approx) / len(contour)
            curvatures.append(curv_value)

    centers.sort(key=lambda p: p[0])
    dist = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]

    return areas, sizes, centers, curvatures, dist

def calculate_statistics(data, name):
    return {
        f"{name}_mean": round(statistics.mean(data),2) if data else 0,
        f"{name}_median": round(statistics.median(data),2) if data else 0,
        f"{name}_stdev": round(statistics.stdev(data),2) if len(data) > 1 else 0
    }

def calculate_readibility(input_text, extracted_text):
    extracted_text = extracted_text.replace(" ","")
    input_text = input_text.replace(" ","")
    n = len(extracted_text)-1
    dp = [0]*(n+1)
    dp[n] = 0
    seq_length = 1

    for i in range(n-1,-1,-1):
        sequence = extracted_text[i:i+seq_length+1]
        if sequence in input_text: 
            dp[i] = 1
            dp[i-1] = 1
            seq_length+=1
        else:
            dp[i] = 0
            seq_length = 0
        
    total_matches = sum(dp)
    readibility = round(total_matches/len(input_text),4)
    
    return readibility

def analyze_text(infile, input_text):
    text = extract_text(infile)
    readibility = calculate_readibility(input_text, text)
    gray_image = load_image(infile)
    contours, _ = find_contours(gray_image)
    areas, sizes, centers, curvatures, dist = analyze_contours(contours)

    features = {
        "filename": os.path.basename(infile),
        "continous_lines": len(areas),
        "text": text.strip().replace('\n', ' ').replace('=',''),
        "area_size_ratio": round(statistics.mean(areas) / statistics.mean(sizes), 2) if sizes else 0,
        "readibility": readibility
    }

    features.update(calculate_statistics(areas, "area"))
    features.update(calculate_statistics(sizes, "size"))
    features.update(calculate_statistics(dist, "spacing"))
    features.update(calculate_statistics(curvatures, "curvature"))
    
    return features

def export_csv(features_list, out_file="handwrite_analysis_results.csv"):
    df = pd.DataFrame(features_list)
    df.to_csv(out_file, index=False)

def main():
    input_text = "i think hamburgers and french fries work well together"
    input_folder = "handwriting_samples_30"
    input_files = get_image_files(input_folder)

    all_features = []
    for infile in input_files:
        try:
            features = analyze_text(infile, input_text)
            all_features.append(features)
            print(f"Processed: {infile}")
        except Exception as e:
            print(f"Error with {infile}: {e}")

    export_csv(all_features)
    print("Done! CSV saved.")

if __name__ == "__main__":
    main()

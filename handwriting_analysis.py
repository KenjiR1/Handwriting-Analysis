# -*- coding: utf-8 -*-
# Main code for collecting dependent variables from handwritten text. handwritten text files should be
# jpg or png, and be stored within a folder. The export will be a .csv file

import cv2
import statistics
import pytesseract
import pandas as pd
import os

def get_image_files(folder, extension=(".jpg", ".png")):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extension)
    ]

def extract_text(infile):
    pytesseract.pytesseract.tesseract_cmd = 'TESSERACT.EXE PATH' # Update with tesseract.exe path
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

    centers.sort(key=lambda c: c[0])
    dist = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)] # subtract each center x-cord by the previous center x-cord

    return areas, sizes, centers, curvatures, dist

def calculate_statistics(data, name):
    return {
        f"{name}_mean": round(statistics.mean(data),2) if data else 0,
        f"{name}_median": round(statistics.median(data),2) if data else 0,
        f"{name}_stdev": round(statistics.stdev(data),2) if len(data) > 1 else 0
    }

def calculate_readability(input_text, extracted_text):
    s_extract = extracted_text.replace(" ","")
    s_true = input_text.replace(" ","")
    
    m = len(s_extract) # Row
    n = len(s_true) # Column
    
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            cost = 0 if s_extract[i - 1] == s_true[j - 1] else 1
            dp[i][j] = min(
                cost + dp[i-1][j-1], # Sub
                1 + dp[i][j-1], # Insert
                1 + dp[i-1][j], # Delete
                )
    
    cost = dp[m][n]
    readability = max(1 - cost / n,0)
    
    return readability

def analyze_text(infile, input_text):
    text = extract_text(infile)
    readibility = calculate_readability(input_text, text)
    gray_image = load_image(infile)
    contours, _ = find_contours(gray_image)
    areas, sizes, centers, curvatures, dist = analyze_contours(contours)

    dependent_vars = {
        "filename": os.path.basename(infile),
        "continous_lines": len(areas),
        "text": text.strip().replace('\n', ' ').replace('=',''),
        "area_size_ratio": round(statistics.mean(areas) / statistics.mean(sizes), 2) if sizes else 0,
        "readability": readibility
    }

    dependent_vars.update(calculate_statistics(areas, "area"))
    dependent_vars.update(calculate_statistics(sizes, "size"))
    dependent_vars.update(calculate_statistics(dist, "spacing"))
    dependent_vars.update(calculate_statistics(curvatures, "curvature"))
    
    return dependent_vars

def export_csv(features_list, out_file="total_results.csv"):
    df = pd.DataFrame(features_list)
    df.to_csv(out_file, index=False)

def main():
    input_text = "" # Replace with actual sentence
    input_folder = "" # Replace with folder
    input_files = get_image_files(input_folder)

    all_data = []
    for infile in input_files:
        try:
            dependent_vars = analyze_text(infile, input_text)
            all_data.append(dependent_vars)
            print(f"Processed: {infile}")
        except Exception as e:
            print(f"Error with {infile}: {e}")

    export_csv(all_data)
    print("Finished! CSV results saved.")

if __name__ == "__main__":
    main()

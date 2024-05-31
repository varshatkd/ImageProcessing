# This code finds the accuracy of extraction after adding different compression percentages to the watermarked image



import Levenshtein
import string
import cv2
import numpy as np
import pandas as pd
import pywt
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to generate a random URL
def remove_spaces_before_slashes(text):
    return re.sub(r'\s+(?=//)', '', text)

# Function to convert text to image
def text_to_image(text, image_size=(256, 256), font_size=20):
    image = Image.new("RGB", image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)  # Adjust font file path as needed

    # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate text width and height
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(image)
compr_percentage=[10,20,30]
# List of URLs for certificates
urls = [
    "https://YekNRSVjQn.com",
    "https://3KWigYJwLe.com",
    "https://rigf4dk87n.com",
    "https://EXGDygjtiz.com",
    "https://1VsTAYt3Hk.com",
    "https://Bb20RXOU4K.com",
    "https://Tk1GdCba38.com",
    "https://HSfXHmHZ9X.com",
    "https://k6wnDThRwU.com",
    "https://qx2SwCdWUr.com",
    "https://QCyeowBpXM.com",
    "https://eqZiyYq11d.com",
    "https://xaXfORt9BZ.com",
    "https://QP8PppHiyO.com",
    "https://MSgaNxd5cc.com",
    "https://3auL4SvXur.com",
    "https://c3vFjwiYvH.com",
    "https://2aBnQweJrD.com",
    "https://meOt8FBHCI.com",
    "https://apxOr9nXMj.com",
    "https://9q1TtWRoDX.com",
    "https://TMofGyAst4.com",
    "https://A3KFMURphJ.com",
    "https://YoxEfYLXYR.com",
    "https://POHjGbk9eJ.com",
    "https://KNke2vAZRZ.com",
    "https://Ps1srUkAyt.com",
    "https://X2s8Cina1m.com",
    "https://b2E13fAHry.com",
    "https://K9RvpAqkNQ.com",
    "https://oMAr4GQz0p.com",
    "https://qZ5FvaqHyR.com",
    "https://Z6M3vzWkAo.com",
    "https://oHrjNUqCVY.com",
    "https://EZHSJ84Pr9.com",
    "https://cL8OBpggIr.com",
    "https://6hX1Z9PvLW.com",
    "https://SGNddw4BOY.com",
    "https://CJwuKdgjCy.com",
    "https://S9tSuLujgP.com",
    "https://RA8A5x1L23.com",
    "https://Tr5Ymc3Z29.com",
    "https://GBzBy5nJmu.com",
    "https://absTeTWM0z.com",
    "https://gkYo3g7rmY.com",
    "https://ya4rOStqAR.com",
    "https://zmerKdlpgB.com",
    "https://p1Ye2AYDHu.com",
    "https://YYVcRTKLPQ.com",
    "https://ss3vBnHGzR.com",
    "https://D68KbRRPpm.com",
    "https://6KkV9LsxK8.com",
]

avg_acc = 0
result=[]
# Main loop to process multiple images
for i, url in enumerate(urls, start=1):
    watermark_text = url

    # Convert watermark text to image
    watermark_image = text_to_image(watermark_text, font_size=20)

    #cv2.imshow("Original Watermark", watermark_image)
    cv2.imwrite(f"ori_watermark_{i}.png", watermark_image)

    # Split the watermark image into its color channels
    watermark_b, watermark_g, watermark_r = cv2.split(watermark_image)

    # Apply DCT to each color channel
    watermark_dct_b = cv2.dct(np.float32(watermark_b))
    watermark_dct_g = cv2.dct(np.float32(watermark_g))
    watermark_dct_r = cv2.dct(np.float32(watermark_r))

    # Combine the DCT results into a single image
    watermark_dct = cv2.merge((watermark_dct_b, watermark_dct_g, watermark_dct_r))

    rows, cols, _ = watermark_dct.shape
    
    # Modify the DCT coefficients in the high-frequency subband
    for k in range(rows):
        for l in range(cols - k, cols):
            watermark_dct[k, l] = 0

    # Read the host image
    host_path = f"{i}.png"
    host_image = cv2.imread(host_path)
    host_image = cv2.resize(host_image, (1024, 1024))
    
    # Apply DWT to the host image
    wavelet = 'haar'
    coeffs = pywt.dwt2(host_image, wavelet)
    LL, (LH, HL, HH) = coeffs
    LL2, (LH2, HL2, HH2) = pywt.dwt2(HH, wavelet)

    # Embed the watermark into the HH subband of the host image
    for k in range(rows):
        for l in range(0, cols - k):
            HH2[k, l] = watermark_dct[k, l, 0]  # Apply to the first channel (blue channel)
            HH2[k, l] = watermark_dct[k, l, 1]  # Apply to the second channel (green channel)
            HH2[k, l] = watermark_dct[k, l, 2]  # Apply to the third channel (red channel)

    # Combine the modified DWT coefficients
    coeffs_2 = (LL2, (LH2, HL2, HH2))

    # Reconstruct the watermarked image
    watermarked_hh = pywt.idwt2(coeffs_2, wavelet)
    watermarked_image = pywt.idwt2((LL, (LH, HL, watermarked_hh)), wavelet)
   
    #cv2.imshow("Watermarked Image", cv2.resize(watermarked_image, (512, 512)))
    cv2.imwrite(f"watermarked_image_color{host_path}", cv2.resize(watermarked_image, (512, 512)))
    
    for compression_quality in compr_percentage:
        # compression_quality = 70  # Adjust quality as needed
        compressed_image_path = f"compressed_watermarked_image_{i}.png"
        cv2.imwrite(compressed_image_path, watermarked_image , [cv2.IMWRITE_PNG_COMPRESSION, compression_quality])



        coeffs_w = pywt.dwt2(watermarked_image, wavelet)
        LL_w, (LH_w, HL_w, HH_w) = coeffs_w

        LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(HH_w, wavelet)

        row_hh, cols_hh, _ = HH2_w.shape

        # Loop to zero out coefficients in the DWT of the watermarked image
        for k in range(row_hh):
            for l in range(cols_hh - k, cols_hh):
                if l >= 0:  # Check if the index is valid
                    HH2_w[k, l] = 0

        # Extract the watermark from the HH subband
        extr = cv2.idct(HH2_w)
        extr = cv2.resize(extr, (256, 256))
        cv2.imwrite(f"extracted_watermark_{host_path}", extr)
        image_path = f"extracted_watermark_{host_path}"

        extracted_text_org = pytesseract.image_to_string(image_path, lang='eng', config='--psm 11')

        print(f"Original Text from image {i}:", watermark_text)
        print(f"Extracted Text from image {i}:", extracted_text_org)
        extracted_text = remove_spaces_before_slashes(extracted_text_org.strip())
        distance = Levenshtein.distance(watermark_text, extracted_text)
        
        accuracy = 1.0 if distance == 0 else 0.0
        avg_acc += accuracy
        result.append({
                'Image': i,
                'Compression Quality': compression_quality,
                'Original Text': watermark_text,
                'Extracted Text': extracted_text,
                'Accuracy': accuracy
            })

        # Calculate average accuracy
avg_acc = avg_acc / (len(urls) * len(compr_percentage))
print("Average Accuracy:", avg_acc)


cv2.waitKey(0)
cv2.destroyAllWindows()

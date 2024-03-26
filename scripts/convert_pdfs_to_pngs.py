# Copyright Anon 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import fitz  # PyMuPDF

# Directory path for the PDFs
pdf_dir = "./imgs/results/"

# Find all PDF files in the directory
pdf_files = glob.glob(pdf_dir + "*.pdf")

# Convert each PDF to a PNG with higher quality
for pdf_file in pdf_files:
    try:
        # Open the PDF file
        doc = fitz.open(pdf_file)

        # Conversion settings for higher quality
        zoom_x = 2.0  # horizontal zoom
        zoom_y = 2.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

        # Convert each page to PNG
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)  # use zoom factor during conversion
            output_file = pdf_file.replace(".pdf", f"_{page_num}.png")
            pix.save(output_file)

        # Close the document
        doc.close()

        # Delete the PDF file
        os.remove(pdf_file)

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

# List PNG files to confirm
png_files = glob.glob(pdf_dir + "*.png")
print(png_files)

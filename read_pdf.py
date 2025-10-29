#!/usr/bin/env python3
"""
Script to read and extract text from PDF file
"""
import sys
from pypdf import PdfReader

def read_pdf(pdf_path):
    """Read PDF and extract text from all pages"""
    try:
        reader = PdfReader(pdf_path)

        print(f"PDF Information:")
        print(f"  Number of pages: {len(reader.pages)}")

        if reader.metadata:
            print(f"\nMetadata:")
            for key, value in reader.metadata.items():
                print(f"  {key}: {value}")

        print(f"\n{'='*80}")
        print("CONTENT:")
        print('='*80)

        # Extract text from each page
        full_text = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            print(f"\n--- Page {i} ---")
            print(text)
            full_text.append(text)

        return full_text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    pdf_path = "AVAILABLE_GENERIC_NAMES_FOR_TRILOBITES.pdf"
    text_content = read_pdf(pdf_path)

    print(f"\n{'='*80}")
    print(f"Summary: Successfully extracted text from {len(text_content)} pages")
    print('='*80)

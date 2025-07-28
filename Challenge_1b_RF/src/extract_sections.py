import os
import pdfplumber

def extract_sections_from_pdf(pdf_path):
    """
    Extracts section titles, text, and page numbers from a PDF.
    Returns a list of dicts: {section_title, text, page_number}
    """
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Simple heuristic: lines in larger/bold font are section titles
            lines = text.split('\n')
            for line in lines:
                if len(line.strip()) > 0:
                    sections.append({
                        'section_title': line.strip(),
                        'text': text,
                        'page_number': page_num
                    })
    return sections

def extract_all_sections(pdf_dir):
    """
    Extracts sections from all PDFs in a directory.
    Returns a dict: {pdf_filename: [sections]}
    """
    pdf_sections = {}
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, fname)
            pdf_sections[fname] = extract_sections_from_pdf(pdf_path)
    return pdf_sections

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_sections.py <PDF_DIR>")
    else:
        pdf_dir = sys.argv[1]
        sections = extract_all_sections(pdf_dir)
        print(sections) 
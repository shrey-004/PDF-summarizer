import pdfplumber
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------- Step 1: Extract text from PDF --------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -------- Step 2: Clean text --------
def clean_text(text):
    # Remove extra spaces & line breaks
    text = ' '.join(text.split())
    return text

# -------- Step 3: Chunk text by sentences --------
def chunk_text(text, max_tokens=800):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= max_tokens:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# -------- Step 4: Load a long-document summarization model --------
model_name = "derekiya/bart_fine_tuned_model-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)  # device=-1 means CPU

# -------- Step 5: Summarize --------
def summarize_text(text, max_tokens=800, max_length=200, min_length=50):
    chunks = chunk_text(text, max_tokens)
    chunk_summaries = []

    # First pass: Summarize each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        chunk_summaries.append(summary)

    # Second pass: Summarize all summaries into one
    combined_text = " ".join(chunk_summaries)
    final_summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return final_summary

# -------- Step 6: Run --------
if __name__ == "__main__":
    pdf_path = "/home/shrey/PDF-summarizer/data/pdffs/CV_DSE.pdf"  # Change path if needed
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    final_summary = summarize_text(cleaned_text)
    print("\nFinal Summary:\n", final_summary)

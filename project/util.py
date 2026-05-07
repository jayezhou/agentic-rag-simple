import os
import config
import pymupdf
import pymupdf4llm
from pathlib import Path
import glob
import base64
from openai import OpenAI
import mammoth

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VLM_SYSTEM_PROMPT = """You are an expert document parser specializing in converting PDF pages to markdown format.

**Your task:**
Extract ALL content from the provided page image and return it as clean, well-structured markdown.

**Text Extraction Rules:**
1. Preserve the EXACT text as written (including typos, formatting, special characters)
2. Maintain the logical reading order (top-to-bottom, left-to-right)
3. Preserve hierarchical structure using appropriate markdown headers (#, ##, ###)
4. Keep paragraph breaks and line spacing as they appear
5. Use markdown lists (-, *, 1.) for bullet points and numbered lists
6. Preserve text emphasis: **bold**, *italic*, `code`
7. For multi-column layouts, extract left column first, then right column

**Tables:**
- Convert all tables to markdown table format
- Preserve column alignment and structure
- Use | for columns and - for headers

**Mathematical Formulas:**
- Convert to LaTeX format: inline `$formula$`, display `$$formula$$`
- If LaTeX conversion is uncertain, describe the formula clearly

**Images, Diagrams, Charts:**
- Insert markdown image placeholder: `![Description](image)`
- Provide a detailed, informative description including:
  * Type of visual (photo, diagram, chart, graph, illustration)
  * Main subject or purpose
  * Key elements, labels, or data points
  * Colors, patterns, or notable visual features
  * Context or relationship to surrounding text
- For charts/graphs: mention axes, data trends, and key values
- For diagrams: describe components and their relationships

**Special Elements:**
- Footnotes: Use markdown footnote syntax `[^1]`
- Citations: Preserve as written
- Code blocks: Use triple backticks with language specification
- Quotes: Use `>` for blockquotes
- Links: Preserve as `[text](url)` if visible

**Quality Guidelines:**
- DO NOT add explanations, comments, or meta-information
- DO NOT skip or summarize content
- DO NOT invent or hallucinate text not present in the image
- DO NOT include "Here is the markdown..." or similar preambles
- Output ONLY the markdown content, nothing else

**Output Format:**
Return raw markdown with no wrapper, no code blocks, no explanations.
Start immediately with the page content.
""".strip()


# ============================================================================
# Category 1: Simple PDFs - Fast Text Extraction (PyMuPDF4LLM)
# ============================================================================

def pdf_to_markdown_simple(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
    print(f"  ✓ Converted using PyMuPDF4LLM (Category 1): {Path(pdf_path).name}")


# ============================================================================
# Category 2: Medium Complexity PDFs - OCR + Structure Recognition (Docling)
# ============================================================================

def pdf_to_markdown_medium(pdf_path, output_dir):
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()

        output_path = Path(output_dir) / Path(pdf_path).stem
        output_path.with_suffix(".md").write_text(markdown_content, encoding='utf-8')
        print(f"  ✓ Converted using Docling (Category 2): {Path(pdf_path).name}")

    except ImportError:
        print(f"  ⚠️  Docling not installed. Install with: pip install docling")
        print(f"  → Falling back to PyMuPDF4LLM for: {Path(pdf_path).name}")
        pdf_to_markdown_simple(pdf_path, output_dir)
    except Exception as e:
        print(f"  ✗ Docling error for {Path(pdf_path).name}: {e}")
        print(f"  → Falling back to PyMuPDF4LLM")
        pdf_to_markdown_simple(pdf_path, output_dir)


# ============================================================================
# Category 3: Complex PDFs - Vision-Language Models (VLMs)
# ============================================================================

def pdf_to_markdown_complex(pdf_path, output_dir, base_url=None, api_key=None):
    if base_url is None:
        base_url = os.getenv("DASHSCOPE_BASE_URL")
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    if not base_url or not api_key:
        print(f"  ⚠️  DashScope credentials not found. Set DASHSCOPE_BASE_URL and DASHSCOPE_API_KEY")
        print(f"  → Falling back to PyMuPDF4LLM for: {Path(pdf_path).name}")
        pdf_to_markdown_simple(pdf_path, output_dir)
        return

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        pdf_document = pymupdf.open(pdf_path)
        markdown_pages = {}

        print(f"  📄 Processing with VLM (Category 3): {Path(pdf_path).name} ({pdf_document.page_count} pages)...")

        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]

                pix = page.get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))
                img_data = pix.tobytes("png")

                img_base64 = base64.b64encode(img_data).decode('utf-8')

                completion = client.chat.completions.create(
                    model="qwen3-vl-30b-a3b-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": VLM_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Convert this PDF page to clean, structured markdown. Extract all text, describe images, and preserve the layout."
                                },
                            ],
                        },
                    ],
                )

                markdown_pages[page_num + 1] = completion.choices[0].message.content
                print(f"    ✓ Page {page_num + 1}/{pdf_document.page_count}")

            except Exception as e:
                print(f"    ✗ Error on page {page_num + 1}: {e}")
                markdown_pages[page_num + 1] = f"<!-- Error processing page: {e} -->"

        pdf_document.close()

        combined_markdown = "\n\n---\n\n".join([
            f"# Page {page_num}\n\n{content}"
            for page_num, content in markdown_pages.items()
        ])

        output_path = Path(output_dir) / Path(pdf_path).stem
        output_path.with_suffix(".md").write_text(combined_markdown, encoding='utf-8')
        print(f"  ✓ Saved: {output_path.with_suffix('.md')}")

    except Exception as e:
        print(f"  ✗ VLM error for {Path(pdf_path).name}: {e}")
        print(f"  → Falling back to PyMuPDF4LLM")
        pdf_to_markdown_simple(pdf_path, output_dir)


# ============================================================================
# Intelligent PDF Analysis and Routing
# ============================================================================

def analyze_pdf_complexity(pdf_path):
    doc = pymupdf.open(str(pdf_path))

    if doc.page_count == 0:
        doc.close()
        return 1, "Empty document, using simple method"

    page_indices = [0]
    if doc.page_count > 1:
        page_indices.append(doc.page_count - 1)
    if doc.page_count > 3:
        page_indices.append(doc.page_count // 2)

    total_text_length = 0
    total_image_count = 0
    pages_sampled = 0

    for page_idx in page_indices:
        try:
            page = doc[page_idx]
            text = page.get_text()
            total_text_length += len(text.strip())
            total_image_count += len(page.get_images())
            pages_sampled += 1
        except Exception as e:
            print(f"  ⚠️ Error sampling page {page_idx}: {e}")
            continue

    doc.close()

    if pages_sampled == 0:
        return 1, "Error sampling pages, using simple method"

    avg_text_length = total_text_length / pages_sampled
    avg_image_count = total_image_count / pages_sampled

    # Category 3: Image-heavy documents
    if avg_image_count > 3 or (avg_image_count > 2 and avg_text_length < 100):
        return 3, f"Image-heavy document (avg_images={avg_image_count:.1f}, avg_text={avg_text_length:.0f})"
    # Category 2: Scanned documents without heavy images
    elif avg_text_length < 100:
        return 2, f"Scanned document detected (avg_text={avg_text_length:.0f}, avg_images={avg_image_count:.1f})"
    # Category 1: Simple digital PDF
    else:
        return 1, f"Simple digital PDF (avg_images={avg_image_count:.1f}, avg_text={avg_text_length:.0f})"


def pdf_to_markdown(pdf_path, output_dir, method="auto"):
    pdf_path = Path(pdf_path)

    if method == "auto":
        category, reason = analyze_pdf_complexity(pdf_path)
        print(f"  → Auto-detected Category {category}: {reason}")

        if category == 3:
            method = "complex"
        elif category == 2:
            method = "medium"
        else:
            method = "simple"

    if method == "complex":
        pdf_to_markdown_complex(pdf_path, output_dir)
    elif method == "medium":
        pdf_to_markdown_medium(pdf_path, output_dir)
    else:
        pdf_to_markdown_simple(pdf_path, output_dir)


def pdfs_to_markdowns(path_pattern, overwrite: bool = False, method="auto"):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            print(f"\n📄 Converting: {pdf_path.name}")
            pdf_to_markdown(pdf_path, output_dir, method=method)
        else:
            print(f"⊙ Skipped (already exists): {pdf_path.name}")


def docx_to_markdown(docx_path, output_dir):
    """Convert DOCX to Markdown using mammoth."""
    docx_path = Path(docx_path)
    output_path = Path(output_dir) / docx_path.stem
    md_path = output_path.with_suffix(".md")

    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_markdown(f)
    md_path.write_text(result.value, encoding="utf-8")
    print(f"  ✓ Converted using mammoth: {docx_path.name}")
    if result.messages:
        for msg in result.messages:
            print(f"    ⚠ {msg}")

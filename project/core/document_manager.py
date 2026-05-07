import time
from pathlib import Path
import shutil
import config
from util import pdfs_to_markdowns, docx_to_markdown


class DocumentManager:

    def __init__(self, rag_system, pdf_conversion_method="auto"):
        """
        Args:
            rag_system: RAG system instance
            pdf_conversion_method: "auto", "simple", "medium", "complex"
        """
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_conversion_method = pdf_conversion_method

    def add_documents(self, document_paths, progress_callback=None):
        if not document_paths:
            return 0, 0

        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md", ".docx"]]

        if not document_paths:
            return 0, 0

        added = 0
        skipped = 0

        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")

            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"

            try:
                suffix = Path(doc_path).suffix.lower()
                # Only convert if markdown doesn't exist yet
                if not md_path.exists():
                    if suffix == ".md":
                        shutil.copy(doc_path, md_path)
                    elif suffix == ".docx":
                        docx_to_markdown(str(doc_path), self.markdown_dir)
                    else:
                        pdfs_to_markdowns(str(doc_path), overwrite=False, method=self.pdf_conversion_method)
                else:
                    print(f"  ⊙ Markdown already exists, reusing: {md_path.name}")

                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)

                if not child_chunks:
                    skipped += 1
                    continue

                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)

                # Retry embedding+insert with backoff (transient API errors are common)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        collection.add_documents(child_chunks)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        wait = 2 ** attempt
                        print(f"  ⚠ Embedding API error (attempt {attempt+1}/{max_retries}), "
                              f"retrying in {wait}s: {e}")
                        time.sleep(wait)

                self.rag_system.parent_store.save_many(parent_chunks)

                added += 1

            except Exception as e:
                print(f"\n{'!'*60}")
                print(f"ERROR processing {doc_path}: {e}")
                print(f"Markdown file may be cleaned up to allow re-ingestion.")
                print(f"{'!'*60}\n")
                # Clean up markdown so next upload can retry from scratch
                if md_path.exists():
                    md_path.unlink()
                    print(f"  → Removed incomplete markdown: {md_path.name}")
                skipped += 1

        return added, skipped

    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])

    def clear_all(self):
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
            self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.clear_collection(self.rag_system.collection_name)

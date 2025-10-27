"""
Text preprocessing pipeline for cleaning training data
Handles HTML tags, special tokens, whitespace, and other artifacts
"""

import re
import unicodedata
from pathlib import Path


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for cleaning training data
    """

    def __init__(
        self,
        remove_html_tags=True,
        remove_urls=True,
        normalize_whitespace=True,
        normalize_unicode=True,
        remove_special_tokens=True,
        remove_control_chars=True,
        remove_duplicate_lines=True,
        min_line_length=10,
        max_line_length=10000,
    ):
        """
        Args:
            remove_html_tags: Remove HTML/XML tags like <div>, </n>, etc.
            remove_urls: Remove URLs (http/https)
            normalize_whitespace: Normalize spaces, tabs, newlines
            normalize_unicode: Normalize unicode characters (NFKC)
            remove_special_tokens: Remove common special tokens like [UNK], <pad>, etc.
            remove_control_chars: Remove control characters
            remove_duplicate_lines: Remove duplicate consecutive lines
            min_line_length: Minimum line length to keep
            max_line_length: Maximum line length to keep
        """
        self.remove_html_tags = remove_html_tags
        self.remove_urls = remove_urls
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_tokens = remove_special_tokens
        self.remove_control_chars = remove_control_chars
        self.remove_duplicate_lines = remove_duplicate_lines
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length

    def clean_html_tags(self, text):
        """Remove HTML/XML tags"""
        # Remove common HTML/XML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove HTML entities
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        return text

    def clean_urls(self, text):
        """Remove URLs"""
        # Remove http/https URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove www URLs
        text = re.sub(r"www\.\S+", "", text)
        return text

    def clean_special_tokens(self, text):
        """Remove common special tokens from models/datasets"""
        special_tokens = [
            r"\[UNK\]",
            r"\[PAD\]",
            r"\[CLS\]",
            r"\[SEP\]",
            r"\[MASK\]",
            r"<unk>",
            r"<pad>",
            r"<s>",
            r"</s>",
            r"<eos>",
            r"<bos>",
            r"<\|endoftext\|>",
        ]
        for token in special_tokens:
            text = re.sub(token, "", text, flags=re.IGNORECASE)
        return text

    def clean_control_chars(self, text):
        """Remove control characters except newlines and tabs"""
        # Keep newline and tab, remove other control chars
        text = "".join(char for char in text if char in "\n\t" or not unicodedata.category(char).startswith("C"))
        return text

    def clean_unicode(self, text):
        """Normalize unicode characters"""
        # NFKC normalization (compatibility decomposition followed by canonical composition)
        text = unicodedata.normalize("NFKC", text)
        return text

    def clean_whitespace(self, text):
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Replace multiple tabs with single space
        text = re.sub(r"\t+", " ", text)
        # Replace multiple newlines with double newline (paragraph separation)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Remove trailing/leading whitespace on each line
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text

    def remove_duplicates(self, text):
        """Remove duplicate consecutive lines"""
        lines = text.split("\n")
        cleaned_lines = []
        prev_line = None

        for line in lines:
            if line != prev_line or line.strip() == "":
                cleaned_lines.append(line)
            prev_line = line

        return "\n".join(cleaned_lines)

    def filter_lines(self, text):
        """Filter lines by length"""
        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            line_len = len(line.strip())
            # Keep empty lines for paragraph separation
            if line_len == 0:
                filtered_lines.append(line)
            # Keep lines within length bounds
            elif self.min_line_length <= line_len <= self.max_line_length:
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def clean_text(self, text):
        """
        Apply all cleaning steps to text

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        original_length = len(text)

        # Apply cleaning steps in order
        if self.remove_html_tags:
            text = self.clean_html_tags(text)

        if self.remove_urls:
            text = self.clean_urls(text)

        if self.remove_special_tokens:
            text = self.clean_special_tokens(text)

        if self.normalize_unicode:
            text = self.clean_unicode(text)

        if self.remove_control_chars:
            text = self.clean_control_chars(text)

        if self.normalize_whitespace:
            text = self.clean_whitespace(text)

        if self.remove_duplicate_lines:
            text = self.remove_duplicates(text)

        # Filter lines by length
        text = self.filter_lines(text)

        # Final whitespace cleanup
        text = text.strip()

        cleaned_length = len(text)
        reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0

        print(f"Cleaning stats: {original_length:,} -> {cleaned_length:,} chars ({reduction_pct:.1f}% reduction)")

        return text

    def process_file(self, input_path, output_path):
        """
        Process a text file with cleaning pipeline

        Args:
            input_path: Path to input text file
            output_path: Path to save cleaned text
        """
        print(f"Processing: {input_path}")

        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Original size: {len(text):,} characters")

        # Clean text
        cleaned_text = self.clean_text(text)

        # Write output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"Cleaned text saved to: {output_path}")
        print(f"Final size: {len(cleaned_text):,} characters")
        print()


def preprocess_dataset(input_dir, output_dir, **preprocessor_kwargs):
    """
    Preprocess all text files in a directory

    Args:
        input_dir: Directory containing input text files
        output_dir: Directory to save cleaned files
        **preprocessor_kwargs: Arguments passed to TextPreprocessor
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize preprocessor
    preprocessor = TextPreprocessor(**preprocessor_kwargs)

    # Find all text files
    text_files = list(input_dir.glob("*.txt"))

    if not text_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(text_files)} text file(s) to process\n")

    # Process each file
    for input_file in text_files:
        output_file = output_dir / f"cleaned_{input_file.name}"
        preprocessor.process_file(input_file, output_file)

    print("All files processed successfully!")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean text data for training")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file or directory")
    parser.add_argument("--min-line-length", type=int, default=10, help="Minimum line length")
    parser.add_argument("--max-line-length", type=int, default=10000, help="Maximum line length")
    parser.add_argument("--no-remove-html", action="store_true", help="Don't remove HTML tags")
    parser.add_argument("--no-remove-urls", action="store_true", help="Don't remove URLs")
    parser.add_argument("--no-normalize-whitespace", action="store_true", help="Don't normalize whitespace")
    parser.add_argument("--no-normalize-unicode", action="store_true", help="Don't normalize unicode")
    parser.add_argument("--no-remove-special-tokens", action="store_true", help="Don't remove special tokens")
    parser.add_argument("--no-remove-control-chars", action="store_true", help="Don't remove control characters")
    parser.add_argument("--no-remove-duplicates", action="store_true", help="Don't remove duplicate lines")

    args = parser.parse_args()

    # Build preprocessor arguments
    preprocessor_kwargs = {
        "remove_html_tags": not args.no_remove_html,
        "remove_urls": not args.no_remove_urls,
        "normalize_whitespace": not args.no_normalize_whitespace,
        "normalize_unicode": not args.no_normalize_unicode,
        "remove_special_tokens": not args.no_remove_special_tokens,
        "remove_control_chars": not args.no_remove_control_chars,
        "remove_duplicate_lines": not args.no_remove_duplicates,
        "min_line_length": args.min_line_length,
        "max_line_length": args.max_line_length,
    }

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Process single file
        preprocessor = TextPreprocessor(**preprocessor_kwargs)
        preprocessor.process_file(input_path, output_path)
    elif input_path.is_dir():
        # Process directory
        preprocess_dataset(input_path, output_path, **preprocessor_kwargs)
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()

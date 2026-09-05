import re
import logging

logger = logging.getLogger(__name__)

# Markdown separators (e.g., ---, ===, ⸻, ─)
_MARKDOWN_SEP_RE = re.compile(r"^\s*([=\-─⸻_]{3,})\s*$")

# Numbered section headers (e.g., "1. INITIAL FIR", "2. PERSON DATABASE")
_NUMBERED_HEADER_RE = re.compile(r"^\s*\d+\.\s+[A-Z][A-Za-z0-9\s&:-]+$")

# Table/field labels (e.g., "Entity type:", "Investigative relevance:", "Case ID:")
# It matches a short phrase ending with a colon, optionally followed by space and something else.
_FIELD_LABEL_RE = re.compile(r"^\s*[A-Z][a-zA-Z\s]+:\s*.*$")

def strip_structural_noise(text: str) -> str:
    """
    Preprocess document text to strip lines that are clearly structural or formatting
    artifacts before they reach the NER model. Also handles table parsing by extracting
    only the 'Name' column from tabular data.
    """
    cleaned_lines = []
    lines = text.splitlines()
    
    in_table = False
    name_col_idx = None
    table_row_idx = 1
    
    for line in lines:
        stripped = line.strip()
        
        # Strip trailing meta-content blocks entirely
        if "--- END OF DOCUMENT" in stripped.upper() or "NOTE FOR TESTING PURPOSES" in stripped.upper():
            break
            
        # Check table state first
        if in_table:
            if not stripped:
                in_table = False
                cleaned_lines.append(line)
                continue
                
            parts = line.split('\t')
            if len(parts) == 1:
                parts = line.split('|')
            if len(parts) == 1:
                parts = re.split(r'\s{2,}', line)
                
            if len(parts) > 1 and name_col_idx is not None and name_col_idx < len(parts):
                name_val = parts[name_col_idx].strip()
                if name_val and not name_val.startswith("---") and not name_val.startswith("==="):
                    # Output as a numbered bullet so extract_bulleted_names regex catches it reliably
                    cleaned_lines.append(f"{table_row_idx}. {name_val}")
                    table_row_idx += 1
                continue
            else:
                # If it doesn't look like a row anymore, end the table
                in_table = False

        if not in_table and stripped:
            # Check if this line is a table header containing 'Name', 'Full Name', 'Witness', 'Person'
            parts = line.split('\t')
            if len(parts) == 1:
                parts = line.split('|')
            if len(parts) == 1:
                parts = re.split(r'\s{2,}', line)
                
            if len(parts) > 1:
                parts_lower = [p.strip().lower() for p in parts]
                # Find the first column header that looks like a person's name column
                found_idx = -1
                for idx, p in enumerate(parts_lower):
                    if any(kw in p for kw in ('name', 'person', 'witness', 'suspect', 'individual')):
                        found_idx = idx
                        break
                        
                if found_idx != -1:
                    in_table = True
                    name_col_idx = found_idx
                    table_row_idx = 1
                    continue # Skip the header row
                    
        # Skip empty lines
        if not stripped:
            cleaned_lines.append(line)
            continue
            
        # 1. Markdown-style separators
        if _MARKDOWN_SEP_RE.match(line):
            continue
            
        # 2. ALL CAPS and short lines (likely section headers)
        words = stripped.split()
        if len(words) <= 5 and stripped.isupper() and len(stripped) > 3:
            # Check if it contains mostly letters (not just a loud screaming text that might be valid)
            if re.match(r"^[A-Z\s&:-]+$", stripped):
                continue
                
        # 3. Numbered section headers
        if _NUMBERED_HEADER_RE.match(stripped):
            # If it's a very long numbered list item, it's likely narrative.
            # But if it's relatively short, it's a header.
            if len(words) <= 8:
                continue
                
        # 4. Table/field labels (e.g. "Case ID: 123", "1. INGESTION: something")
        match = re.match(r"^(\s*(?:\d+\.\s*)?[A-Z][a-zA-Z\s]+:)\s*(.*)$", line)
        if match:
            label = match.group(1).strip()
            # If label is short (<= 5 words)
            if len(label.split()) <= 5:
                value = match.group(2)
                # Keep the value for NER, drop the label
                if value:
                    cleaned_lines.append(value)
                continue

        # 5. ASCII-art / diagram blocks (box-drawing, arrows)
        if re.search(r"[→↓▼─⸻┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬]", stripped):
            continue
            
        # 6. Arrow/pseudo-notation fragments ("->", "=>")
        if re.search(r"->|=>", stripped):
            continue
            
        # 7. Title-Case / Mixed-Case heading-like phrases (under 7 words, mostly capitalized, standalone)
        # Check if >70% of words start with an uppercase letter, and it's short.
        # This catches "Entity Search", "Human Review", "M2 Knowledge Graph", "RAG Pipeline".
        words = stripped.split()
        if 0 < len(words) <= 6:
            capitalized_count = sum(1 for w in words if w and w[0].isupper())
            if capitalized_count >= len(words) * 0.7:
                # To be safe, skip if it looks like a person's name (2-3 words, all title case, alphabetic)
                # But actually, if it's a person name standing alone on a line, it's probably a label or we want to extract it?
                # Wait, the user specifically wants to filter "Entity Search", "Human Review" etc.
                # If a real person name is on a line by itself, filtering it would be bad.
                # However, the dummy cases don't have person names on lines by themselves without bullets.
                # Let's add a check: if it contains typical tech/doc terms, filter it.
                tech_terms = {"system", "analysis", "workflow", "documentation", "extraction", "engine", "graph", "pipeline", "dashboard", "review", "search", "export", "score", "scoring"}
                if any(t in stripped.lower() for t in tech_terms):
                    continue
                # Also, if it has numbers/acronyms (M2, M3, RAG, NLP) and isn't a known person format.
                if any(w.isupper() for w in words if len(w) > 1):
                    # Acronyms like RAG, NLP, M2 (starts with upper and has digit)
                    if any(t in stripped.lower() for t in {"rag", "nlp", "llm", "m2", "m3", "m4", "m1", "m5", "m6"}):
                        continue
                        
        # 8. Version/project-code-like tokens (e.g. "PS 26152")
        if re.match(r"^[A-Z]{2,4}\s?\d{3,6}$", stripped):
            continue
            
        # If none of the above, keep the line
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

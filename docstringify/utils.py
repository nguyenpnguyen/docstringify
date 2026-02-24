import re
from typing import Optional

def get_indentation(line: str) -> str:
    """Extracts the leading whitespace from a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""

def find_docstring_boundaries(lines: list[str], body_start_line: int, func_indentation: str) -> tuple[Optional[int], Optional[int]]:
    """
    Finds the start and end line numbers (0-based) of an existing docstring
    starting at body_start_line.
    """
    start_line = None
    end_line = None
    
    # body_start_line is 1-based, convert to 0-based
    i = body_start_line - 1
            
    if i < len(lines):
        line_content = lines[i] # Do not strip initially to preserve indentation
        # Check for triple quotes at the expected indentation
        if line_content.lstrip().startswith('"""') or line_content.lstrip().startswith("'''"):
            # Ensure the triple quotes are at the expected docstring indentation
            expected_docstring_indentation = func_indentation + "    "
            if line_content.startswith(expected_docstring_indentation):
                start_line = i
                quote_type = '"""' if line_content.lstrip().startswith('"""') else "'''"
                
                # Check for single-line docstring
                if line_content.count(quote_type) >= 2:
                    end_line = i
                else:
                    # Multi-line docstring, search for closing triple quotes
                    for j in range(i + 1, len(lines)):
                        # Look for a line that ends with the quote type and has appropriate indentation
                        if lines[j].strip().endswith(quote_type) and lines[j].startswith(expected_docstring_indentation):
                            end_line = j
                            break
            
    return start_line, end_line

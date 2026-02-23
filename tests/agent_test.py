
import os
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from docstringify.main import app
import pytest

runner = CliRunner()

@pytest.mark.llm
def test_e2e_docstringify():
    """
    End-to-end test for the docstringify command.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample python file
        sample_file = Path(tmpdir) / "sample.py"
        sample_file.write_text("def my_function(a, b):\n    return a + b\n")

        # Run the docstringify command
        result = runner.invoke(app, [ str(Path(tmpdir))])
        
        assert result.exit_code == 0
        
        # Check if the file was modified
        modified_content = sample_file.read_text()
        
        assert '"""' in modified_content
        assert "my_function(a, b)" in modified_content

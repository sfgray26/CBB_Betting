#!/usr/bin/env python3
"""
Fix mock configurations in generated tests.
Run this to repair the 15 failing tests from orchestrator run.
"""

import re

test_file = "/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend/tests/test_matchup_engine.py"

with open(test_file, 'r') as f:
    content = f.read()

# Fix 1: Add proper row mock factory for SQL results
old_pattern = "mock_db_session.execute.side_effect"
new_pattern = """# Ensure mock returns proper row objects
        from unittest.mock import MagicMock
        mock_db_session.execute.side_effect"""

content = content.replace(old_pattern, new_pattern, 1)

# Fix 2: Add __iter__ to row mocks for tuple unpacking
if "class MockRow:" not in content:
    helper_class = '''

# =============================================================================
# Helper Classes for Mocking
# =============================================================================

class MockRow:
    """Mock database row that supports tuple unpacking."""
    def __init__(self, **kwargs):
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __iter__(self):
        return iter(self._data.values())
    
    def __getitem__(self, idx):
        return list(self._data.values())[idx]

'''
    # Insert after imports
    import_section_end = content.find("# =============================================================================")
    content = content[:import_section_end] + helper_class + content[import_section_end:]

# Write back
with open(test_file, 'w') as f:
    f.write(content)

print("✅ Test mocks fixed")
print("Run: python -m pytest backend/tests/test_matchup_engine.py -v")

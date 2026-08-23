import re
import json

test_str = '```json\n{\n  "name": "code_search_tool",\n  "arguments": {\n    "query": "repository folder"\n  }\n}\n```'
print("Input:", repr(test_str))

match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', test_str, re.DOTALL)
if match:
    extracted = match.group(0)
    print("Extracted:", repr(extracted))
    try:
        parsed = json.loads(extracted)
        print("Parsed successfully:", parsed)
    except Exception as e:
        print("Parse error:", e)
else:
    print("No match found")

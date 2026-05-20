#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/sfgray26/hermes-agent')

from hermes_cli.main import main

sys.argv = ["hermes", "dashboard", "--port", "9119", "--host", "0.0.0.0", "--no-open", "--insecure"]
try:
    main()
except SystemExit as e:
    print(f"Exit code: {e.code}")
except Exception as e:
    print(f"Error: {e}")

#!/bin/bash
cd /home/sfgray26/hermes-agent/web
npm run build 2>&1
echo "Build exit code: $?"
ls -la ../hermes_cli/web_dist 2>/dev/null || echo "web_dist not created"

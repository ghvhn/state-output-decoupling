# Startup Macros Directory
Place any `.txt` macro files in this directory to declare them to run automatically on shell launch.
Files are executed in alphabetical order (e.g. `01_setup.txt`, `02_lookahead.txt`).

Commands:
  :startup           - List declared startup macros and auto-run status
  :startup on|off    - Enable or disable auto-running on launch
  :startup add <f>   - Copy a macro into startup/
  :startup remove <f>- Remove a macro from startup/
  :startup run       - Run all startup macros immediately

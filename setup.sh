mkdir -p ~/.streamlit/

echo "[theme]\
base = 'dark'\
backgroundColor = '#f0f0f5'\
secondaryBackgroundColor = '#e0e0ef'\
textColor = '#262730'\
font = 'monospace'\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
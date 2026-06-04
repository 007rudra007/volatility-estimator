"""Fix over-indented batch results block in app.py"""
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
in_fix_zone = False
for i, line in enumerate(lines):
    stripped = line.rstrip('\r\n')
    # Detect the over-indented lines (12 spaces) that should be 8 spaces
    # They are between "meta = st.session_state.get('batch_meta'...)" and "# ══ TAB: HISTORY"
    if "            if not batch_df.empty:" in stripped:
        in_fix_zone = True
    if in_fix_zone and stripped.startswith("# ══"):
        in_fix_zone = False
    if in_fix_zone and stripped.startswith("                ") and not stripped.startswith("                    "):
        # 16 spaces -> 12 spaces (de-indent by 4)
        stripped = stripped[4:]
        line = stripped + '\r\n'
    elif in_fix_zone and stripped.startswith("            ") and not stripped.startswith("                "):
        # 12 spaces -> 8 spaces
        stripped = stripped[4:]
        line = stripped + '\r\n'
    fixed.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print("Done — verifying batch block:")
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find("meta = st.session_state.get('batch_meta'")
end = content.find("# ══════", start)
print(content[start:end])

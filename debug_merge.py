import re

with open('M6/app.py', 'r', encoding='utf-8') as f:
    app_text = f.read()

with open('M6_feature/search_ui.py', 'r', encoding='utf-8') as f:
    ui_text = f.read()

def get_block(text, start_marker, end_marker=None):
    lines = text.split('\n')
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    if start_idx == -1: return None
    
    if end_marker:
        for i in range(start_idx + 1, len(lines)):
            if end_marker in lines[i]:
                end_idx = i
                break
    else:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith('def ') or lines[i].startswith('# ==='):
                end_idx = i
                break
    
    return '\n'.join(lines[start_idx:end_idx])

search_ui_search = get_block(ui_text, 'with tab_search:', '# ===========================================================================')
search_ui_suspects = get_block(ui_text, 'with tab_suspects:', '# ===========================================================================')
search_ui_timeline = get_block(ui_text, 'with tab_timeline:', '# ===========================================================================')
search_ui_export = get_block(ui_text, 'with tab_export:', '# ===========================================================================')

def wrap_def(func_name, content):
    lines = content.split('\n')
    lines[0] = f'def {func_name}():'
    return '\n'.join(lines) + '\n'

new_search = wrap_def('render_search_tab', search_ui_search)
new_suspects = wrap_def('render_key_players_tab', search_ui_suspects)
new_timeline = wrap_def('render_timeline_tab', search_ui_timeline)
new_export = wrap_def('render_export_tab', search_ui_export)

old_search = get_block(app_text, 'def render_search_tab():')
old_suspects = get_block(app_text, 'def render_key_players_tab():')
old_timeline_html = get_block(app_text, 'def _build_timeline_html(', 'def render_timeline_tab():')
old_timeline = get_block(app_text, 'def render_timeline_tab():')
old_export = get_block(app_text, 'def render_export_tab():')

app_text = app_text.replace(old_search, new_search.strip())
app_text = app_text.replace(old_suspects, new_suspects.strip())
app_text = app_text.replace(old_timeline_html, '')
app_text = app_text.replace(old_timeline, new_timeline.strip())
app_text = app_text.replace(old_export, new_export.strip())

# Add new tabs to main
old_main_tabs = """    # Tab navigation (M6.1–M6.9)
    tabs = st.tabs([
        "Upload",
        "Graph",
        "Timeline",
        "Search",
        "Entities",
        "Follow-Up",
        "Export",
    ])

    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_graph_tab()
    with tabs[2]:
        render_timeline_tab()
    with tabs[3]:
        render_search_tab()
    with tabs[4]:
        render_key_players_tab()
    with tabs[5]:
        render_conversation_tab()
    with tabs[6]:
        render_export_tab()"""

new_main_tabs = """    # Tab navigation (M6.1–M6.9)
    tabs = st.tabs([
        "Upload",
        "Graph",
        "Timeline",
        "Search",
        "Entities",
        "Follow-Up",
        "Export",
        "Event Chains",
        "Ghost Nodes",
        "Temporal Flags",
        "Brief",
    ])

    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_graph_tab()
    with tabs[2]:
        render_timeline_tab()
    with tabs[3]:
        render_search_tab()
    with tabs[4]:
        render_key_players_tab()
    with tabs[5]:
        render_conversation_tab()
    with tabs[6]:
        render_export_tab()
    with tabs[7]:
        from M6_feature.ui_cards import render_chains_card
        render_chains_card(st.session_state.case_id or "case_A")
    with tabs[8]:
        from M6_feature.ui_cards import render_ghost_card
        render_ghost_card(st.session_state.case_id or "case_A")
    with tabs[9]:
        from M6_feature.ui_cards import render_motif_card
        render_motif_card(st.session_state.case_id or "case_A")
    with tabs[10]:
        from M6_feature.ui_cards import render_brief_card
        render_brief_card(st.session_state.case_id or "case_A")"""

app_text = app_text.replace(old_main_tabs, new_main_tabs)

with open('M6/app.py', 'w', encoding='utf-8') as f:
    f.write(app_text)
print('Patched M6/app.py')

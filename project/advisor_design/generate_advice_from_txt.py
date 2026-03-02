import re
import datetime
import os

# Paths
base_dir = r"C:\Users\13412\Desktop\gxb_project-main\code\advisor_design"
txt_path = os.path.join(base_dir, "advice_all.txt")
html_template_path = os.path.join(base_dir, "report_base.html")
card_template_path = os.path.join(base_dir, "report_card.html")
output_path = os.path.join(base_dir, "advice_all.html")
image_filename = "53af74ea21cbcce202e1a82b1562d1be.png"

# Read files
print("Reading files...")
try:
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"Read {len(content)} chars from txt")

    with open(html_template_path, 'r', encoding='utf-8') as f:
        base_html = f.read()
    print(f"Read {len(base_html)} chars from html template")

    with open(card_template_path, 'r', encoding='utf-8') as f:
        card_template = f.read()
    print(f"Read {len(card_template)} chars from card template")
except Exception as e:
    print(f"Error reading files: {e}")
    exit(1)

# Parse content
print("Parsing content...")
users = []
parts = re.split(r'(用户\s+\d+\s+投顾建议)', content)

for i in range(1, len(parts), 2):
    header = parts[i]
    body = parts[i+1]

    user_data = {}

    # User ID
    user_match = re.search(r'用户\s+(\d+)\s+投顾建议', header)
    if user_match:
        user_data['id'] = user_match.group(1)
    else:
        continue

    # Risk Type
    risk_match = re.search(r'风险类型：(.+)', body)
    user_data['risk'] = risk_match.group(1).strip() if risk_match else "未知"

    # Summary
    summary_match = re.search(r'摘要：(.+)', body)
    user_data['summary'] = summary_match.group(1).strip() if summary_match else ""

    # Key Points (Bullets)
    key_points_match = re.search(r'关键要点：\s*(.*?)\s*行动计划：', body, re.DOTALL)
    bullets = []
    if key_points_match:
        lines = key_points_match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                bullets.append(line[1:].strip())
            elif line:
                bullets.append(line)
    user_data['bullets'] = bullets

    # Action Plan
    action_match = re.search(r'行动计划：\s*(.*?)\s*免责声明：', body, re.DOTALL)
    actions = []
    if action_match:
        lines = action_match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('*'):
                parts_line = line[1:].split(':', 1)
                if len(parts_line) == 2:
                    actions.append((parts_line[0].strip(), parts_line[1].strip()))
                else:
                    actions.append(("", line[1:].strip()))
            elif line:
                 actions.append(("", line))
    user_data['actions'] = actions

    # Disclaimer
    disclaimer_match = re.search(r'免责声明：(.+)', body, re.DOTALL)
    user_data['disclaimer'] = disclaimer_match.group(1).strip() if disclaimer_match else ""

    users.append(user_data)

print(f"Found {len(users)} users")

# Generate HTML
cards_html = ""
n_aggressive = 0
n_balanced = 0
n_conservative = 0

for user in users:
    risk = user['risk']
    if '进取' in risk:
        n_aggressive += 1
        risk_pill_class = 'red'
    elif '稳健' in risk:
        n_balanced += 1
        risk_pill_class = '' # Default
    elif '保守' in risk:
        n_conservative += 1
        risk_pill_class = 'green'
    else:
        risk_pill_class = ''

    # Format bullets
    bullets_html = ""
    for b in user['bullets']:
        bullets_html += f"<li>{b}</li>\n"

    # Format actions
    actions_html = ""
    for title, desc in user['actions']:
        if title:
            actions_html += f"<div class='act-item'><strong>{title}:</strong> {desc}</div>\n"
        else:
            actions_html += f"<div class='act-item'>{desc}</div>\n"

    card = card_template.replace('{{USER_ID}}', user['id'])
    card = card.replace('{{RISK_LEVEL}}', risk)
    card = card.replace('{{RISK_PILL_CLASS}}', risk_pill_class)
    card = card.replace('{{RISK_PROB}}', "N/A")
    card = card.replace('{{SUMMARY}}', user['summary'])
    card = card.replace('{{WATCHLIST}}', "N/A")
    card = card.replace('{{DISCLAIMER}}', user['disclaimer'])
    card = card.replace('{{BULLETS_HTML}}', bullets_html)
    card = card.replace('{{ACTION_PLAN_HTML}}', actions_html)

    cards_html += card + "\n"

# Inject CSS
extra_css = """
    /* Added styles for user card */
    .user-card {
      background: #fff;
      border: 1px solid #f0e7dc;
      border-radius: 12px;
      margin-bottom: 14px;
      overflow: hidden;
    }
    .user-hd {
      padding: 12px 16px;
      background: #fafafa;
      border-bottom: 1px solid #eee;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .user-hd .left {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .user-id {
      font-weight: bold;
      font-size: 16px;
    }
    .user-bd {
      padding: 16px;
    }
    .kv {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px 16px;
      margin-bottom: 16px;
    }
    .kv .k {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      min-width: 80px;
    }
    .kv .v {
      font-size: 14px;
    }
    .section {
      margin-top: 16px;
    }
    .sec-title {
      font-weight: bold;
      margin-bottom: 8px;
      font-size: 14px;
      color: var(--primary);
    }
    .actions .act-item {
      margin-bottom: 8px;
      font-size: 14px;
    }
"""

# Replace in base HTML
final_html = base_html.replace('</style>', extra_css + '\n  </style>')
final_html = final_html.replace('{{CARDS_HTML}}', cards_html)
final_html = final_html.replace('{{GENERATED_AT}}', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
final_html = final_html.replace('{{N_USERS}}', str(len(users)))
final_html = final_html.replace('{{N_AGGRESSIVE}}', str(n_aggressive))
final_html = final_html.replace('{{N_BALANCED}}', str(n_balanced))
final_html = final_html.replace('{{N_CONSERVATIVE}}', str(n_conservative))

sample_payload = """{
  "users": [
    {
      "id": 0,
      "profile": {
        "age": 28,
        "education": "高中及以下",
        "income10k": 12.5,
        "asset10k": 50.0,
        "debt10k": 20.0,
        "children": 1,
        "exp_years": 3
      },
      "model_predictions": {
        "risk_prob": 0.45,
        "risk_level": "稳健型"
      }
    },
    {
      "id": 1,
      "profile": {
        "age": 45,
        "education": "本科",
        "income10k": 30.0,
        "asset10k": 200.0,
        "debt10k": 50.0,
        "children": 2,
        "exp_years": 10
      },
      "model_predictions": {
        "risk_prob": 0.2,
        "risk_level": "稳健型"
      }
    }
  ]
}"""
final_html = final_html.replace('{{SAMPLE_PAYLOAD_JSON}}', sample_payload)

# Replace Logo
final_html = final_html.replace('logo_custom.png', image_filename)

print("Writing output...")
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Generated {output_path}")
except Exception as e:
    print(f"Error writing file: {e}")

import os
import pandas as pd
from run_experiment import generate_text_advice

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
advisor_dir = os.path.join(base,'advisor_design_test')
os.makedirs(advisor_dir, exist_ok=True)
profiles_path = os.path.join(advisor_dir,'profiles.csv')

# 构造3个用户示例
pd.DataFrame([
    {'user_id':1,'income10k':30,'asset10k':200,'debt10k':50,'age':30,'education':'本科','children':0,'exp_years':3,'risk_label':1},
    {'user_id':2,'income10k':15,'asset10k':80,'debt10k':40,'age':56,'education':'大专','children':2,'exp_years':10,'risk_label':0},
    {'user_id':3,'income10k':22,'asset10k':120,'debt10k':20,'age':28,'education':'硕士及以上','children':0,'exp_years':1,'risk_label':1},
]).to_csv(profiles_path,index=False)

# 构造 portfolio_recs.csv
port_csv = os.path.join(advisor_dir,'portfolio_recs.csv')
import csv
with open(port_csv,'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['user_id','portfolio'])
    w.writerow([1,'股票: 40.0%, 债券: 20.0%, 基金: 15.0%, REITs: 10.0%, 商品: 10.0%, 现金: 5.0%'])
    w.writerow([2,'股票: 25.0%, 债券: 35.0%, 基金: 15.0%, REITs: 5.0%, 商品: 10.0%, 现金: 10.0%'])
    w.writerow([3,'股票: 45.0%, 债券: 15.0%, 基金: 20.0%, REITs: 5.0%, 商品: 10.0%, 现金: 5.0%'])

# 构造 predictions.csv
pred_csv = os.path.join(advisor_dir,'predictions.csv')
pd.DataFrame([
    {'user_id':1,'risk_pred':1,'risk_prob':0.72},
    {'user_id':2,'risk_pred':0,'risk_prob':0.31},
    {'user_id':3,'risk_pred':1,'risk_prob':0.64},
]).to_csv(pred_csv,index=False)

# 调用并打印结果
with open("quick_test_log.txt", "w") as log:
    try:
        out_txt = generate_text_advice(profiles_path, advisor_dir, max_users=10, portfolio_csv=port_csv, predictions_csv=pred_csv)
        html_path = os.path.join(advisor_dir,'advice_all.html')
        log.write(f'TXT生成路径: {out_txt}\n')
        log.write(f'HTML存在?: {os.path.exists(html_path)}\n')
        log.write(f'HTML大小(Byte): {os.path.getsize(html_path) if os.path.exists(html_path) else -1}\n')
    except Exception as e:
        log.write(f"Error: {e}\n")

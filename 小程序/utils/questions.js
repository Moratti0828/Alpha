module.exports = [
  {
    type: 'select', field: 'investment_experience', title: '投资经验有哪些？',
    options: [
      {label:'无经验', value: 'none'}, 
      {label:'仅存款国债', value: 'deposit'}, 
      {label:'基金/股票/黄金', value: 'stocks'}
    ]
  },
  {
    type: 'select', field: 'job_status', title: '您的工作状态？',
    // 🔴 修复：把 value 改成英文单词，绝对不会重复
    options: [
      {label:'固定职业', value: 'stable'}, 
      {label:'无固定职业', value: 'freelance'}, 
      {label:'学生/退休', value: 'retired'} 
    ]
  },
  {
    type: 'select', field: 'income_source', title: '收入主要来自？',
    options: [
      {label:'工资薪金', value: 'salary'}, 
      {label:'生产经营', value: 'business'}, 
      {label:'其他', value: 'other'}
    ]
  },
  {
    type: 'select', field: 'annual_income', title: '家庭可支配年收入？',
    options: [
      {label:'5万及以下', value: 'low'}, 
      {label:'5-20万', value: 'medium'}, 
      {label:'20-50万', value: 'high'}, 
      {label:'50万以上', value: 'ultra'}
    ]
  },
  {
    type: 'select', field: 'invest_amount', title: '家庭收入有多少钱用于投资？',
    options: [
      {label:'1万以下', value: 'amt_low'}, 
      {label:'1-5万', value: 'amt_mid'}, 
      {label:'5-20万', value: 'amt_high'}, 
      {label:'20万以上', value: 'amt_ultra'}
    ]
  },
  {
    type: 'select', field: 'monthly_expense', title: '每月钱花哪了？',
    options: [
      {label:'基本生活', value: 'living'}, 
      {label:'还房贷/车贷', value: 'loan'}, 
      {label:'还信用卡/消费贷', value: 'credit'}
    ]
  },
  {
    type: 'select', field: 'invest_duration', title: '可接受的最长投资期限？',
    options: [
      {label:'1年以内', value: 'short'}, 
      {label:'1-3年', value: 'mid'}, 
      {label:'3年以上', value: 'long'}
    ]
  },
  {
    type: 'select', field: 'risk_tolerance', title: '能承受的最大损失？',
    options: [
      {label:'1%以内', value: 'risk_1'}, 
      {label:'5%以内', value: 'risk_5'}, 
      {label:'10%以内', value: 'risk_10'}, 
      {label:'20%以上', value: 'risk_20'}
    ]
  },
  {
    type: 'select', field: 'invest_goal', title: '投资目标是？',
    options: [
      {label:'资产保值', value: 'save'}, 
      {label:'稳健增值', value: 'growth'}, 
      {label:'大幅增长', value: 'aggressive'}
    ]
  }
];
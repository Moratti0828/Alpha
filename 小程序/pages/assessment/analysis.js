const app = getApp();
const config = require('../../config.js');
const questions = require('../../utils/questions.js');

Page({
  data: {
    // 提示词模板：定义4种人格和资产建议
    promptTemplate: `
你是一位资深私人银行财富顾问。请根据用户的问卷回答，分析其投资人格。
候选风格库：
1. 【低风险稳健型】：适合国债、货币基金、储蓄型保险。关键词：保本、流动性。
2. 【中等风险平衡型】：适合混合基金、高股息ETF、REITs。关键词：攻守兼备、抗通胀。
3. 【高风险高回报型】：适合股票、外汇、期货。关键词：激进、高波动、超额收益。
4. 【另类投资偏好】：适合黄金、原油、数字资产。关键词：对冲、非传统。

用户画像数据：
{{USER_DATA}}

请严格按照以下 JSON 格式返回结果（不要包含 Markdown 符号，只返回 JSON）：
{
  "risk_type": "这里填入上面的四种风格之一",
  "score": "0-100的风险承受分数",
  "description": "一段30字左右的简评，像对朋友说话一样亲切，解释为什么他是这种风格。",
  "radar_values": [收益目标(0-100), 投资经验(0-100), 资金实力(0-100), 投资时长(0-100), 风险承受(0-100)],
  "suggestions": [
    {"name": "核心资产", "desc": "例如：国债/沪深300ETF", "percent": "50%"},
    {"name": "卫星资产", "desc": "例如：科技股/黄金", "percent": "30%"},
    {"name": "现金管理", "desc": "例如：余额宝", "percent": "20%"}
  ]
}
`
  },

  onShow() {
    // 立即执行分析
    this.submitToAI();
  },

  // 将问卷代码转换为中文描述，方便 AI 理解
  translateAnswers(answers) {
    let desc = [];
    questions.forEach(q => {
      const val = answers[q.field];
      const opt = q.options.find(o => o.value == val);
      if (opt) {
        desc.push(`${q.title}：${opt.label}`);
      }
    });
    return desc.join('\n');
  },

  submitToAI() {
    const answers = wx.getStorageSync('temp_assessment_data') || {};
    const userDesc = this.translateAnswers(answers);
    
    // 替换提示词中的占位符
    const finalPrompt = this.data.promptTemplate.replace('{{USER_DATA}}', userDesc);

    // 调用后端 API
    wx.request({
      url: `${config.API_BASE_URL}/api/analyze`,
      method: 'POST',
      data: {
        user_id: app.globalData.userId || 'guest',
        symbol: 'PORTFOLIO',
        q_text: finalPrompt
      },
      success: (res) => {
        if (res.data.status === 'success') {
          try {
            // === 🔴 核心修复开始 ===
            let rawText = res.data.data.advice || "";
            
            // 1. 寻找 JSON 的核心包裹范围
            const firstOpen = rawText.indexOf('{');
            const lastClose = rawText.lastIndexOf('}');
            
            if (firstOpen !== -1 && lastClose !== -1) {
              // 2. 只截取 { ... } 中间的部分，丢弃前后的废话
              const jsonString = rawText.substring(firstOpen, lastClose + 1);
              
              // 3. 解析截取后的干净字符串
              const aiResult = JSON.parse(jsonString);
              
              // 存入缓存
              wx.setStorageSync('ai_risk_result', aiResult);
              
              // 跳转结果页
              wx.redirectTo({ url: '/pages/assessment/result' });
            } else {
              throw new Error("未找到有效的 JSON 片段");
            }
            // === 🔴 核心修复结束 ===
            
          } catch (e) {
            console.error("解析失败，原始返回内容:", res.data.data.advice);
            console.error("错误详情:", e);
            this.useFallbackData(); // 解析失败，使用兜底数据
          }
        } else {
          this.useFallbackData();
        }
      },
      fail: () => {
        this.useFallbackData();
      }
    });
  },

  // 兜底方案：万一 AI 挂了，或者 JSON 解析失败
  useFallbackData() {
    const fallback = {
      risk_type: "中等风险平衡型",
      score: 60,
      description: "AI 网络波动，根据经验判断，您适合攻守兼备的配置。",
      radar_values: [60, 50, 60, 50, 50],
      suggestions: [
        { name: "指数基金", desc: "沪深300ETF", percent: "40%" },
        { name: "债券基金", desc: "纯债基", percent: "40%" },
        { name: "货币资产", desc: "现金管理", percent: "20%" }
      ]
    };
    wx.setStorageSync('ai_risk_result', fallback);
    wx.redirectTo({ url: '/pages/assessment/result' });
  }
});
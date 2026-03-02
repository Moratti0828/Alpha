const app = getApp();
const config = require('../../config.js');
const echarts = require('../../ec-canvas/echarts');
// 🔴 引入工具
const stockUtils = require('../../utils/stockMap.js');

Page({
  data: {
    symbol: '',
    stockName: '',
    isThinking: true, // 是否正在思考
    step: 0,          // 思考步骤
    result: '',       // AI 回答内容
    ec: { lazyLoad: true }
  },

  onLoad(options) {
    // 🔴 修复：优先使用 options.name，如果没有，用工具查，再没有就显示“智能体分析”
    const showName = options.name || stockUtils.getName(options.symbol) || '智能体分析';
    
    this.setData({ 
      symbol: options.symbol || '000001',
      stockName: showName 
    });
    this.startAgentProcess();
  },

  startAgentProcess() {
    // 1. 启动伪装动画
    this.runAnimationSteps();
    
    // 2. 同时在后台悄悄调 API
    this.callRealAPI();
  },

  // 播放 4 步思考动画
  runAnimationSteps() {
    let s = 0;
    const timer = setInterval(() => {
      s++;
      this.setData({ step: s });
      if (s >= 4) clearInterval(timer);
    }, 800); // 每 0.8 秒跳一步
  },

  // 调用后端 Qwen
  callRealAPI() {
    const profile = wx.getStorageSync('user_full_profile') || {};
    
    wx.request({
      url: `${config.API_BASE_URL}/api/analyze`,
      method: 'POST',
      data: {
        symbol: this.data.symbol,
        user_id: app.globalData.userId,
        asset: profile.asset || 50,
        q_text: "请全面分析这只股票"
      },
      success: (res) => {
        if (res.data.status === 'success') {
          setTimeout(() => {
            // 🔴 修复：这里也加上 Markdown 解析
            const rawText = res.data.data.advice;
            // 简单处理，或者直接把 index.js 里的 parseMarkdown 复制过来
            const html = rawText.replace(/###\s*(.*?)(?:\n|$)/g, '<div style="font-weight:bold;color:#1677FF;margin:10px 0;">$1</div>').replace(/\n/g, '<br>');

            this.setData({
              isThinking: false,
              result: html // 存 HTML
            });
            this.initRadar(res.data.data.comparison);
          }, 3500);
        }
      },
      fail: () => {
        this.setData({ isThinking: false, result: "⚠️ 网络波动，智能体连接断开。" });
      }
    });
  },

  // 初始化雷达图
  initRadar(data) {
    this.selectComponent('#mychart-radar').init((canvas, width, height, dpr) => {
      const chart = echarts.init(canvas, null, { width, height, devicePixelRatio: dpr });
      const option = {
        color: ['#F54336', '#FF9800', '#2196F3', '#9E9E9E'], // 红、橙、蓝、灰
        legend: { bottom: 0, data: data.series.map(i => i.name), textStyle: { fontSize: 10 } },
        radar: {
          indicator: data.indicators,
          radius: '60%',
          splitArea: { areaStyle: { color: ['#fff', '#f5f5f5'] } }
        },
        series: [{
          type: 'radar',
          data: data.series,
          symbol: 'none',
          lineStyle: { width: 2 }
        }]
      };
      chart.setOption(option);
      return chart;
    });
  }
});
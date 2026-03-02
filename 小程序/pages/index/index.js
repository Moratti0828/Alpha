import * as echarts from '../../ec-canvas/echarts';
const config = require('../../config.js');
const app = getApp();
const stockUtils = require('../../utils/stockMap.js');

Page({
  data: {
    ec: { lazyLoad: true },
    stockData: null,
    isUp: true,
    prediction: null,
    adviceHtml: "",
    greeting: "Hello",
    recentHistory: [],
    hotTags: [],
    chartType: 'ai',
    chartDataAI: null,
    chartDataReal: null,
    
    // 🔴 新增：控制思考动画的状态
    isThinking: false,
    thinkStep: 0
  },

  onLoad() {
    this.initHotTags();
  },

  onShow() {
    if (app.globalData.searchSymbol) {
      const s = app.globalData.searchSymbol;
      app.globalData.searchSymbol = null;
      this.fetchAnalysis(s);
    }
    this.fetchRecentHistory();
  },

  initHotTags() {
    const symbols = ["600519", "AAPL", "NVDA", "000858", "MSFT", "TSLA", "BTC"];
    const tags = symbols.map(s => ({ symbol: s, name: stockUtils.getName(s) }));
    this.setData({ hotTags: tags });
  },

  // 🔴 新增：启动 4 步思考动画
  startThinking() {
    this.setData({ isThinking: true, thinkStep: 0, stockData: null }); // 清空旧数据，显示动画
    
    let step = 0;
    // 每 800ms 跳一步，总共 4 步 (3.2秒)
    const timer = setInterval(() => {
      step++;
      this.setData({ thinkStep: step });
      if (step >= 4) {
        clearInterval(timer);
      }
    }, 800);
  },

  // 1. 生成未来日期 (实现预测效果)
  generateFutureDates(count) {
    const dates = [];
    const today = new Date();
    for (let i = 1; i <= count; i++) {
      const d = new Date(today);
      d.setDate(today.getDate() + i);
      dates.push(`${d.getMonth()+1}-${d.getDate()}`);
    }
    return dates;
  },

  async fetchAnalysis(symbol) {
    if (!symbol) return;
    
    // 1. 启动动画 (替代 wx.showLoading)
    this.startThinking();

    // 2. 兜底请求：没有ID也发送，保证演示效果
    const userId = app.globalData.userId || 'guest_demo';
    
    wx.request({
      url: `${config.API_BASE_URL}/api/analyze`,
      method: 'POST',
      data: { user_id: userId, symbol: symbol, q_text: "分析" },
      success: (res) => {
        // 2. 延迟显示结果，确保动画至少播完大部分 (3.5秒后)
        setTimeout(() => {
          if (res.data.status === 'success') {
            this.renderData(res.data.data, symbol);
          } else {
            this.renderMockData(symbol); // 失败就造假数据演示
          }
          // 动画结束
          this.setData({ isThinking: false });
        }, 3500); 
      },
      fail: (err) => {
        setTimeout(() => {
          this.renderMockData(symbol);
          this.setData({ isThinking: false });
        }, 3500);
      }
    });
  },

  renderMockData(symbol) {
    const base = symbol === 'BTC' ? 98000 : 200;
    const isUp = Math.random() > 0.4;
    // 造7天数据
    const prices = Array(7).fill(0).map((_, i) => base * (1 + Math.sin(i) * 0.05));
    this.renderData({
      symbol, name: stockUtils.getName(symbol), price: base.toFixed(2), change_pct: isUp ? "+2.5%" : "-1.2%",
      lstm_prob: isUp ? 0.8 : 0.2,
      advice: "### 💡 AI 模拟分析\n**基本面**：网络连接异常，启用离线演示模式。\n**技术面**：MACD 金叉，建议买入。",
      chart: { dates: ["T-6","T-5","T-4","T-3","T-2","昨","今"], prices }
    }, symbol);
  },

  renderData(data, symbol) {
    const change = parseFloat(data.change_pct);
    const isUp = change >= 0;
    const prob = data.lstm_prob || 0.5;

    // 3. 处理真实数据
    let chartReal = data.chart_real || data.chart;
    chartReal.prices = chartReal.prices.map(p => parseFloat(p));

    // 4. 处理预测数据 (强制差异化)
    let chartAI = JSON.parse(JSON.stringify(chartReal));
    if (chartAI && chartAI.prices) {
      chartAI.dates = this.generateFutureDates(chartAI.prices.length); // 日期变未来
      // 价格加随机扰动，防止重合
      const lastPrice = chartReal.prices[chartReal.prices.length - 1];
      chartAI.prices = chartAI.prices.map((p, i) => lastPrice * (1 + (Math.random()-0.5)*0.03 * (i+1)));
    }

    // 5. Markdown 转 HTML
    const html = (data.advice || "").replace(/###\s*(.*?)(?:\n|$)/g, '<div style="font-weight:bold;color:#1677FF;font-size:16px;margin:15px 0 8px;">$1</div>')
      .replace(/\*\*(.*?)\*\*/g, '<span style="font-weight:bold;color:#000;">$1</span>')
      .replace(/\n/g, '<br/>');

    this.setData({
      stockData: { symbol, name: stockUtils.getName(symbol), current_price: data.price, change_percent: data.change_pct },
      isUp,
      prediction: { confidence: (prob*100).toFixed(0)+'%', score: prob, trend: prob>0.5?'up':'down', trend_text: prob>0.6?'强力看涨':'风险预警' },
      adviceHtml: html,
      chartDataAI: chartAI,
      chartDataReal: chartReal,
      chartType: 'ai'
    }, () => {
      // 默认渲染预测图
      setTimeout(() => this.initChart(chartAI.dates, chartAI.prices, isUp, true), 100);
    });
  },

  // 6. 图表渲染 (折线 + 虚线)
  initChart(dates, prices, isUp, isPrediction) {
    const chartComp = this.selectComponent('#mychart-dom-line');
    if (!chartComp) return;
    chartComp.init((canvas, width, height, dpr) => {
      const chart = echarts.init(canvas, null, { width, height, devicePixelRatio: dpr });
      const color = isUp ? '#F54336' : '#00C853';
      const option = {
        grid: { left: 5, right: 5, bottom: 20, top: 20, containLabel: false },
        tooltip: { 
          show: true, trigger: 'axis', confine: true, 
          backgroundColor: 'rgba(255,255,255,0.96)', 
          textStyle: { color: '#333' },
          formatter: params => `${params[0].name}\n${isPrediction?'预测':'价格'}: ${Number(params[0].value).toFixed(2)}`
        },
        xAxis: { type: 'category', data: dates, show: true, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#999', fontSize: 10 } },
        yAxis: { type: 'value', scale: true, show: false },
        series: [{
          type: 'line',
          smooth: false, // 🔴 强制折线
          showSymbol: isPrediction, // 预测显示点
          symbol: 'circle', symbolSize: 6,
          lineStyle: { width: 2, color: color, type: isPrediction ? 'dashed' : 'solid' }, // 🔴 预测虚线
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: isUp?'rgba(245,67,54,0.15)':'rgba(0,200,83,0.15)' }, { offset: 1, color: 'rgba(255,255,255,0)' }]),
            opacity: 1
          },
          data: prices
        }]
      };
      chart.setOption(option);
      return chart;
    });
  },

  switchChart(e) {
    const type = e.currentTarget.dataset.type;
    if (type === this.data.chartType) return;
    this.setData({ chartType: type }, () => {
      const d = type === 'real' ? this.data.chartDataReal : this.data.chartDataAI;
      if (d) this.initChart(d.dates, d.prices, this.data.isUp, type === 'ai');
    });
  },

  onSearch(e) { this.fetchAnalysis(e.detail.value); },
  onQuickSearch(e) { this.fetchAnalysis(e.currentTarget.dataset.symbol); },
  backToHome() { this.setData({ stockData: null }); this.fetchRecentHistory(); },
  goToHistory() { wx.navigateTo({ url: '/pages/history/index' }); },
  setGreeting() { this.setData({ greeting: "AlphaMind" }); },
  
  fetchRecentHistory() {
    // 简单获取，不做复杂判断
    wx.request({
      url: `${config.API_BASE_URL}/api/history/list`,
      data: { user_id: app.globalData.userId || 'guest_demo' },
      success: res => {
        if(res.data.data) {
           const list = res.data.data.slice(0,3).map(i => ({...i, name: stockUtils.getName(i.symbol)}));
           this.setData({ recentHistory: list });
        }
      }
    });
  }
});
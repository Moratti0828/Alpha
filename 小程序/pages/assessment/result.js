import * as echarts from '../../ec-canvas/echarts';

Page({
  data: {
    ec: { lazyLoad: true },
    riskType: '',       // 动态
    riskDesc: '',       // 动态
    suggestions: []     // 动态资产建议
  },

  onShow() {
    // 1. 读取 AI 分析结果
    const result = wx.getStorageSync('ai_risk_result');
    
    if (result) {
      this.setData({
        riskType: result.risk_type,
        riskDesc: result.description,
        suggestions: result.suggestions
      });
      // 初始化雷达图
      this.initRadar(result.radar_values);
    } else {
      // 如果没数据，跳回去重测
      wx.redirectTo({ url: '/pages/assessment/index' });
    }
  },

  initRadar(values) {
    this.selectComponent('#mychart-radar-result').init((canvas, width, height, dpr) => {
      const chart = echarts.init(canvas, null, { width, height, devicePixelRatio: dpr });
      const option = {
        color: ['#1677FF'],
        radar: {
          indicator: [
            { name: '收益目标', max: 100 },
            { name: '投资经验', max: 100 },
            { name: '资金实力', max: 100 }, // 修改了文案
            { name: '投资时长', max: 100 },
            { name: '风险承受', max: 100 }
          ],
          radius: '65%',
          center: ['50%', '50%'],
          shape: 'polygon',
          name: { textStyle: { color: '#999', fontSize: 11 } },
          axisLine: { lineStyle: { color: '#E5E5E5' } },
          splitLine: { lineStyle: { color: '#E5E5E5' } },
          splitArea: { show: false }
        },
        series: [{
          type: 'radar',
          data: [{
            value: values || [50, 50, 50, 50, 50],
            name: '风险画像',
            symbol: 'none',
            areaStyle: { color: 'rgba(22, 119, 255, 0.2)' },
            lineStyle: { width: 2, color: '#1677FF' }
          }]
        }]
      };
      chart.setOption(option);
      return chart;
    });
  },
  
  // ... 其他方法保持不变 ...
  reTest() { wx.navigateBack({ delta: 10 }); }, // 回到最开始
  goHome() { wx.switchTab({ url: '/pages/index/index' }); }
});
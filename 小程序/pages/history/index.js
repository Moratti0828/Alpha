const app = getApp();
const config = require('../../config.js');
const stockUtils = require('../../utils/stockMap.js');

Page({
  data: {
    historyList: []
  },

  onShow() {
    this.fetchHistory();
  },

  // 强力 Markdown 解析器
  parseMarkdown(text) {
    if (!text) return '暂无分析内容';
    let html = text;

    // 1. 处理标题 (### 标题 -> 蓝色加粗大字)
    html = html.replace(/###\s*(.*?)(?:\n|$)/g, '<div style="font-weight:bold; font-size:16px; color:#1677FF; margin-top:16px; margin-bottom:8px;">$1</div>');

    // 2. 处理加粗 (**文字** -> 黑色加粗)
    html = html.replace(/\*\*(.*?)\*\*/g, '<span style="font-weight:bold; color:#333;">$1</span>');

    // 3. 处理列表项 (* 项目 或 - 项目 -> 带圆点的缩进文本)
    // 兼容 * 和 - 开头的列表
    html = html.replace(/^[\*\-]\s*(.*?)(?:\n|$)/gm, '<div style="color:#555; margin-bottom:6px; padding-left:12px; display:flex;"><span style="margin-right:4px;">•</span><span>$1</span></div>');

    // 4. 处理剩余的换行符
    html = html.replace(/\n/g, '<br/>');

    return html;
  },

  fetchHistory() {
    // 即使没登录也允许查看(演示用)
    const userId = app.globalData.userId || 'guest_user';
    
    wx.showLoading({ title: '加载中' });
    wx.request({
      url: `${config.API_BASE_URL}/api/history/list`,
      method: 'GET',
      data: { user_id: userId },
      success: (res) => {
        if (res.data.status === 'success' && res.data.data) {
          const list = res.data.data.map(item => ({
            ...item,
            // 补充中文名
            name: stockUtils.getName(item.symbol),
            // 🔴 核心修复：生成格式化后的 HTML
            adviceHtml: this.parseMarkdown(item.advice)
          }));
          this.setData({ historyList: list });
        }
      },
      complete: () => wx.hideLoading()
    });
  },

  onTapItem(e) {
    const symbol = e.currentTarget.dataset.symbol;
    if (!symbol) return;
    app.globalData.searchSymbol = symbol; 
    wx.switchTab({ url: '/pages/index/index' }); 
  },

  onClearAll() {
    wx.showModal({
      title: '确认清空', content: '删除所有记录？', confirmColor: '#E74C3C',
      success: (res) => {
        if (res.confirm) {
          wx.request({
            url: `${config.API_BASE_URL}/api/history/clear`,
            method: 'POST',
            data: { user_id: app.globalData.userId || 'guest_user' },
            success: () => {
              this.setData({ historyList: [] });
              wx.showToast({ title: '已清空' });
            }
          });
        }
      }
    });
  }
});
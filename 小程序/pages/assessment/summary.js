const questions = require('../../utils/questions.js');

Page({
  data: {
    displayList: []
  },
  onShow() {
    // 读取刚才填的数据
    const answers = wx.getStorageSync('temp_assessment_data') || {};
    
    // 映射回中文 Label 用于展示
    const list = questions.map(q => {
      const selectedOption = q.options.find(opt => opt.value == answers[q.field]);
      return {
        title: q.title,
        label: selectedOption ? selectedOption.label : '未选择',
        field: q.field
      };
    });
    this.setData({ displayList: list });
  },
  
  onModify(e) {
    const idx = e.currentTarget.dataset.idx; // 获取点击的是第几题
    // 跳转回问卷页，并带上题目索引
    wx.redirectTo({ 
      url: `/pages/assessment/index?jumpTo=${idx}` 
    });
  },

  onSubmit() {
    // 跳转到“分析转场页” (截图3)
    wx.navigateTo({ url: '/pages/assessment/analysis' });
  }
});
const questions = require('../../utils/questions.js');

Page({
  data: {
    questions: questions,
    currentIndex: 0,    // 当前题目索引
    answers: {},        // 存储用户的回答
    inputText: '',      // 输入框内容（如果有输入题）
    totalCount: questions.length,
    progress: 0         // 进度百分比
  },

  onLoad(options) {
    // 如果有 jumpTo 参数，直接跳到那一题
    if (options.jumpTo) {
      this.setData({ currentIndex: parseInt(options.jumpTo) });
    }
    this.updateProgress();
  },

  // 更新进度条
  updateProgress() {
    const p = ((this.data.currentIndex + 1) / this.data.totalCount) * 100;
    this.setData({ progress: p });
  },

  // 选择选项
  onSelect(e) {
    const { field, value } = e.currentTarget.dataset;
    const { currentIndex, questions } = this.data;

    // 1. 记录答案
    this.data.answers[field] = value;
    
    // 2. 视觉反馈（可选：增加点击效果）
    this.setData({ answers: this.data.answers });

    // 3. 延时跳转下一题（让用户看到点击效果）
    setTimeout(() => {
      this.nextQuestion();
    }, 200);
  },

  // 输入框输入
  onInput(e) {
    this.setData({ inputText: e.detail.value });
  },

  // 提交输入题
  onSubmitInput() {
    if (!this.data.inputText) return wx.showToast({ title: '请输入内容', icon: 'none' });
    
    const currentQ = this.data.questions[this.data.currentIndex];
    this.data.answers[currentQ.field] = this.data.inputText;
    this.nextQuestion();
  },

  // 下一题逻辑
  nextQuestion() {
    const nextIndex = this.data.currentIndex + 1;

    // 如果是最后一题，保存并跳转
    if (nextIndex >= this.data.totalCount) {
      this.finishAssessment();
    } else {
      // 滑动到下一页
      this.setData({ currentIndex: nextIndex }, () => {
        this.updateProgress();
      });
    }
  },

  // 返回上一题
  prevQuestion() {
    if (this.data.currentIndex > 0) {
      this.setData({ currentIndex: this.data.currentIndex - 1 }, () => {
        this.updateProgress();
      });
    } else {
      // 如果是第一题，点击返回则退出
      wx.navigateBack();
    }
  },

  // 手动点击下一题（用于修改答案后快速通过）
  manualNextQuestion() {
    this.nextQuestion();
  },

  // 完成评估
  finishAssessment() {
    wx.showLoading({ title: '生成画像中...' });
    
    // 保存答案到本地缓存
    wx.setStorageSync('temp_assessment_data', this.data.answers);
    
    setTimeout(() => {
      wx.hideLoading();
      // 跳转到确认页
      wx.navigateTo({ url: '/pages/assessment/summary' });
    }, 1000);
  }
});
const app = getApp();
const config = require('../../config.js');

// ✅ 修复：改回相对路径，确保 100% 能找到图片
const DEFAULT_AVATAR = "../../images/default_avatar.png"; 

Page({
  data: {
    // 完整表单
    form: {
      nickName: "",          
      age: 28, 
      education: '本科', 
      income: 20, 
      asset: 50, 
      debt: 0, 
      children: 0, 
      exp_years: 3, 
      action_mean: 0, 
      q_text: "近期市场波动较大，我应该如何操作？"
    },
    // 页面展示用的头像
    displayAvatar: DEFAULT_AVATAR,
    
    // 下拉菜单配置
    eduOptions: ['高中及以下', '大专', '本科', '硕士及以上'], 
    eduIndex: 2,
    
    actionOptions: ['-1 (强烈卖出)', '-0.5 (减仓)', '0 (观望/持有)', '0.5 (建仓)', '1 (强烈买入)'],
    actionValues: [-1, -0.5, 0, 0.5, 1], 
    actionIndex: 2, 
    actionLabel: '0 (观望/持有)',
    
    userProfile: {}, 
    userInfo: { nickName: "" },
    
    // 控制表单显示
    showForm: false
  },

  onShow() {
    this.setData({ userProfile: app.globalData.userProfile });
    
    // 优先读取本地缓存
    const cachedForm = wx.getStorageSync('user_full_profile');
    if (cachedForm) {
      this.restoreFormState(cachedForm);
    } else {
      this.fetchUserData();
    }
  },

  // ✅ 核心：数据回显逻辑
  restoreFormState(data) {
    const eIdx = this.data.eduOptions.indexOf(data.education);
    const aIdx = this.data.actionValues.indexOf(parseFloat(data.action_mean));
    
    this.setData({ 
      // 合并表单数据
      form: { ...this.data.form, ...data },
      
      // ⚠️ 强制锁定头像为默认图 (除非你想做 Base64，但现在我们追求稳定)
      displayAvatar: DEFAULT_AVATAR,
      
      // 恢复下拉菜单
      eduIndex: eIdx !== -1 ? eIdx : 2,
      actionIndex: aIdx !== -1 ? aIdx : 2,
      actionLabel: aIdx !== -1 ? this.data.actionOptions[aIdx] : '0 (观望/持有)',
      
      // 恢复昵称
      'userInfo.nickName': data.nickName 
    });
  },

  // 昵称编辑后自动保存
  onEditNickname(e) {
    const val = e.detail.value;
    if(!val) return;
    this.setData({ 'form.nickName': val, 'userInfo.nickName': val });
    this.saveProfile(false);
  },

  // 通用输入框监听
  onInput(e) { 
    this.setData({ [`form.${e.currentTarget.dataset.field}`]: e.detail.value }); 
  },

  // 学历下拉监听
  onEduChange(e) { 
    const i = e.detail.value; 
    this.setData({ 
      eduIndex: i, 
      'form.education': this.data.eduOptions[i] 
    }); 
  },

  // 交易倾向下拉监听
  onActionChange(e) { 
    const i = e.detail.value; 
    this.setData({ 
      actionIndex: i, 
      actionLabel: this.data.actionOptions[i], 
      'form.action_mean': this.data.actionValues[i] 
    }); 
  },

  // 1. 跳转查看雷达图
  viewRadar() {
    // 跳转到 Assessment 结果页，那里有完整的雷达图
    wx.navigateTo({
      url: '/pages/assessment/result',
      success: () => {
        console.log("跳转到雷达图页面");
      },
      fail: (err) => {
        console.error("跳转失败", err);
        wx.showToast({ title: '页面跳转失败', icon: 'none' });
      }
    });
  },

  // 2. 保存后自动收起
  saveProfile() {
    const profile = this.data.form;
    wx.setStorageSync('user_full_profile', profile);
    wx.showToast({ title: '保存成功', icon: 'success' });
    this.setData({ showForm: false }); // 收起弹窗
    
    // 如果有后端同步，在这里调用
    // this.syncToBackend(profile);
  },

  // 从服务器拉取
  fetchUserData() {
    if (!app.globalData.userId) return;
    wx.request({
      url: `${config.API_BASE_URL}/api/user/get`,
      data: { user_id: app.globalData.userId },
      success: (res) => {
        if (res.data.status === 'success' && res.data.data && res.data.data.profile) {
          const p = res.data.data.profile;
          this.restoreFormState(p);
          wx.setStorageSync('user_full_profile', p);
        }
      }
    });
  },

  // 处理头像加载错误
  handleImageError() {
    this.setData({ 
      displayAvatar: DEFAULT_AVATAR 
    });
  },

  // 模拟选择头像
  chooseAvatar() {
    wx.showToast({ 
      title: '头像刷新成功', 
      icon: 'none' 
    });
    // 实际开发中调用 wx.chooseImage
  },

  // 策略按钮点击处理
  handleStrategyClick(e) {
    const type = e.currentTarget.dataset.type;

    if (type === 'conservative') {
      // 1. 想要稳健投资 -> 跳转到首页(终端)
      wx.switchTab({
        url: '/pages/index/index',
        success: () => {
          // 可以在这里加个提示，模拟筛选效果
          setTimeout(() => {
            wx.showToast({ title: '已切换至稳健策略', icon: 'success' });
          }, 500);
        }
      });
    } else if (type === 'aggressive') {
      // 2. 追求更高收益 -> 跳转到结果页(重新选策略) 或 首页
      // 既然是"追求更高收益"，通常意味着要修改配置，我们引导去结果页重选
      wx.navigateTo({
        url: '/pages/assessment/result',
        success: () => {
           wx.showToast({ title: '请选择进取型策略', icon: 'none' });
        }
      });
    }
  },

  // 开始专业评估
  startAssessment() {
    // 1. 清理之前的临时数据，确保从第一题开始
    wx.removeStorageSync('temp_assessment_data');
    
    // 2. 跳转到问卷页
    wx.navigateTo({
      url: '/pages/assessment/index',
      fail: (err) => {
        wx.showToast({ title: '跳转失败', icon: 'none' });
        console.error(err);
      }
    });
  },
  
  // 切换表单显示
  toggleForm() {
    this.setData({
      showForm: !this.data.showForm
    });
  }
});
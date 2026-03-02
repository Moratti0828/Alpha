const app = getApp();
const config = require('../../config.js');

Page({
  data: {
    isLoading: true
  },

  onLoad() {
    this.checkLoginStatus();
  },

  async checkLoginStatus() {
    // 1. 检查本地是否有画像缓存
    const localProfile = wx.getStorageSync('user_full_profile');
    
    if (localProfile) {
      // 有缓存，直接进首页
      this.goToHome();
    } else {
      // 2. 没有缓存，尝试去服务器拉取 (防止换手机后没数据)
      // 注意：这里需要延时一点点，确保 app.js 里的 login 完成
      setTimeout(() => {
        this.fetchUserProfile();
      }, 500);
    }
  },

  fetchUserProfile() {
    const userId = app.globalData.userId;
    if (!userId) {
      // 如果还没拿到 ID，停在欢迎页让用户点开始
      this.setData({ isLoading: false });
      return;
    }

    wx.request({
      url: `${config.API_BASE_URL}/api/user/get`,
      data: { user_id: userId },
      success: (res) => {
        if (res.data.status === 'success' && res.data.data) {
          // 服务器有数据 -> 存缓存 -> 进首页
          wx.setStorageSync('user_full_profile', res.data.data.profile);
          this.goToHome();
        } else {
          // 服务器也没数据 -> 新用户 -> 停在欢迎页
          this.setData({ isLoading: false });
        }
      },
      fail: () => {
        // 网络错误 -> 停在欢迎页
        this.setData({ isLoading: false });
      }
    });
  },

  onStart() {
    // 点击“开启旅程”按钮
    // 强制跳转到用户中心去填写画像
    wx.switchTab({ url: '/pages/user/index' });
  },

  goToHome() {
    wx.switchTab({ url: '/pages/index/index' });
  }
});
const config = require('./config.js'); 

App({
  globalData: {
    userId: null, // 这里将存储真实的 OpenID
    userProfile: null
  },

  onLaunch: function () {
    // 1. 尝试从缓存读取 OpenID (避免每次打开都重新登录)
    const cachedOpenId = wx.getStorageSync('real_openid');
    if (cachedOpenId) {
      this.globalData.userId = cachedOpenId;
      console.log('✅ 已读取本地 OpenID:', cachedOpenId);
    } else {
      // 2. 如果没有，发起登录
      this.doLogin();
    }
    
    this.initDeviceInfo();
  },

  doLogin: function() {
    wx.login({
      success: res => {
        if (res.code) {
          // 发送 code 到你的 Python 服务器
          wx.request({
            url: `${config.API_BASE_URL}/api/login`,
            method: 'POST',
            data: { code: res.code },
            success: (response) => {
              if (response.data.status === 'success') {
                const openid = response.data.openid;
                this.globalData.userId = openid;
                wx.setStorageSync('real_openid', openid); // 存入缓存
                console.log('🎉 登录成功, OpenID:', openid);
              } else {
                console.error('登录失败:', response.data.msg);
              }
            },
            fail: (err) => {
              console.error('请求后端失败(请检查是否开启调试模式或域名配置):', err);
            }
          })
        }
      }
    });
  },

  initDeviceInfo() {
    this.globalData.deviceInfo = wx.getWindowInfo();
  }
});
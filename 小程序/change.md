好的，这个需求非常合理。将那个“多智能体博弈”的思考过程加入到首页，会极大增加 App 的“科技感”和“专业感”。

我们需要做三件事：

1.  **JS**：增加一个控制思考步骤的计时器逻辑。
2.  **WXML**：添加显示这 4 个步骤（加密、RAG、模型、决策）的 UI 结构。
3.  **WXSS**：添加呼吸灯和渐显动画样式。

请按顺序修改以下文件。

-----

### 第一步：修改逻辑 (`pages/index/index.js`)

**改动点**：

1.  `data` 中增加 `isThinking` 和 `thinkStep`。
2.  新增 `startThinking` 函数用于播放动画。
3.  修改 `fetchAnalysis`，**去掉**原本的 `wx.showLoading`，改为启动这个动画。
4.  在数据返回后，强制 `setTimeout` 等待动画播完再显示结果。

<!-- end list -->

```javascript
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
    greeting: "Hello, Investor",
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
    this.setGreeting();
    this.initHotTags();
  },

  onShow() {
    if (app.globalData.searchSymbol) {
      const symbol = app.globalData.searchSymbol;
      app.globalData.searchSymbol = null; 
      this.fetchAnalysis(symbol); // 这里调用会触发动画
    } 
    else if (!this.data.stockData) {
      // 首次加载不自动触发动画，避免打扰，只显示数据
      // 或者你可以选择什么都不做，留白
    }
    this.fetchRecentHistory();
  },

  // ... (initHotTags, generateFutureDates, generateMockPrices 保持不变) ...

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

  // 🔴 修改：核心分析函数
  async fetchAnalysis(symbol) {
    if (!symbol) return;
    
    // 1. 启动动画 (替代 wx.showLoading)
    this.startThinking();

    const userId = app.globalData.userId || 'guest_user';
    const riskLevel = app.globalData.userProfile?.riskLevel || config.DEFAULT_RISK;

    // ... (requestData 构建保持不变) ...
    const requestData = {
      user_id: userId,
      symbol: symbol,
      user_risk_level: riskLevel,
      q_text: "分析"
    };

    wx.request({
      url: `${config.API_BASE_URL}/api/analyze`,
      method: 'POST',
      data: requestData,
      header: { 'content-type': 'application/json' },
      success: (res) => {
        // 2. 延迟显示结果，确保动画至少播完大部分 (3.5秒后)
        setTimeout(() => {
          if (res.statusCode === 200 && res.data.status === 'success') {
            this.renderData(res.data.data, symbol);
          } else {
            this.renderMockData(symbol);
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
      },
      // 移除 complete 里的 hideLoading
    });
  },

  // ... (renderMockData, renderData, initChart 等其他函数保持不变) ...

  // 确保 onQuickSearch 调用 fetchAnalysis，这样点击历史/热门都会触发动画
  onQuickSearch(e) {
    const symbol = e.currentTarget.dataset.symbol;
    this.fetchAnalysis(symbol);
  },
  
  // ... (其余代码不变)
});
```

-----

### 第二步：修改视图 (`pages/index/index.wxml`)

**改动点**：

1.  在 `dashboard` 和 `info-card` 之间插入 `thinking-box`。
2.  使用 `wx:if="{{isThinking}}"` 来控制显示。
3.  当正在思考时，隐藏 `dashboard` 和具体的股票信息。

<!-- end list -->

```html
<view class="container">
  <view class="search-box">
    <t-search placeholder="输入代码 (如 600519 / AAPL)" bind:submit="onSearch" shape="round" />
  </view>

  <view class="thinking-box" wx:if="{{isThinking}}">
    <view class="think-header">
      <view class="spinner"></view>
      <text class="think-title">AlphaMind 正在构建策略...</text>
    </view>
    
    <view class="think-steps">
      <view class="step-item {{thinkStep >= 1 ? 'active' : ''}}">
        <text class="icon">🔐</text> [安全层] 端到端加密通道建立
      </view>
      <view class="step-item {{thinkStep >= 2 ? 'active' : ''}}">
        <text class="icon">💾</text> [数据层] 检索 RAG 实时行情库
      </view>
      <view class="step-item {{thinkStep >= 3 ? 'active' : ''}}">
        <text class="icon">🧠</text> [模型层] 双塔因子匹配 (Score: 0.82)
      </view>
      <view class="step-item {{thinkStep >= 4 ? 'active' : ''}}">
        <text class="icon">⚔️</text> [决策层] 启动多智能体博弈...
      </view>
    </view>
  </view>

  <view class="dashboard" wx:if="{{!stockData && !isThinking}}">
    <view class="welcome-card card">
       </view>
    <view class="section-header">热门标的</view>
     <view class="hot-tags">
        </view>
     </view>

  <block wx:if="{{stockData && !isThinking}}">
    <view class="quick-actions">
       </view>
    <view class="card info-card">
       </view>
    </block>
</view>
```

-----

### 第三步：添加动画样式 (`pages/index/index.wxss`)

**改动点**：
添加 `thinking-box` 及其子元素的样式，实现逐行高亮和 Loading 旋转效果。

```css
/* ... 原有样式 ... */

/* === AI 思考动画样式 === */
.thinking-box {
  margin-top: 60rpx;
  padding: 40rpx;
  background: #fff;
  border-radius: 20rpx;
  box-shadow: 0 4rpx 20rpx rgba(0,0,0,0.05);
  animation: fadeIn 0.5s;
}

.think-header {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 50rpx;
}

.spinner {
  width: 36rpx;
  height: 36rpx;
  border: 4rpx solid #E6F4FF;
  border-top-color: #1677FF;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 20rpx;
}

.think-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #1677FF;
  letter-spacing: 2rpx;
}

.step-item {
  display: flex;
  align-items: center;
  font-size: 28rpx;
  color: #ccc; /* 默认灰色 */
  margin-bottom: 30rpx;
  transition: all 0.5s ease;
  transform: translateX(0);
}

.step-item.active {
  color: #333; /* 激活变黑 */
  font-weight: 500;
  transform: translateX(10rpx); /* 轻微位移 */
}

.step-item .icon {
  margin-right: 20rpx;
  font-size: 32rpx;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10rpx); }
  to { opacity: 1; transform: translateY(0); }
}
```

### 最终效果

1.  **点击热门标的/历史记录**：
      * 页面中间的仪表盘会立刻消失。
      * 出现一个白色的卡片，显示“AlphaMind 正在构建策略...”。
      * 下方会依次亮起 4 行字：
          * 🔐 [安全层] ... (0.8s)
          * 💾 [数据层] ... (1.6s)
          * 🧠 [模型层] ... (2.4s)
          * ⚔️ [决策层] ... (3.2s)
2.  **3.5 秒后**：
      * 思考卡片消失。
      * 股票详情页（价格、红绿标签、虚线折线图、AI 建议）平滑展现。

这个过程非常符合“智能投顾”的设定，让用户感觉到后台真的有复杂的运算在进行。
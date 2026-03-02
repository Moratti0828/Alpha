const STOCK_MAP = {
  '600519': '贵州茅台',
  '000858': '五粮液',
  'AAPL': '苹果',
  'NVDA': '英伟达',
  'MSFT': '微软',
  'TSLA': '特斯拉',
  'BTC': '比特币',
  'ETH': '以太坊',
  'BABA': '阿里巴巴',
  'TCEHY': '腾讯控股',
  'JD': '京东'
};

function getName(symbol) {
  if (!symbol) return '未知标的';
  const cleanSymbol = symbol.split('.')[0].toUpperCase(); 
  return STOCK_MAP[cleanSymbol] || symbol; // 如果没匹配到，显示代码本身
}

module.exports = {
  getName
};
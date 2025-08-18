const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // 只代理API请求，不代理前端路由
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      logLevel: 'debug'
    })
  );
  
  // 代理后端健康检查
  app.use(
    '/health',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      logLevel: 'debug'
    })
  );
  
  // 代理WebSocket连接
  app.use(
    '/ws',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      ws: true,
      logLevel: 'debug'
    })
  );
};
import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API配置
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 创建axios实例
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    console.log(`API请求: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API响应: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API响应错误:', error);
    
    if (error.response) {
      // 服务器返回错误状态码
      const { status, data } = error.response;
      console.error(`HTTP ${status}:`, data);
      
      switch (status) {
        case 404:
          console.error('请求的资源不存在');
          break;
        case 500:
          console.error('服务器内部错误');
          break;
        case 503:
          console.error('服务暂时不可用');
          break;
        default:
          console.error('未知错误');
      }
    } else if (error.request) {
      // 网络错误
      console.error('网络连接错误，请检查网络设置');
    } else {
      // 其他错误
      console.error('请求配置错误:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// API接口类型定义
export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
}

// 板块分析API
export const sectorApi = {
  // 获取TOP板块分析
  getTopSectors: (lookbackMonths: number = 6, topN: number = 5) =>
    api.get<ApiResponse>(`/api/sectors`, {
      params: { lookback_months: lookbackMonths, top_n: topN }
    }),
  
  // 获取板块列表
  getSectorList: () =>
    api.get<ApiResponse>('/api/sectors/list'),
  
  // 获取单个板块分析
  getSectorAnalysis: (sectorName: string, analysisDays: number = 30) =>
    api.get<ApiResponse>(`/api/sectors/${sectorName}`, {
      params: { analysis_days: analysisDays }
    }),
  
  // 板块性能验证
  validatePerformance: () =>
    api.get<ApiResponse>('/api/sectors/performance/validation'),
};

// 股票分析API
export const stockApi = {
  // 智能股票筛选
  selectStocks: (sectors: string[], stocksPerSector: number = 5) =>
    api.post<ApiResponse>('/api/stocks/select', {
      sectors,
      stocks_per_sector: stocksPerSector
    }),
  
  // 单股票分析
  analyzeStock: (stockCode: string, analysisDays: number = 30) =>
    api.get<ApiResponse>(`/api/stocks/${stockCode}`, {
      params: { analysis_days: analysisDays }
    }),
  
  // 获取股票列表
  getStockList: (sector?: string, minScore?: number) =>
    api.get<ApiResponse>('/api/stocks/', {
      params: { sector, min_score: minScore }
    }),
  
  // 批量股票分析
  bulkAnalyze: (stockCodes: string[], analysisType: string = 'basic') =>
    api.post<ApiResponse>('/api/stocks/bulk-analyze', null, {
      params: { stock_codes: stockCodes, analysis_type: analysisType }
    }),
  
  // 股票性能验证
  validatePerformance: () =>
    api.get<ApiResponse>('/api/stocks/performance/validation'),
};

// 投资组合API
export const portfolioApi = {
  // 投资组合分析
  analyzePortfolio: (holdings: any[]) =>
    api.post<ApiResponse>('/api/portfolio/analyze', { holdings }),
  
  // 获取预设组合
  getPresetPortfolio: () =>
    api.get<ApiResponse>('/api/portfolio/preset'),
  
  // 组合优化
  optimizePortfolio: (holdings: any[], optimizationTarget: string = 'risk_return') =>
    api.post<ApiResponse>('/api/portfolio/optimize', holdings, {
      params: { optimization_target: optimizationTarget }
    }),
  
  // 风险评估
  assessRisk: (portfolioValue: number, riskTolerance: string = 'medium') =>
    api.get<ApiResponse>('/api/portfolio/risk-assessment', {
      params: { portfolio_value: portfolioValue, risk_tolerance: riskTolerance }
    }),
  
  // 组合回测
  backtest: (startDate: string, endDate: string, rebalanceFrequency: string = 'monthly') =>
    api.get<ApiResponse>('/api/portfolio/performance/backtest', {
      params: { 
        start_date: startDate, 
        end_date: endDate, 
        rebalance_frequency: rebalanceFrequency 
      }
    }),
};

// 交易助手API
export const tradingApi = {
  // 交易决策检查
  checkTradingDecision: (request: any) =>
    api.post<ApiResponse>('/api/trading/check', request),
  
  // 情绪控制建议
  getEmotionControl: (request: any) =>
    api.post<ApiResponse>('/api/trading/emotion-control', request),
  
  // 获取交易规则
  getTradingRules: () =>
    api.get<ApiResponse>('/api/trading/rules'),
  
  // 更新交易规则
  updateTradingRules: (rules: any) =>
    api.put<ApiResponse>('/api/trading/rules', null, { params: rules }),
  
  // 获取交易历史
  getTradingHistory: (days: number = 30, analysisType: string = 'summary') =>
    api.get<ApiResponse>('/api/trading/history', {
      params: { days, analysis_type: analysisType }
    }),
};

// 系统API
export const systemApi = {
  // 健康检查
  healthCheck: () =>
    api.get<ApiResponse>('/health'),
  
  // 系统信息
  getSystemInfo: () =>
    api.get<ApiResponse>('/api/system/info'),
  
  // 系统状态
  getSystemStatus: () =>
    api.get<ApiResponse>('/api/system/status'),
};

export default api;
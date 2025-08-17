import React, { createContext, useContext, useState, ReactNode } from 'react';
import { message } from 'antd';
import { sectorApi, stockApi, portfolioApi, tradingApi, systemApi } from '../services/api';

interface ApiContextType {
  loading: boolean;
  error: string | null;
  
  // 板块分析相关
  getTopSectors: (lookbackMonths?: number, topN?: number) => Promise<any>;
  getSectorList: () => Promise<any>;
  getSectorAnalysis: (sectorName: string, analysisDays?: number) => Promise<any>;
  validateSectorPerformance: () => Promise<any>;
  
  // 股票分析相关
  selectStocks: (sectors: string[], stocksPerSector?: number) => Promise<any>;
  analyzeStock: (stockCode: string, analysisDays?: number) => Promise<any>;
  getStockList: (sector?: string, minScore?: number) => Promise<any>;
  bulkAnalyzeStocks: (stockCodes: string[], analysisType?: string) => Promise<any>;
  validateStockPerformance: () => Promise<any>;
  
  // 投资组合相关
  analyzePortfolio: (holdings: any[]) => Promise<any>;
  getPresetPortfolio: () => Promise<any>;
  optimizePortfolio: (holdings: any[], optimizationTarget?: string) => Promise<any>;
  assessPortfolioRisk: (portfolioValue: number, riskTolerance?: string) => Promise<any>;
  backtestPortfolio: (startDate: string, endDate: string, rebalanceFrequency?: string) => Promise<any>;
  
  // 交易助手相关
  checkTradingDecision: (request: any) => Promise<any>;
  getEmotionControl: (request: any) => Promise<any>;
  getTradingRules: () => Promise<any>;
  updateTradingRules: (rules: any) => Promise<any>;
  getTradingHistory: (days?: number, analysisType?: string) => Promise<any>;
  
  // 系统相关
  healthCheck: () => Promise<any>;
  getSystemInfo: () => Promise<any>;
  getSystemStatus: () => Promise<any>;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

interface ApiProviderProps {
  children: ReactNode;
}

export const ApiProvider: React.FC<ApiProviderProps> = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 通用API调用包装器
  const apiCall = async <T,>(apiFunction: () => Promise<any>, errorMessage?: string): Promise<T | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiFunction();
      
      if (response.data?.success) {
        return response.data.data || response.data;
      } else {
        const errorMsg = response.data?.message || errorMessage || 'API调用失败';
        setError(errorMsg);
        message.error(errorMsg);
        return null;
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.message || err.message || errorMessage || 'API调用失败';
      setError(errorMsg);
      message.error(errorMsg);
      console.error('API调用错误:', err);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // 板块分析API方法
  const getTopSectors = (lookbackMonths = 6, topN = 5) => 
    apiCall(() => sectorApi.getTopSectors(lookbackMonths, topN), '获取板块分析失败');

  const getSectorList = () => 
    apiCall(() => sectorApi.getSectorList(), '获取板块列表失败');

  const getSectorAnalysis = (sectorName: string, analysisDays = 30) => 
    apiCall(() => sectorApi.getSectorAnalysis(sectorName, analysisDays), '获取板块详细分析失败');

  const validateSectorPerformance = () => 
    apiCall(() => sectorApi.validatePerformance(), '板块性能验证失败');

  // 股票分析API方法
  const selectStocks = (sectors: string[], stocksPerSector = 5) => 
    apiCall(() => stockApi.selectStocks(sectors, stocksPerSector), '股票筛选失败');

  const analyzeStock = (stockCode: string, analysisDays = 30) => 
    apiCall(() => stockApi.analyzeStock(stockCode, analysisDays), '股票分析失败');

  const getStockList = (sector?: string, minScore?: number) => 
    apiCall(() => stockApi.getStockList(sector, minScore), '获取股票列表失败');

  const bulkAnalyzeStocks = (stockCodes: string[], analysisType = 'basic') => 
    apiCall(() => stockApi.bulkAnalyze(stockCodes, analysisType), '批量股票分析失败');

  const validateStockPerformance = () => 
    apiCall(() => stockApi.validatePerformance(), '股票性能验证失败');

  // 投资组合API方法
  const analyzePortfolio = (holdings: any[]) => 
    apiCall(() => portfolioApi.analyzePortfolio(holdings), '投资组合分析失败');

  const getPresetPortfolio = () => 
    apiCall(() => portfolioApi.getPresetPortfolio(), '获取预设组合失败');

  const optimizePortfolio = (holdings: any[], optimizationTarget = 'risk_return') => 
    apiCall(() => portfolioApi.optimizePortfolio(holdings, optimizationTarget), '投资组合优化失败');

  const assessPortfolioRisk = (portfolioValue: number, riskTolerance = 'medium') => 
    apiCall(() => portfolioApi.assessRisk(portfolioValue, riskTolerance), '风险评估失败');

  const backtestPortfolio = (startDate: string, endDate: string, rebalanceFrequency = 'monthly') => 
    apiCall(() => portfolioApi.backtest(startDate, endDate, rebalanceFrequency), '组合回测失败');

  // 交易助手API方法
  const checkTradingDecision = (request: any) => 
    apiCall(() => tradingApi.checkTradingDecision(request), '交易决策检查失败');

  const getEmotionControl = (request: any) => 
    apiCall(() => tradingApi.getEmotionControl(request), '获取情绪控制建议失败');

  const getTradingRules = () => 
    apiCall(() => tradingApi.getTradingRules(), '获取交易规则失败');

  const updateTradingRules = (rules: any) => 
    apiCall(() => tradingApi.updateTradingRules(rules), '更新交易规则失败');

  const getTradingHistory = (days = 30, analysisType = 'summary') => 
    apiCall(() => tradingApi.getTradingHistory(days, analysisType), '获取交易历史失败');

  // 系统API方法
  const healthCheck = () => 
    apiCall(() => systemApi.healthCheck(), '系统健康检查失败');

  const getSystemInfo = () => 
    apiCall(() => systemApi.getSystemInfo(), '获取系统信息失败');

  const getSystemStatus = () => 
    apiCall(() => systemApi.getSystemStatus(), '获取系统状态失败');

  const value: ApiContextType = {
    loading,
    error,
    
    // 板块分析
    getTopSectors,
    getSectorList,
    getSectorAnalysis,
    validateSectorPerformance,
    
    // 股票分析
    selectStocks,
    analyzeStock,
    getStockList,
    bulkAnalyzeStocks,
    validateStockPerformance,
    
    // 投资组合
    analyzePortfolio,
    getPresetPortfolio,
    optimizePortfolio,
    assessPortfolioRisk,
    backtestPortfolio,
    
    // 交易助手
    checkTradingDecision,
    getEmotionControl,
    getTradingRules,
    updateTradingRules,
    getTradingHistory,
    
    // 系统
    healthCheck,
    getSystemInfo,
    getSystemStatus,
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};

export const useApi = (): ApiContextType => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export default ApiContext;
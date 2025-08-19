import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { message } from 'antd';
import { 
  dataService, 
  MarketData, 
  SectorData, 
  PortfolioData, 
  SystemStatus 
} from '../services/dataService';

interface DataContextType {
  isConnected: boolean;
  connectionState: string;
  marketData: MarketData | null;
  sectorData: SectorData | null;
  portfolioData: PortfolioData | null;
  systemStatus: SystemStatus | null;
  refreshData: () => Promise<void>;
  startPolling: (interval?: number) => void;
  stopPolling: () => void;
  lastUpdated: string | null;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

interface DataProviderProps {
  children: ReactNode;
}

export const DataProvider: React.FC<DataProviderProps> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState('disconnected');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [sectorData, setSectorData] = useState<SectorData | null>(null);
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  useEffect(() => {
    // 注册数据监听器
    dataService.subscribe('market_data', (data: MarketData) => {
      setMarketData(data);
      setLastUpdated(new Date().toLocaleString());
      console.log('市场数据已更新:', data);
    });

    dataService.subscribe('sector_updates', (data: SectorData) => {
      setSectorData(data);
      setLastUpdated(new Date().toLocaleString());
      console.log('板块数据已更新:', data);
    });

    dataService.subscribe('portfolio_alerts', (data: PortfolioData) => {
      setPortfolioData(data);
      setLastUpdated(new Date().toLocaleString());
      console.log('投资组合数据已更新:', data);
    });

    dataService.subscribe('system_status', (data: SystemStatus) => {
      setSystemStatus(data);
      setLastUpdated(new Date().toLocaleString());
      console.log('系统状态已更新:', data);
    });

    dataService.subscribe('error', (error: any) => {
      console.error('数据查询错误:', error);
      message.error('数据查询失败，请检查网络连接');
      setIsConnected(false);
      setConnectionState('error');
    });

    // 启动数据定时查询
    startPolling();

    // 设置状态监控
    const statusInterval = setInterval(() => {
      const polling = dataService.isPolling;
      const state = dataService.connectionState;
      
      setIsConnected(polling);
      setConnectionState(state);
    }, 1000);

    // 清理函数
    return () => {
      clearInterval(statusInterval);
      dataService.stopPolling();
      dataService.unsubscribe('market_data');
      dataService.unsubscribe('sector_updates');
      dataService.unsubscribe('portfolio_alerts');
      dataService.unsubscribe('system_status');
      dataService.unsubscribe('error');
    };
  }, []);

  const startPolling = (interval?: number): void => {
    try {
      setConnectionState('connecting');
      dataService.startPolling(interval);
      setIsConnected(true);
      setConnectionState('connected');
      message.success('已启动数据实时更新');
    } catch (error) {
      setConnectionState('error');
      message.error('启动数据更新失败');
    }
  };

  const stopPolling = (): void => {
    dataService.stopPolling();
    setIsConnected(false);
    setConnectionState('disconnected');
    message.info('已停止数据实时更新');
  };

  const refreshData = async (): Promise<void> => {
    try {
      setConnectionState('connecting');
      await dataService.refreshData();
      setConnectionState('connected');
      message.success('数据刷新成功');
    } catch (error) {
      setConnectionState('error');
      message.error('数据刷新失败');
      throw error;
    }
  };

  const value: DataContextType = {
    isConnected,
    connectionState,
    marketData,
    sectorData,
    portfolioData,
    systemStatus,
    refreshData,
    startPolling,
    stopPolling,
    lastUpdated,
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = (): DataContextType => {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};

export default DataContext;
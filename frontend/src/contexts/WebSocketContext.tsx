import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { message } from 'antd';
import { 
  webSocketService, 
  SubscriptionType, 
  MarketData, 
  SectorUpdate, 
  PortfolioAlert, 
  TradingSignal, 
  SystemStatus 
} from '../services/websocket';

interface WebSocketContextType {
  isConnected: boolean;
  connectionState: string;
  marketData: MarketData | null;
  sectorUpdates: SectorUpdate | null;
  portfolioAlerts: PortfolioAlert | null;
  tradingSignals: TradingSignal | null;
  systemStatus: SystemStatus | null;
  subscribe: (type: SubscriptionType) => void;
  unsubscribe: (type: SubscriptionType) => void;
  connect: () => Promise<void>;
  disconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState('disconnected');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [sectorUpdates, setSectorUpdates] = useState<SectorUpdate | null>(null);
  const [portfolioAlerts, setPortfolioAlerts] = useState<PortfolioAlert | null>(null);
  const [tradingSignals, setTradingSignals] = useState<TradingSignal | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    // 设置连接事件处理器
    webSocketService.onConnect(() => {
      setIsConnected(true);
      setConnectionState('connected');
      message.success('实时数据连接已建立');
    });

    webSocketService.onDisconnect(() => {
      setIsConnected(false);
      setConnectionState('disconnected');
      message.warning('实时数据连接已断开，正在尝试重连...');
    });

    webSocketService.onError((error) => {
      setIsConnected(false);
      setConnectionState('error');
      message.error('实时数据连接失败');
      console.error('WebSocket错误:', error);
    });

    // 更新连接状态
    const updateConnectionState = () => {
      setConnectionState(webSocketService.connectionState);
      setIsConnected(webSocketService.isConnected);
    };

    const stateUpdateInterval = setInterval(updateConnectionState, 1000);

    // 清理函数
    return () => {
      clearInterval(stateUpdateInterval);
      webSocketService.disconnect();
    };
  }, []);

  const connect = async (): Promise<void> => {
    try {
      setConnectionState('connecting');
      await webSocketService.connect();
    } catch (error) {
      setConnectionState('error');
      throw error;
    }
  };

  const disconnect = (): void => {
    webSocketService.disconnect();
    setIsConnected(false);
    setConnectionState('disconnected');
  };

  const subscribe = (type: SubscriptionType): void => {
    switch (type) {
      case 'market_data':
        webSocketService.subscribe(type, (data: MarketData) => {
          setMarketData(data);
        });
        break;
      
      case 'sector_updates':
        webSocketService.subscribe(type, (data: SectorUpdate) => {
          setSectorUpdates(data);
        });
        break;
      
      case 'portfolio_alerts':
        webSocketService.subscribe(type, (data: PortfolioAlert) => {
          setPortfolioAlerts(data);
          
          // 显示重要警告通知
          if (data.alerts.length > 0) {
            data.alerts.forEach(alert => {
              if (alert.severity === 'warning') {
                message.warning(alert.message);
              } else if (alert.severity === 'error') {
                message.error(alert.message);
              } else {
                message.info(alert.message);
              }
            });
          }
        });
        break;
      
      case 'trading_signals':
        webSocketService.subscribe(type, (data: TradingSignal) => {
          setTradingSignals(data);
          
          // 显示高置信度交易信号
          if (data.signals.length > 0) {
            data.signals
              .filter(signal => signal.confidence > 0.7)
              .forEach(signal => {
                message.info(`交易信号: ${signal.message}`);
              });
          }
        });
        break;
      
      case 'system_status':
        webSocketService.subscribe(type, (data: SystemStatus) => {
          setSystemStatus(data);
        });
        break;
    }
  };

  const unsubscribe = (type: SubscriptionType): void => {
    webSocketService.unsubscribe(type);
    
    // 清除对应的状态
    switch (type) {
      case 'market_data':
        setMarketData(null);
        break;
      case 'sector_updates':
        setSectorUpdates(null);
        break;
      case 'portfolio_alerts':
        setPortfolioAlerts(null);
        break;
      case 'trading_signals':
        setTradingSignals(null);
        break;
      case 'system_status':
        setSystemStatus(null);
        break;
    }
  };

  const value: WebSocketContextType = {
    isConnected,
    connectionState,
    marketData,
    sectorUpdates,
    portfolioAlerts,
    tradingSignals,
    systemStatus,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export default WebSocketContext;
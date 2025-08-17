import { io, Socket } from 'socket.io-client';

// WebSocket配置
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export type SubscriptionType = 
  | 'market_data' 
  | 'sector_updates' 
  | 'portfolio_alerts' 
  | 'trading_signals' 
  | 'system_status';

export interface WebSocketMessage {
  type: string;
  subscription?: SubscriptionType;
  data?: any;
  timestamp: string;
  message?: string;
}

export interface MarketData {
  timestamp: string;
  market_indices: {
    sh_composite: {
      value: number;
      change: number;
      change_pct: number;
    };
    sz_component: {
      value: number;
      change: number;
      change_pct: number;
    };
  };
  hot_stocks: Array<{
    code: string;
    name: string;
    price: number;
    change_pct: number;
  }>;
}

export interface SectorUpdate {
  timestamp: string;
  top_sectors: Array<{
    name: string;
    score: number;
    change: number;
    trend: string;
  }>;
  market_sentiment: string;
  optimization_status: string;
}

export interface PortfolioAlert {
  timestamp: string;
  alerts: Array<{
    type: string;
    message: string;
    severity: string;
    stock_code?: string;
  }>;
  portfolio_performance: {
    total_return: number;
    best_performer: string;
    worst_performer: string;
  };
}

export interface TradingSignal {
  timestamp: string;
  signals: Array<{
    type: string;
    message: string;
    confidence: number;
    time_horizon?: string;
    advice?: string;
  }>;
  market_emotion: {
    fear_greed_index: number;
    trend: string;
    recommendation: string;
  };
}

export interface SystemStatus {
  timestamp: string;
  services: {
    [key: string]: string;
  };
  performance_metrics: {
    sector_accuracy: string;
    stock_win_rate: string;
    portfolio_return: string;
    system_uptime: string;
  };
  active_connections: number;
  optimization_status: {
    last_optimization: string;
    improvement: string;
    status: string;
  };
}

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  
  // 事件回调
  private messageHandlers: Map<string, (data: any) => void> = new Map();
  private connectionHandlers: Array<() => void> = [];
  private disconnectionHandlers: Array<() => void> = [];
  private errorHandlers: Array<(error: any) => void> = [];

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // 创建WebSocket连接 (使用原生WebSocket而不是socket.io)
        this.socket = new WebSocket(`${WS_URL}/ws/realtime`) as any;
        
        this.socket.onopen = () => {
          console.log('WebSocket连接已建立');
          this.reconnectAttempts = 0;
          this.connectionHandlers.forEach(handler => handler());
          resolve();
        };

        this.socket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('WebSocket消息解析失败:', error);
          }
        };

        this.socket.onclose = () => {
          console.log('WebSocket连接已断开');
          this.disconnectionHandlers.forEach(handler => handler());
          this.attemptReconnect();
        };

        this.socket.onerror = (error) => {
          console.error('WebSocket连接错误:', error);
          this.errorHandlers.forEach(handler => handler(error));
          reject(error);
        };

      } catch (error) {
        console.error('WebSocket连接失败:', error);
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`WebSocket重连尝试 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('WebSocket重连失败:', error);
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('WebSocket重连次数已达上限');
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('收到WebSocket消息:', message);
    
    // 处理不同类型的消息
    switch (message.type) {
      case 'welcome':
        console.log('WebSocket欢迎消息:', message.message);
        break;
      
      case 'subscription_confirmed':
        console.log(`订阅确认: ${message.subscription}`);
        break;
      
      case 'data_update':
        if (message.subscription && message.data) {
          const handler = this.messageHandlers.get(message.subscription);
          if (handler) {
            handler(message.data);
          }
        }
        break;
      
      case 'error':
        console.error('WebSocket错误:', message.message);
        break;
      
      default:
        console.log('未知消息类型:', message.type);
    }
  }

  // 订阅数据流
  subscribe(subscriptionType: SubscriptionType, handler: (data: any) => void): void {
    this.messageHandlers.set(subscriptionType, handler);
    
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'subscribe',
        subscription: subscriptionType
      }));
    }
  }

  // 取消订阅
  unsubscribe(subscriptionType: SubscriptionType): void {
    this.messageHandlers.delete(subscriptionType);
    
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'unsubscribe',
        subscription: subscriptionType
      }));
    }
  }

  // 事件监听器
  onConnect(handler: () => void): void {
    this.connectionHandlers.push(handler);
  }

  onDisconnect(handler: () => void): void {
    this.disconnectionHandlers.push(handler);
  }

  onError(handler: (error: any) => void): void {
    this.errorHandlers.push(handler);
  }

  // 检查连接状态
  get isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  // 获取连接状态
  get connectionState(): string {
    if (!this.socket) return 'disconnected';
    
    switch (this.socket.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }
}

// 导出单例实例
export const webSocketService = new WebSocketService();
export default webSocketService;
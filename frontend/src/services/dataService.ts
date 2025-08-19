// 数据查询服务 - 替代WebSocket的定时查询方式
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

export interface SectorData {
  timestamp: string;
  top_sectors: Array<{
    name: string;
    score: number;
    change: number;
    trend: string;
  }>;
  market_sentiment: string;
}

export interface PortfolioData {
  timestamp: string;
  holdings: Array<{
    code: string;
    name: string;
    position: number;
    profit: string;
    note: string;
  }>;
  total_value: number;
  total_profit: string;
  market_summary: string;
}

export interface SystemStatus {
  status: string;
  service: string;
  data_policy: string;
  market_status: string;
}

class DataService {
  private updateInterval: NodeJS.Timeout | null = null;
  private listeners: Map<string, (data: any) => void> = new Map();
  private isRunning = false;
  private readonly DEFAULT_INTERVAL = 5000; // 5秒更新一次

  // 启动定时查询
  startPolling(interval: number = this.DEFAULT_INTERVAL): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('启动数据定时查询，间隔:', interval, 'ms');
    
    // 立即查询一次
    this.fetchAllData();
    
    // 设置定时器
    this.updateInterval = setInterval(() => {
      this.fetchAllData();
    }, interval);
  }

  // 停止定时查询
  stopPolling(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.isRunning = false;
    console.log('停止数据定时查询');
  }

  // 获取所有数据
  private async fetchAllData(): Promise<void> {
    try {
      // 并行获取所有数据
      const [marketData, sectorData, portfolioData, systemStatus] = await Promise.all([
        this.fetchMarketData(),
        this.fetchSectorData(), 
        this.fetchPortfolioData(),
        this.fetchSystemStatus()
      ]);

      // 通知所有监听器
      this.notifyListeners('market_data', marketData);
      this.notifyListeners('sector_updates', sectorData);
      this.notifyListeners('portfolio_alerts', portfolioData);
      this.notifyListeners('system_status', systemStatus);

    } catch (error) {
      console.error('数据查询失败:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.notifyListeners('error', { error: errorMessage, timestamp: new Date().toISOString() });
    }
  }

  // 获取市场数据
  private async fetchMarketData(): Promise<MarketData> {
    const response = await fetch(`${API_BASE_URL}/api/indices`);
    if (!response.ok) {
      throw new Error(`市场数据查询失败: ${response.status}`);
    }
    const indices = await response.json();
    
    // 转换为MarketData格式
    return {
      timestamp: indices.updated_at,
      market_indices: {
        sh_composite: {
          value: indices.indices?.['上证综指']?.current || 0,
          change: indices.indices?.['上证综指']?.change || 0,
          change_pct: indices.indices?.['上证综指']?.change_pct || 0
        },
        sz_component: {
          value: indices.indices?.['深证成指']?.current || 0,
          change: indices.indices?.['深证成指']?.change || 0,
          change_pct: indices.indices?.['深证成指']?.change_pct || 0
        }
      },
      hot_stocks: [] // 此API暂不提供热门股票数据
    };
  }

  // 获取板块数据
  private async fetchSectorData(): Promise<SectorData> {
    const response = await fetch(`${API_BASE_URL}/api/sectors`);
    if (!response.ok) {
      throw new Error(`板块数据查询失败: ${response.status}`);
    }
    const sectors = await response.json();
    
    return {
      timestamp: sectors.updated_at,
      top_sectors: sectors.sectors.map((sector: any) => ({
        name: sector.name,
        score: sector.score,
        change: sector.change_pct,
        trend: sector.change_pct > 0 ? 'up' : 'down'
      })),
      market_sentiment: sectors.market_summary || '积极'
    };
  }

  // 获取投资组合数据
  private async fetchPortfolioData(): Promise<PortfolioData> {
    const response = await fetch(`${API_BASE_URL}/api/portfolio`);
    if (!response.ok) {
      throw new Error(`投资组合数据查询失败: ${response.status}`);
    }
    return await response.json();
  }

  // 获取系统状态
  private async fetchSystemStatus(): Promise<SystemStatus> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`系统状态查询失败: ${response.status}`);
    }
    return await response.json();
  }

  // 注册数据监听器
  subscribe(dataType: string, callback: (data: any) => void): void {
    this.listeners.set(dataType, callback);
    console.log(`注册监听器: ${dataType}`);
  }

  // 取消监听
  unsubscribe(dataType: string): void {
    this.listeners.delete(dataType);
    console.log(`取消监听器: ${dataType}`);
  }

  // 通知监听器
  private notifyListeners(dataType: string, data: any): void {
    const listener = this.listeners.get(dataType);
    if (listener) {
      listener(data);
    }
  }

  // 手动刷新数据
  async refreshData(): Promise<void> {
    console.log('手动刷新数据');
    await this.fetchAllData();
  }

  // 检查服务状态
  get isPolling(): boolean {
    return this.isRunning;
  }

  // 获取连接状态字符串
  get connectionState(): string {
    return this.isRunning ? 'connected' : 'disconnected';
  }

  // 单次获取特定数据的方法
  async getMarketData(): Promise<MarketData> {
    return this.fetchMarketData();
  }

  async getSectorData(): Promise<SectorData> {
    return this.fetchSectorData();
  }

  async getPortfolioData(): Promise<PortfolioData> {
    return this.fetchPortfolioData();
  }

  async getSystemStatus(): Promise<SystemStatus> {
    return this.fetchSystemStatus();
  }
}

// 导出单例实例
export const dataService = new DataService();
export default dataService;
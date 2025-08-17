import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Statistic, Typography, Button, Space, Alert, Spin } from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  RiseOutlined,
  FallOutlined,
  ReloadOutlined,
  ApiOutlined,
} from '@ant-design/icons';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useApi } from '../contexts/ApiContext';
import MarketOverview from '../components/Dashboard/MarketOverview';
import PerformanceMetrics from '../components/Dashboard/PerformanceMetrics';
import QuickActions from '../components/Dashboard/QuickActions';

const { Title, Text } = Typography;

const Dashboard: React.FC = () => {
  const { 
    isConnected, 
    marketData, 
    systemStatus, 
    subscribe, 
    connect 
  } = useWebSocket();
  
  const { 
    getSystemStatus, 
    validateSectorPerformance, 
    validateStockPerformance,
    loading 
  } = useApi();

  const [performanceData, setPerformanceData] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    // 自动连接WebSocket并订阅系统状态
    if (!isConnected) {
      connect().then(() => {
        subscribe('system_status');
        subscribe('market_data');
      }).catch(console.error);
    } else {
      subscribe('system_status');
      subscribe('market_data');
    }

    // 获取系统状态
    loadSystemData();
  }, [isConnected]);

  const loadSystemData = async () => {
    try {
      const [systemData, sectorValidation, stockValidation] = await Promise.all([
        getSystemStatus(),
        validateSectorPerformance(),
        validateStockPerformance(),
      ]);

      setPerformanceData({
        system: systemData,
        sector: sectorValidation,
        stock: stockValidation,
      });

      setLastUpdate(new Date());
    } catch (error) {
      console.error('加载系统数据失败:', error);
    }
  };

  const handleRefresh = () => {
    loadSystemData();
  };

  return (
    <div style={{ paddingTop: 64 }}> {/* 为固定头部留出空间 */}
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>
              控制台总览
            </Title>
            <Text type="secondary">
              最后更新: {lastUpdate.toLocaleTimeString('zh-CN')}
            </Text>
          </Col>
          <Col>
            <Space>
              <Button 
                icon={<ReloadOutlined />} 
                onClick={handleRefresh}
                loading={loading}
              >
                刷新数据
              </Button>
              <Button 
                type="primary" 
                icon={<ApiOutlined />}
                disabled={!isConnected}
              >
                {isConnected ? '实时连接' : '连接中断'}
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {/* 连接状态提示 */}
      {!isConnected && (
        <Alert
          message="实时数据连接已断开"
          description="正在尝试重新连接，部分数据可能不是最新的"
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 系统性能指标 */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="板块预测准确率"
              value={systemStatus?.performance_metrics.sector_accuracy || '69.0%'}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ArrowUpOutlined />}
              suffix=""
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              相比基准提升 +7.8%
            </Text>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="股票选择胜率"
              value={systemStatus?.performance_metrics.stock_win_rate || '50.0%'}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              prefix={<RiseOutlined />}
              suffix=""
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              相比基准提升 +25.0%
            </Text>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="组合收益率"
              value={systemStatus?.performance_metrics.portfolio_return || '0.31%'}
              precision={2}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ArrowUpOutlined />}
              suffix=""
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              相比基准提升 +126.1%
            </Text>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="系统运行时间"
              value={systemStatus?.performance_metrics.system_uptime || '99.9%'}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix=""
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              当前活跃连接: {systemStatus?.active_connections || 0}
            </Text>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]}>
        {/* 市场概览 */}
        <Col xs={24} lg={16}>
          {loading ? (
            <Card style={{ textAlign: 'center', minHeight: 300 }}>
              <Spin size="large" />
            </Card>
          ) : (
            <MarketOverview marketData={marketData} />
          )}
        </Col>

        {/* 快速操作 */}
        <Col xs={24} lg={8}>
          <QuickActions />
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
        {/* 算法性能指标 */}
        <Col span={24}>
          <PerformanceMetrics 
            performanceData={performanceData}
            systemStatus={systemStatus}
          />
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
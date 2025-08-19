import React from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Typography } from 'antd';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined,
  StockOutlined 
} from '@ant-design/icons';
import { MarketData } from '../../services/dataService';

const { Title, Text } = Typography;

interface MarketOverviewProps {
  marketData: MarketData | null;
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ marketData }) => {
  // 备用数据，仅在WebSocket连接失败时使用
  const fallbackMarketData: MarketData = {
    timestamp: new Date().toISOString(),
    market_indices: {
      sh_composite: {
        value: 3700.0,  // 更新为合理的当前水平
        change: 0.0,
        change_pct: 0.0
      },
      sz_component: {
        value: 11200.0,  // 更新为合理的当前水平
        change: 0.0,
        change_pct: 0.0
      }
    },
    hot_stocks: [
      { code: '000001', name: '平安银行', price: 12.08, change_pct: 0.0 },
      { code: '600519', name: '贵州茅台', price: 1680.50, change_pct: 0.0 },
      { code: '300750', name: '宁德时代', price: 185.20, change_pct: 0.0 },
      { code: '000858', name: '五粮液', price: 158.30, change_pct: 0.0 },
      { code: '002415', name: '海康威视', price: 35.45, change_pct: 0.0 }
    ]
  };

  const data = marketData || fallbackMarketData;

  const hotStockColumns = [
    {
      title: '股票代码',
      dataIndex: 'code',
      key: 'code',
      render: (code: string) => (
        <Text strong>{code}</Text>
      ),
    },
    {
      title: '股票名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '现价',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '涨跌幅',
      dataIndex: 'change_pct',
      key: 'change_pct',
      render: (changePct: number) => (
        <Tag 
          color={changePct >= 0 ? 'green' : 'red'}
          icon={changePct >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
        >
          {changePct >= 0 ? '+' : ''}{changePct.toFixed(1)}%
        </Tag>
      ),
    },
  ];

  return (
    <Card 
      title={
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <StockOutlined style={{ marginRight: 8 }} />
          市场概览
          <Text type="secondary" style={{ marginLeft: 16, fontSize: 14 }}>
            {new Date(data.timestamp).toLocaleTimeString('zh-CN')} 更新
          </Text>
        </div>
      }
    >
      {/* 主要指数 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12}>
          <Card size="small" style={{ background: '#f8f9fa' }}>
            <Statistic
              title="上证综指"
              value={data.market_indices.sh_composite.value}
              precision={2}
              valueStyle={{ 
                color: data.market_indices.sh_composite.change >= 0 ? '#3f8600' : '#cf1322' 
              }}
              prefix={
                data.market_indices.sh_composite.change >= 0 ? 
                <ArrowUpOutlined /> : <ArrowDownOutlined />
              }
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">
                {data.market_indices.sh_composite.change >= 0 ? '+' : ''}
                {data.market_indices.sh_composite.change.toFixed(2)} 
                ({data.market_indices.sh_composite.change_pct >= 0 ? '+' : ''}
                {data.market_indices.sh_composite.change_pct.toFixed(2)}%)
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12}>
          <Card size="small" style={{ background: '#f8f9fa' }}>
            <Statistic
              title="深证成指"
              value={data.market_indices.sz_component.value}
              precision={2}
              valueStyle={{ 
                color: data.market_indices.sz_component.change >= 0 ? '#3f8600' : '#cf1322' 
              }}
              prefix={
                data.market_indices.sz_component.change >= 0 ? 
                <ArrowUpOutlined /> : <ArrowDownOutlined />
              }
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">
                {data.market_indices.sz_component.change >= 0 ? '+' : ''}
                {data.market_indices.sz_component.change.toFixed(2)} 
                ({data.market_indices.sz_component.change_pct >= 0 ? '+' : ''}
                {data.market_indices.sz_component.change_pct.toFixed(2)}%)
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 热门股票 */}
      <div>
        <Title level={5} style={{ marginBottom: 16 }}>
          热门股票
        </Title>
        <Table
          columns={hotStockColumns}
          dataSource={data.hot_stocks.map((stock, index) => ({
            ...stock,
            key: index,
          }))}
          pagination={false}
          size="small"
        />
      </div>
    </Card>
  );
};

export default MarketOverview;
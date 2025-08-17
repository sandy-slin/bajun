import React, { useEffect, useState } from 'react';
import { Card, Typography, Button, Space, Alert, Spin, Empty, Collapse, Tag, Divider, Statistic, Row, Col } from 'antd';
import { BarChartOutlined, ReloadOutlined, StockOutlined, TrophyOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { useApi } from '../contexts/ApiContext';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

const SectorAnalysisEnhanced: React.FC = () => {
  const { getTopSectors, loading, error } = useApi();
  const [sectorData, setSectorData] = useState<any>(null);
  const [showAllSectors, setShowAllSectors] = useState<boolean>(false);

  useEffect(() => {
    loadSectorData();
  }, []);

  const loadSectorData = async () => {
    const data = await getTopSectors(6, 5);
    setSectorData(data);
  };

  const handleRefresh = () => {
    loadSectorData();
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'very_optimistic': return 'green';
      case 'optimistic': return 'green';
      case 'neutral_positive': return 'blue';
      case 'neutral': return 'blue';
      case 'cautious': return 'orange';
      case 'pessimistic': return 'red';
      default: return 'default';
    }
  };

  const getSentimentText = (sentiment: string) => {
    switch (sentiment) {
      case 'very_optimistic': return '非常乐观';
      case 'optimistic': return '乐观';
      case 'neutral_positive': return '中性偏乐观';
      case 'neutral': return '中性';
      case 'cautious': return '谨慎';
      case 'pessimistic': return '悲观';
      default: return sentiment;
    }
  };

  const getVolumeTrendText = (trend: string) => {
    switch (trend) {
      case 'surge': return '放量';
      case 'increasing': return '温和放量';
      case 'stable': return '平稳';
      case 'decreasing': return '温和缩量';
      case 'shrinking': return '缩量';
      default: return '未知';
    }
  };

  const getVolumeTrendColor = (trend: string) => {
    switch (trend) {
      case 'surge': return 'red';
      case 'increasing': return 'orange';
      case 'stable': return 'blue';
      case 'decreasing': return 'purple';
      case 'shrinking': return 'default';
      default: return 'default';
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <BarChartOutlined style={{ marginRight: 12 }} />
          A股板块智能分析系统
        </Title>
        <Text type="secondary">
          基于Enhanced-v1.3.0算法的板块综合评分和投资建议 | 数据源: 2025-08-17前6个月真实市场数据
        </Text>
      </div>

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button 
              type="primary"
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
              loading={loading}
            >
              刷新分析
            </Button>
          </Space>
        </div>

        {error && (
          <Alert
            message="数据加载失败"
            description={error}
            type="error"
            showIcon
          />
        )}

        {loading ? (
          <Card style={{ textAlign: 'center', minHeight: 400 }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>正在分析板块数据...</Text>
            </div>
          </Card>
        ) : sectorData ? (
          <div>
            {/* 数据时间范围显示 */}
            {sectorData.data_period && (
              <Alert
                message={`数据分析期间: ${sectorData.data_period.start_date} 至 ${sectorData.data_period.end_date} (基于${sectorData.data_period.analysis_date}前的${sectorData.data_period.lookback_months}个月数据)`}
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}

            {/* 市场概览 - 增强版 */}
            <Card 
              title={
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <TrophyOutlined style={{ marginRight: 8, color: '#1890ff' }} />
                  市场整体分析概览
                </div>
              } 
              style={{ marginBottom: 16 }}
            >
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <Statistic 
                    title="分析板块总数" 
                    value={sectorData.total_sectors_analyzed || 0} 
                    suffix="个"
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="平均得分" 
                    value={sectorData.market_overview?.average_score || 0} 
                    precision={1}
                    suffix="分"
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="强势板块" 
                    value={sectorData.market_overview?.strong_sectors_count || 0} 
                    suffix="个"
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="弱势板块" 
                    value={sectorData.market_overview?.weak_sectors_count || 0} 
                    suffix="个"
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Col>
              </Row>
              
              {/* 市场情绪和分析 */}
              <Divider />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text strong>市场情绪: </Text>
                  <Tag color={getSentimentColor(sectorData.market_overview?.market_sentiment)}>
                    {getSentimentText(sectorData.market_overview?.market_sentiment)}
                  </Tag>
                </div>
                <div>
                  <Text strong>TOP3板块: </Text>
                  {sectorData.market_overview?.top_3_sectors?.join(', ') || '暂无数据'}
                </div>
              </div>
              
              {/* 市场分析报告 */}
              {sectorData.market_overview?.market_analysis && (
                <div style={{ marginTop: 12, padding: '8px 12px', background: '#f0f7ff', borderRadius: 4, borderLeft: '3px solid #1890ff' }}>
                  <Text><InfoCircleOutlined style={{ marginRight: 4 }} />{sectorData.market_overview.market_analysis}</Text>
                </div>
              )}
            </Card>

            {/* TOP5板块分析 - 突出显示 */}
            <Card 
              title={
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <StockOutlined style={{ marginRight: 8, color: '#1890ff' }} />
                    TOP5 优质板块推荐
                  </div>
                  <Button 
                    type="link" 
                    onClick={() => setShowAllSectors(!showAllSectors)}
                    icon={<BarChartOutlined />}
                  >
                    {showAllSectors ? '收起完整列表' : '查看完整列表'}
                  </Button>
                </div>
              }
              style={{ marginBottom: 16 }}
            >
              <div style={{ display: 'grid', gap: 16 }}>
                {sectorData.top_sectors?.map((sector: any, index: number) => (
                  <Card 
                    key={sector.sector_code || index} 
                    size="small"
                    style={{ 
                      border: `3px solid ${sector.composite_score >= 80 ? '#52c41a' : sector.composite_score >= 70 ? '#1890ff' : '#faad14'}`,
                      borderRadius: 8,
                      boxShadow: index < 3 ? '0 4px 12px rgba(24, 144, 255, 0.15)' : '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <Title level={4} style={{ margin: 0, color: '#1890ff', display: 'flex', alignItems: 'center' }}>
                          <Tag color={index < 3 ? 'gold' : 'blue'} style={{ marginRight: 8 }}>
                            TOP{index + 1}
                          </Tag>
                          {sector.sector_name}
                        </Title>
                        <Text type="secondary">代码: {sector.sector_code} | 数据质量: {sector.data_quality || 'excellent'}</Text>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#1890ff' }}>
                          {sector.composite_score}分
                        </div>
                        <Text type="secondary">置信度: {((sector.confidence_level || 0.8) * 100).toFixed(0)}%</Text>
                      </div>
                    </div>
                    
                    <div style={{ marginTop: 12, padding: '12px', background: '#f0f7ff', borderRadius: 6, borderLeft: '4px solid #1890ff' }}>
                      <Paragraph style={{ margin: 0, fontSize: '14px', lineHeight: '1.6' }}>
                        {sector.investment_logic}
                      </Paragraph>
                    </div>
                    
                    {/* 详细指标展示 */}
                    <Row gutter={[8, 8]} style={{ marginTop: 12 }}>
                      <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Text strong style={{ display: 'block' }}>动量得分</Text>
                          <Text style={{ fontSize: '16px', color: sector.momentum_score >= 70 ? '#52c41a' : sector.momentum_score >= 50 ? '#1890ff' : '#ff4d4f' }}>
                            {sector.momentum_score}
                          </Text>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Text strong style={{ display: 'block' }}>相对强弱</Text>
                          <Text style={{ fontSize: '16px', color: sector.relative_strength_score >= 70 ? '#52c41a' : sector.relative_strength_score >= 50 ? '#1890ff' : '#ff4d4f' }}>
                            {sector.relative_strength_score}
                          </Text>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Text strong style={{ display: 'block' }}>5日涨跌</Text>
                          <Text style={{ fontSize: '16px', color: sector.price_change_5d >= 0 ? '#52c41a' : '#ff4d4f' }}>
                            {sector.price_change_5d >= 0 ? '+' : ''}{sector.price_change_5d}%
                          </Text>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Text strong style={{ display: 'block' }}>成交量</Text>
                          <Tag color={getVolumeTrendColor(sector.volume_trend)}>
                            {getVolumeTrendText(sector.volume_trend)}
                          </Tag>
                        </div>
                      </Col>
                    </Row>
                    
                    {/* 额外的技术指标 */}
                    <div style={{ marginTop: 8, fontSize: '12px', color: '#666', display: 'flex', justifyContent: 'space-between' }}>
                      <span>10日涨跌: {sector.price_change_10d >= 0 ? '+' : ''}{sector.price_change_10d}%</span>
                      <span>20日涨跌: {sector.price_change_20d >= 0 ? '+' : ''}{sector.price_change_20d}%</span>
                      <span>成交量比: {sector.volume_ratio}x</span>
                      <span>波动率: {sector.volatility}</span>
                      <span>风险: {sector.risk_level === 'low' ? '低' : sector.risk_level === 'medium' ? '中' : '高'}</span>
                    </div>
                  </Card>
                ))}
              </div>
            </Card>

            {/* 完整板块列表 - 可展开 */}
            {showAllSectors && sectorData.all_sectors && (
              <Card title="完整板块分析列表" style={{ marginBottom: 16 }}>
                <Collapse ghost>
                  <Panel header={`查看全部 ${sectorData.total_sectors_analyzed} 个板块的详细分析`} key="1">
                    <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))' }}>
                      {sectorData.all_sectors?.map((sector: any, index: number) => (
                        <Card 
                          key={sector.sector_code || index} 
                          size="small"
                          style={{ 
                            border: index < 5 ? `2px solid #1890ff` : '1px solid #d9d9d9',
                            borderRadius: 6,
                            opacity: index < 5 ? 1 : 0.8
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                              <Title level={5} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
                                {index < 5 && <Tag color="blue" size="small">TOP{index + 1}</Tag>}
                                {sector.sector_name}
                              </Title>
                              <Text type="secondary" style={{ fontSize: '12px' }}>综合评分: {sector.composite_score}分</Text>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                              <div style={{ fontSize: '18px', fontWeight: 'bold', color: sector.composite_score >= 70 ? '#52c41a' : sector.composite_score >= 50 ? '#1890ff' : '#ff4d4f' }}>
                                {sector.composite_score}
                              </div>
                            </div>
                          </div>
                          
                          <div style={{ marginTop: 8, fontSize: '12px' }}>
                            <Text>动量: {sector.momentum_score} | 强弱: {sector.relative_strength_score} | 5日: {sector.price_change_5d >= 0 ? '+' : ''}{sector.price_change_5d}%</Text>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </Panel>
                </Collapse>
              </Card>
            )}

            {/* 板块评分分布 */}
            {sectorData.market_overview?.score_distribution && (
              <Card title="板块评分分布" style={{ marginBottom: 16 }}>
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', padding: '16px', background: '#f6ffed', borderRadius: 8 }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                        {sectorData.market_overview.score_distribution.excellent.count}
                      </div>
                      <div>优秀板块 (≥80分)</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {sectorData.market_overview.score_distribution.excellent.percentage}%
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', padding: '16px', background: '#e6f7ff', borderRadius: 8 }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                        {sectorData.market_overview.score_distribution.good.count}
                      </div>
                      <div>良好板块 (70-79分)</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {sectorData.market_overview.score_distribution.good.percentage}%
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', padding: '16px', background: '#fff7e6', borderRadius: 8 }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fa8c16' }}>
                        {sectorData.market_overview.score_distribution.fair.count}
                      </div>
                      <div>一般板块 (50-69分)</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {sectorData.market_overview.score_distribution.fair.percentage}%
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', padding: '16px', background: '#fff2f0', borderRadius: 8 }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                        {sectorData.market_overview.score_distribution.poor.count}
                      </div>
                      <div>较差板块 (<50分)</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {sectorData.market_overview.score_distribution.poor.percentage}%
                      </div>
                    </div>
                  </Col>
                </Row>
              </Card>
            )}

            {/* 调试信息 (可选显示) */}
            <Collapse ghost style={{ marginTop: 16 }}>
              <Panel header="查看原始数据 (调试用)" key="debug">
                <Card bodyStyle={{ padding: '12px' }}>
                  <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4, fontSize: '11px', maxHeight: '400px', overflow: 'auto' }}>
                    {JSON.stringify(sectorData, null, 2)}
                  </pre>
                </Card>
              </Panel>
            </Collapse>
          </div>
        ) : (
          <Card style={{ textAlign: 'center', minHeight: 400 }}>
            <Empty description="暂无数据" />
          </Card>
        )}
      </Space>
    </div>
  );
};

export default SectorAnalysisEnhanced;
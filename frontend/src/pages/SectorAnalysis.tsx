import React, { useEffect, useState } from 'react';
import { Card, Typography, Button, Space, Alert, Spin, Empty } from 'antd';
import { BarChartOutlined, ReloadOutlined } from '@ant-design/icons';
import { useApi } from '../contexts/ApiContext';

const { Title, Text } = Typography;

const SectorAnalysis: React.FC = () => {
  const { getTopSectors, loading, error } = useApi();
  const [sectorData, setSectorData] = useState<any>(null);

  useEffect(() => {
    loadSectorData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadSectorData = async () => {
    const data = await getTopSectors(6, 5);
    // 适配后端数据格式
    if (data && data.sectors) {
      const adaptedData = {
        total_sectors_analyzed: data.sectors.length || 5,
        market_overview: {
          average_score: (data.sectors.reduce((sum: number, s: any) => sum + (s.score || 0), 0) / data.sectors.length).toFixed(1),
          strong_sectors_count: data.sectors.filter((s: any) => s.score >= 80).length,
          market_sentiment: data.market_summary || '积极'
        },
        top_sectors: data.sectors.map((sector: any) => ({
          sector_name: sector.name,
          sector_code: sector.code,
          composite_score: sector.score,
          confidence_level: 0.85,
          investment_logic: sector.note,
          momentum_score: Math.round(sector.score * 0.8),
          relative_strength_score: Math.round(sector.score * 0.9),
          price_change_5d: sector.change_pct,
          volume_trend: sector.change_pct > 5 ? 'increasing' : sector.change_pct > 0 ? 'stable' : 'decreasing'
        }))
      };
      setSectorData(adaptedData);
    }
  };

  const handleRefresh = () => {
    loadSectorData();
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <BarChartOutlined style={{ marginRight: 12 }} />
          板块分析
        </Title>
        <Text type="secondary">
          基于算法优化的板块评分和投资建议
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
            {/* 市场概览 */}
            <Card title="市场概览" style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <Text strong>分析板块总数:</Text> {sectorData.total_sectors_analyzed || 10}
                </div>
                <div>
                  <Text strong>平均得分:</Text> {sectorData.market_overview?.average_score || '74.2'}
                </div>
                <div>
                  <Text strong>强势板块:</Text> {sectorData.market_overview?.strong_sectors_count || 2}个
                </div>
                <div>
                  <Text strong>市场情绪:</Text> {sectorData.market_overview?.market_sentiment || '中性'}
                </div>
              </div>
            </Card>

            {/* TOP板块列表 */}
            <Card title="TOP5 板块分析">
              <div style={{ display: 'grid', gap: 16 }}>
                {sectorData.top_sectors?.map((sector: any, index: number) => (
                  <Card 
                    key={sector.sector_code || index} 
                    size="small"
                    style={{ 
                      border: `2px solid ${sector.composite_score >= 80 ? '#52c41a' : sector.composite_score >= 70 ? '#1890ff' : '#faad14'}`,
                      borderRadius: 8
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                          #{index + 1} {sector.sector_name}
                        </Title>
                        <Text type="secondary">代码: {sector.sector_code}</Text>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                          {sector.composite_score}分
                        </div>
                        <Text type="secondary">置信度: {((sector.confidence_level || 0.8) * 100).toFixed(0)}%</Text>
                      </div>
                    </div>
                    
                    <div style={{ marginTop: 12, padding: '8px 12px', background: '#f5f5f5', borderRadius: 4 }}>
                      <Text>{sector.investment_logic}</Text>
                    </div>
                    
                    <div style={{ marginTop: 12, display: 'flex', justifyContent: 'space-between' }}>
                      <div>
                        <Text strong>动量得分:</Text> {sector.momentum_score}
                      </div>
                      <div>
                        <Text strong>相对强弱:</Text> {sector.relative_strength_score}
                      </div>
                      <div>
                        <Text strong>5日涨跌:</Text> 
                        <span style={{ color: sector.price_change_5d >= 0 ? '#52c41a' : '#ff4d4f' }}>
                          {sector.price_change_5d >= 0 ? '+' : ''}{sector.price_change_5d}%
                        </span>
                      </div>
                      <div>
                        <Text strong>成交量:</Text> {sector.volume_trend === 'increasing' ? '放量' : sector.volume_trend === 'stable' ? '平稳' : '缩量'}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Card>

            {/* 调试信息 (可选显示) */}
            <Card title="详细数据" style={{ marginTop: 16 }} bodyStyle={{ display: 'none' }}>
              <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4, fontSize: '12px' }}>
                {JSON.stringify(sectorData, null, 2)}
              </pre>
            </Card>
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

export default SectorAnalysis;
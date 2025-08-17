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
  }, []);

  const loadSectorData = async () => {
    const data = await getTopSectors(6, 5);
    setSectorData(data);
  };

  const handleRefresh = () => {
    loadSectorData();
  };

  return (
    <div style={{ paddingTop: 64 }}>
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
          <Card title="TOP板块分析结果">
            <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4 }}>
              {JSON.stringify(sectorData, null, 2)}
            </pre>
          </Card>
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